# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Dict, Optional, Tuple, Union
from typing_extensions import Literal

from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_normal
from nerfview import CameraState
from pyquaternion import Quaternion
from dataclasses import fields

from TASE.splatting.common.background_model import MLP_BackgroundModel
from TASE.splatting.trainer.trainer_cfg import Config
from TASE.splatting.util.data_util import resize_intrinsics, resolve_path

# Remove this import? Not neccessary for TASE only.
from TASE.splatting.util.data_util import *

import math
import yaml
from pathlib import Path
import numpy as np

from gsplat.strategy.default import DefaultStrategy


def default_strategy_constructor(loader, node):
    # Optionally, extract data from the node using loader.construct_mapping(node)
    return DefaultStrategy()


yaml.add_constructor(
    "tag:yaml.org,2002:python/object:gsplat.strategy.default.DefaultStrategy",
    default_strategy_constructor,
    Loader=yaml.SafeLoader,
)


# Tell PyYAML how to deserialize
def path_constructor(loader, node):
    # node.value is a sequence like ["home", "user", "project"]
    parts = loader.construct_sequence(node)
    return Path(*parts)


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    path_constructor,
    Loader=yaml.SafeLoader,
)


class ColorMapper:
    """
    Util class that can be used to create a colormapping from a 1D image like a depthmap with pure python.
    """

    colormap = None
    device = "cuda"
    vmin = None
    vmax = None

    def __init__(self, reverse=True, device="cuda"):

        self.device = device
        self.colormap = self.generate_inferno_colormap(reverse=reverse).to(device)

    def generate_inferno_colormap(self, n=256, reverse=False):
        # Define anchor colors (RGB) and their positions (from 0 to 1)
        anchors = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # black at position 0.0
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.4],  # blue at about position 0.3
                [0.1, 0.0, 0.5],
                [0.2, 0.0, 0.55],
                [0.4, 0.0, 0.6],  # puple at about position 0.6
                [0.7, 0.2, 0.3],
                [1.0, 0.5, 0.0],  # orange at about position 0.8
                [1.0, 1.0, 0.0],  # yellow at about position 0.9
                [1.0, 1.0, 1.0],  # white at position 1.0
            ]
        )
        positions = torch.linspace(0, 1, anchors.shape[0])

        # Generate n evenly spaced positions between 0 and 1 for the colormap
        x = torch.linspace(0, 1, n)
        colormap = torch.zeros(n, 3)

        for c in range(3):
            # For each channel, determine the lower anchor indices for each x value
            idx = torch.searchsorted(positions, x, right=True) - 1
            # Clamp indices to valid range
            idx = torch.clamp(idx, 0, positions.numel() - 2)

            lower_positions = positions[idx]
            upper_positions = positions[idx + 1]
            # Compute the interpolation factor
            t = (x - lower_positions) / (upper_positions - lower_positions)

            # Interpolate between the corresponding anchor values
            lower_vals = anchors[:, c][idx]
            upper_vals = anchors[:, c][idx + 1]
            colormap[:, c] = lower_vals + t * (upper_vals - lower_vals)

        if reverse:
            colormap = colormap.flip(0)
        return colormap

    def apply_colormap(
        self,
        depth: torch.Tensor | np.ndarray,
        alpha: torch.Tensor | np.ndarray = None,
        vmin=None,
        vmax=None,
    ):
        """
        Applies a colormap to a single-channel depth image or batch of depth images.

        This function is flexible to accept depthmaps with the following shapes:
        - Single image: (H, W) or (H, W, 1)
        - Batch of images: (B, H, W) or (B, H, W, 1)

        Args:
            depth (torch.Tensor or np.ndarray): Depth values.
            alpha (torch.Tensor or np.ndarray): Optional alpha mask.
            vmin, vmax: Optional values for min and max normalization.

        Returns:
            torch.Tensor or np.ndarray: Color-mapped image(s) of shape (H, W, 3) or (B, H, W, 3) with RGB values.
        """
        # If depth is a numpy array, convert to tensor and mark for re-conversion later.
        to_numpy = False
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).to(self.device)
            to_numpy = True

        # Process alpha if provided and if a numpy array.
        if alpha is not None:
            if isinstance(alpha, np.ndarray):
                alpha = torch.from_numpy(alpha).to(self.device)

        # --- Adjust shapes for depth ---
        # If depth has a channel dimension of size 1 at the end, squeeze it.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)  # Now either (H, W) or (B, H, W)
        elif depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)  # Now (B, H, W)

        # Similarly, adjust alpha dimensions if needed.
        if alpha is not None:
            if alpha.ndim == 3 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)
            elif alpha.ndim == 4 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)

        # Initialize colormap if not already done.
        if self.colormap is None:
            colormap = self.generate_inferno_colormap(reverse=True)
            self.colormap = colormap.to(depth.device)

        # Determine vmin and vmax, either from provided values or from the depth tensor.
        if vmin is None:
            if hasattr(self, "vmin") and self.vmin is not None:
                vmin = self.vmin
            else:
                vmin = depth.min()
        if vmax is None:
            if hasattr(self, "vmax") and self.vmax is not None:
                vmax = self.vmax
            else:
                vmax = depth.max()

        # Normalize depth values to [0, 1]
        depth_norm = (depth - vmin) / (vmax - vmin)
        depth_norm = torch.clamp(depth_norm, 0, 1)

        # Scale normalized depth to colormap indices.
        num_colors = self.colormap.shape[0]
        indices = (depth_norm * (num_colors - 1)).long()
        indices = torch.clamp(indices, 0, self.colormap.shape[0] - 1)

        # Map indices to colors.
        # If there is a batch dimension, this results in (B, H, W, 3);
        # if no batch, then (H, W, 3).
        colored_depth = self.colormap[indices]

        # Optionally apply alpha blending with a background color.
        if alpha is not None:
            # Make sure the background color tensor is on the correct device.
            bg_color = torch.Tensor([0.0, 1.0, 0.0])[None, None, :].to(
                depth.device
            )  # Green background
            # Expand bg_color if a batch dimension exists.
            if colored_depth.ndim == 4:
                bg_color = bg_color.expand(colored_depth.shape[0], -1, -1, -1)
            # Adjust blending: (1-alpha)*background + alpha*colored_depth.
            # We ensure alpha has an extra dimension to match the RGB channels.
            colored_depth = (1 - alpha[..., None]) * bg_color + alpha[
                ..., None
            ] * colored_depth

        # Convert back to numpy if the original input was a numpy array.
        if to_numpy:
            colored_depth = colored_depth.cpu().numpy()
        return colored_depth


class Scene:
    ckpt_dir = None
    semantic_pc = None
    camera_poses: List[CameraState] = []
    splats: torch.nn.ParameterDict = None
    frozen_splats: torch.nn.ParameterDict = None
    sh_degree: int = 0
    background_model: MLP_BackgroundModel = None
    feature_bg_model: MLP_BackgroundModel = None
    cfg: Config = None
    device = None
    depth_colormapper = None
    checkpoint_path = None
    translation: np.ndarray = None
    rotation_wxyz: np.ndarray = None
    cur_pan: float = (
        math.pi  # to allow for continous panning during object centric sampling
    )

    def __init__(
        self, cfg: Optional[Config] = None, device: str = "cuda", verbose: bool = False
    ):

        self.cfg = cfg if cfg is not None else Config()
        self.device = device
        self.name = cfg.scene if cfg else "UnnamedScene"

        self._rng = torch.Generator(device="cpu")
        self.depth_colormapper = ColorMapper(device=self.device)

    def split_frozen_splats(self, pos_and_zero=False):
        if self.frozen_splats is not None:
            self.merge_frozen_splats()

        selection = self.splats["selection"]  # [N, 1]
        if pos_and_zero:
            mask = selection > -0.5
        else:
            mask = selection > 0.5  # [N, 1]

        frozen_splats = {}
        for k in self.splats:
            t_frozen = self.splats[k][~mask.squeeze(-1)].detach().clone()
            t_active = self.splats[k][mask.squeeze(-1)].detach().clone()
            self.splats[k] = torch.nn.Parameter(t_active, requires_grad=True)
            frozen_splats[k] = torch.nn.Parameter(t_frozen, requires_grad=False)
        self.frozen_splats = torch.nn.ParameterDict(frozen_splats)

    def merge_frozen_splats(self):
        if self.frozen_splats is not None:
            n_frozen = self.frozen_splats["means"].shape[0]
            for k in self.frozen_splats:
                if k in self.splats:
                    t1 = self.splats[k].detach().clone()
                    t2 = self.frozen_splats[k].detach().clone()
                    t = torch.cat([t1, t2], dim=0)
                    self.splats[k] = torch.nn.Parameter(t, requires_grad=True)
            self.frozen_splats = None
            print(
                f"[Scene] Merged {n_frozen} frozen splats into active splats. Now {self.splats['means'].shape[0]} splats."
            )

        self.frozen_splats = None

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        background_color: Optional[Tensor] = None,
        background_mode: Optional[Literal["fg", "bg", "fg+bg"]] = "fg+bg",
        color_mode: Literal["SH", "SemSeg", "Features", "Selection"] = "SH",
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sh_degree: Optional[int] = None,
        max_feature_dim: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:

        if background_color is not None:
            background_mode = "fg"
            if (
                isinstance(background_color, list)
                or isinstance(background_color, np.ndarray)
                or isinstance(background_color, tuple)
            ):
                background_color = torch.tensor(
                    background_color, device=self.device, dtype=torch.float32
                )


        if background_mode == "fg" and background_color is None:
            background_color = torch.tensor(
                [1.0, 1.0, 1.0], device=self.device, dtype=torch.float32
            )

        if background_mode == "fg" or background_mode == "fg+bg":
            means = self.splats["means"]  # [N, 3]
            quats = self.splats["quats"]  # [N, 4]
            scales = torch.exp(self.splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        else:
            means = self.frozen_splats["means"]  # [N, 3]
            quats = self.frozen_splats["quats"]  # [N, 4]
            scales = torch.exp(self.frozen_splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.frozen_splats["opacities"])  # [N,]

        if background_mode == "fg+bg" and self.frozen_splats is not None:
            means = torch.cat([means, self.frozen_splats["means"]], dim=0)
            quats = torch.cat([quats, self.frozen_splats["quats"]], dim=0)
            scales = torch.cat([scales, torch.exp(self.frozen_splats["scales"])], dim=0)
            opacities = torch.cat(
                [opacities, torch.sigmoid(self.frozen_splats["opacities"])], dim=0
            )

        if color_mode == "SH":
            if background_mode == "fg" or background_mode == "fg+bg":
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]
            else:
                sh0 = self.frozen_splats["sh0"]
                shN = self.frozen_splats["shN"]
            if background_mode == "fg+bg" and self.frozen_splats is not None:
                sh0 = torch.cat([sh0, self.frozen_splats["sh0"]], dim=0)
                shN = torch.cat([shN, self.frozen_splats["shN"]], dim=0)

            colors = torch.cat([sh0, shN], dim=1)
            sh_degree = (
                sh_degree
                if sh_degree is not None
                else int(math.sqrt(colors.shape[-2]) - 1)
            )
        elif color_mode == "SemSeg":
            if background_mode == "fg" or background_mode == "fg+bg":
                colors = self.splats["classes"]
            else:
                colors = self.frozen_splats["classes"]
            if background_mode == "fg+bg" and self.frozen_splats is not None:
                colors = torch.cat([colors, self.frozen_splats["classes"]], dim=0)
            sh_degree = None
        elif color_mode == "Features":
            if background_mode == "fg" or background_mode == "fg+bg":
                colors = self.splats["features"]
            else:
                colors = self.frozen_splats["features"]
            if background_mode == "fg+bg" and self.frozen_splats is not None:
                colors = torch.cat([colors, self.frozen_splats["features"]], dim=0)
            colors = (
                colors[..., :max_feature_dim] if max_feature_dim is not None else colors
            )
            sh_degree = None
        elif color_mode == "Selection":
            if background_mode == "fg" or background_mode == "fg+bg":
                colors = self.splats["selection"]
            else:
                colors = self.frozen_splats["selection"]
            if background_mode == "fg+bg" and self.frozen_splats is not None:
                colors = torch.cat([colors, self.frozen_splats["selection"]], dim=0)
            sh_degree = None

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        # rasterize the splats
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.strategy.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=self.cfg.camera_model,
            render_mode=render_mode,
            sh_degree=sh_degree,
            **kwargs,
        )

        if self.frozen_splats is not None:
            info["n_active"] = self.splats["means"].shape[0]
        else:
            info["n_active"] = None

        # get the normals from the depth
        if render_mode == "ED" or render_mode == "RGB+ED":
            # get the depth map
            if render_mode == "ED":
                render_depth = render_colors[..., :1]
            else:
                render_depth = render_colors[..., 3:4]
            worldtocam = camtoworlds[..., :3, :3].transpose(-1, -2)
            render_normals_from_depth = depth_to_normal(
                render_depth.clone(),
                camtoworlds=camtoworlds,
                Ks=Ks,
            )
            render_normals = torch.einsum(
                "...ij,...hwj->...hwi", worldtocam, render_normals_from_depth
            )
        else:
            render_normals = None

        # render and add the background
        if background_color is None and color_mode == "Features" and self.feature_bg_model is not None:
            backgrounds = self.feature_bg_model.render_bg_colors(
                Ks=Ks, extrinsics=camtoworlds, height=height, width=width
            )[..., : render_colors.shape[-1]]
            render_colors[..., : render_colors.shape[-1]] = render_colors[
                ..., : render_colors.shape[-1]
            ] + backgrounds * (1 - render_alphas)
        elif background_color is not None and render_colors.shape[3] >= 3:
            render_colors[..., :3] = render_colors[..., :3] + background_color[
                None, None, None, :3
            ] * (1 - render_alphas)
        elif self.background_model is not None and render_colors.shape[3] >= 3:
            backgrounds = self.background_model.render_bg_colors(
                Ks=Ks, extrinsics=camtoworlds, height=height, width=width
            )
            render_colors[..., :3] = render_colors[..., :3] + backgrounds * (
                1 - render_alphas
            )
        if masks is not None:
            # expand the mask to the same size as render_colors
            expanded_masks = masks.expand_as(render_colors).bool()
            render_colors[~expanded_masks] = 0
        return render_colors, render_alphas, render_normals, info

    def render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        raw: bool = False,
        background_color: Optional[Tensor] = None,
        alpha: bool = False,
        **kwargs,
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, _, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.sh_degree,  # active all SH degrees
            render_mode="RGB",
            background_color=background_color,
            **kwargs,
        )  # [1, H, W, 3]

        if raw:
            # return the raw colors
            return render_colors, info

        if alpha:
            render_colors = torch.cat(
                [render_colors, render_alphas], dim=-1
            )  # [1, H, W, 4]

        return render_colors[0].cpu().numpy()

    def _dump_poses(self, save_path: str, c2ws: List[np.ndarray]):
        cam_poses_path = save_path + "_cam_poses.npy"
        np.save(cam_poses_path, np.array(c2ws))
        # also save the scene transform
        if self.translation is None or self.rotation_wxyz is None:
            print(
                "[Scene] Scene translation or rotation is None. Saving identity transform."
            )
            self.translation = np.array([0.0, 0.0, 0.0])
            self.rotation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        scene_trafo_matrix = Quaternion(self.rotation_wxyz).transformation_matrix
        scene_trafo_matrix[:3, 3] = self.translation
        scene_transform = {
            "translation": self.translation,
            "rotation_wxyz": self.rotation_wxyz,
            "matrix": scene_trafo_matrix,
        }
        scene_trafo_path = save_path + "_scene_transform.npy"
        np.save(scene_trafo_path, scene_transform)

    def render_images(
        self,
        save_path: str,
        c2ws: List[np.ndarray],
        img_wh: Tuple[int, int],
        fov: Optional[float] = 0.7,
        depth: bool = False,
        **kwargs,
    ):

        W, H = img_wh

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self._dump_poses(save_path, c2ws)

        render_mode = "RGB+ED" if depth else "RGB"

        if depth and self.depth_colormapper is None:
            self.depth_colormapper = ColorMapper(device=self.device)

        for i, c2w in enumerate(c2ws):
            camera_state = CameraState(
                c2w=c2w,
                aspect=1.0,
                fov=fov,
            )
            render_colors, render_alphas, _, info = self.rasterize_splats(
                camtoworlds=torch.from_numpy(c2w).float().to(self.device)[None],
                Ks=torch.from_numpy(camera_state.get_K(img_wh))
                .float()
                .to(self.device)[None],
                width=W,
                height=H,
                sh_degree=self.sh_degree,
                render_mode=render_mode,
                **kwargs,
            )  # [1, H, W, 3 or 4]
            if depth:
                depth_map = render_colors[0, :, :, 3].detach().cpu().numpy()
                depth_colormap = self.depth_colormapper.apply_colormap(depth_map) * 255
                depth_colormap = depth_colormap.astype(np.uint8)
                depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path + f"depth_{i:03d}.png", depth_colormap)
            render_colors = render_colors[0, :, :, :3]
            render_colors = torch.clamp(render_colors, 0.0, 1.0)
            render_colors = (render_colors.detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            render_colors = cv2.cvtColor(render_colors, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path + f"rgb_{i:03d}.png", render_colors)

    @torch.no_grad()
    def render_video(
        self,
        save_path: str,
        c2ws: List[np.ndarray],
        img_wh: Tuple[int, int],
        fov: Optional[float] = 0.7,
        depth: bool = False,
        **kwargs,
    ):
        import imageio

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # dump the poses
        self._dump_poses(save_path, c2ws)

        W, H = img_wh
        video_writer = imageio.get_writer(save_path + ".mp4", fps=30)
        depth_video_writer = (
            imageio.get_writer(save_path + "_depth.mp4", fps=30) if depth else None
        )

        render_mode = "RGB+ED" if depth else "RGB"

        if depth and self.depth_colormapper is None:
            self.depth_colormapper = ColorMapper(device=self.device)

        for c2w in c2ws:
            camera_state = CameraState(
                c2w=c2w,
                aspect=1.0,
                fov=fov,
            )
            render_colors, render_alphas, _, info = self.rasterize_splats(
                camtoworlds=torch.from_numpy(c2w).float().to(self.device)[None],
                Ks=torch.from_numpy(camera_state.get_K(img_wh))
                .float()
                .to(self.device)[None],
                width=W,
                height=H,
                sh_degree=self.sh_degree,
                render_mode=render_mode,
                **kwargs,
            )  # [1, H, W, 3 or 4]
            if depth:
                depth_map = render_colors[0, :, :, 3].detach().cpu().numpy()
                depth_colormap = self.depth_colormapper.apply_colormap(depth_map) * 255
                depth_colormap = depth_colormap.astype(np.uint8)
                depth_video_writer.append_data(depth_colormap)
            render_colors = render_colors[0, :, :, :3]
            render_colors = torch.clamp(render_colors, 0.0, 1.0)
            render_colors = (render_colors.detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            video_writer.append_data(render_colors)
        video_writer.close()


    def dummy_render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        **kwargs,
    ):
        """A dummy render function that does nothing."""
        W, H = img_wh
        # create a black image
        img = np.ones((H, W, 3), dtype=np.uint8)
        if self.semantic_colormap:
            bg_color = self.semantic_colormap.get(
                -1, (128, 128, 128)
            )  # Default background color if not found
            img *= np.array(bg_color, dtype=np.uint8)
        return img

    def depth_render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        raw: bool = False,
        **kwargs,
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alpha, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            render_mode="ED",
            **kwargs,
        )  # [1, H, W, 1]

        if raw:
            # return the raw depth values
            depth = render_colors[0, :, :, 0]
            alpha = render_alpha[0, :, :, 0]
            return depth.cpu().numpy(), alpha.cpu().numpy()

        # colormap the the depth values
        depth = render_colors[0, :, :, 0]
        alpha = render_alpha[0, :, :, 0]
        colormap = self.depth_colormapper.apply_colormap(depth, alpha)

        return colormap.cpu().numpy()

    def normal_render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        raw: bool = False,
        **kwargs,
    ):
        # Uses normals from depth, not from splats!
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alpha, render_normals, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            render_mode="ED",
            **kwargs,
        )  # [1, H, W, 1]

        if raw:
            # return the raw normals
            normals = render_normals[0, :, :, :3].cpu().numpy()

        # colormap the the depth values
        normals = (render_normals[0, :, :, :3] + 1.0) / 2.0
        alpha = render_alpha[0, :, :, 0]

        # alpha blend the normals with a black background color
        normals = normals * alpha[..., None] + (1 - alpha[..., None]) * torch.tensor(
            [0, 0, 0]
        ).to(normals.device)

        return normals.cpu().numpy()

    def feature_render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        raw: bool = False,
        downsample_factor: int = None,
        **kwargs,
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        if raw:
            max_feature_dim = None  # return all features
        else:
            max_feature_dim = 3  # only return the first 3 features for visualization

        if raw and downsample_factor is not None:
            # if raw output is requested, the resolution is set so that the rendered image can be used as a control map without interpolation
            # it is assumed here, that the CtrlNet takes in maps that have a downsample factor of 8
            feat_W, feat_H = W // downsample_factor, H // downsample_factor
            scale_x = feat_W / W
            scale_y = feat_H / H
            K = resize_intrinsics(
                K=K,
                scale=(scale_x, scale_y),
            )
            W, H = feat_W, feat_H

        render_colors, alphas, normals, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
            color_mode="Features",
            max_feature_dim=max_feature_dim,
            **kwargs,
        )

        if raw:
            # return the raw features
            return render_colors

        # [1, H, W, C]
        features = render_colors[..., :3]  # take the first 3 channels
        features = norm_channel_wise(
            features
        )  # normalize the features across channels to [0, 1]
        # features = features / 2.0 + 0.5  # normalize the features to [0, 1]

        return features[0].cpu().numpy()

    def selection_render_fn(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        raw: bool = False,
        discretize: bool = False,
        **kwargs,
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, alphas, normals, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
            color_mode="Selection",
            **kwargs,
        )

        if raw:
            # return the raw selection values
            return render_colors

        # overlay the selection on the background
        if discretize:
            pos_mul = (render_colors > 0.5).float()
            neg_mul = (render_colors < -0.5).float()
        else:
            pos_mul = render_colors.clamp(0, 1)
            neg_mul = render_colors.clamp(-1, 0).abs()
        overlay = pos_mul * torch.tensor(
            [0.0, 1.0, 0.0], device=render_colors.device
        ) + neg_mul * torch.tensor([1.0, 0.0, 0.0], device=render_colors.device)

        return overlay[0].cpu().numpy()

    @torch.no_grad()
    def renormalize_splats(self, z_up=False):
        center, axes = self.compute_bounding_ellipsoid(
            selected_only=False, percentile=0.9, enlarge_factor=1.0
        )
        # shortest axis is Z
        lengths = axes.norm(dim=1)
        z_axis = axes[torch.argmin(lengths)]
        # Z will usually point down since the origin is the center of mass -> more splats for the ground
        z_axis = z_axis / z_axis.norm()
        if not z_up:
            z_axis = -z_axis
        # X is the longest axis
        x_axis = axes[torch.argmax(lengths)]
        x_axis = x_axis - (x_axis @ z_axis) * z_axis
        x_axis = x_axis / x_axis.norm()
        # Y is the remaining axis
        y_axis = torch.cross(z_axis, x_axis, dim=0)
        y_axis = y_axis / y_axis.norm()
        R_tensor = torch.stack([x_axis, y_axis, z_axis], dim=1).T
        atoll = 1e-7
        rtoll = 1e-4
        R_quat = None
        # get a pyquaternion from the rotation matrix with increasing tolerance if it fails
        while rtoll < 1.0:
            try:
                R_quat = Quaternion(
                    matrix=R_tensor.cpu().numpy(), atol=atoll, rtol=rtoll
                )
                break
            except Exception as e:
                rtoll *= 10.0
                atoll *= 10.0
                print(
                    f"Warning: Increasing tolerance to convert rotation matrix to quaternion: atol={atoll}, rtol={rtoll}. Error: {e}"
                )
        if R_quat is None:
            raise RuntimeError("Failed to convert rotation matrix to quaternion.")

        self.apply_scene_transform(center, R_quat)

    def apply_scene_transform(
        self,
        t_vec: torch.Tensor = None,
        R_quat: Quaternion = None,
        matrix: np.ndarray = None,
    ) -> Tensor:
        means = self.means
        quats = self.quats
        if matrix is not None:
            R_quat = Quaternion(matrix=matrix[:3, :3], rtol=1e-04, atol=1e-07)
            t_vec = torch.tensor(matrix[:3, 3], device=means.device, dtype=means.dtype)
        R_tensor = torch.tensor(
            R_quat.rotation_matrix, device=means.device, dtype=means.dtype
        )
        t_tensor = t_vec.to(means.device, dtype=means.dtype)
        # Transform the means
        means = (R_tensor @ means.T).T
        means = means - t_tensor
        self.splats["means"] = means

        # Rotate the quaternions
        def quat_mul(q1, q2):
            # q1: (...,4), q2: (...,4) in (w,x,y,z)
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            return torch.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dim=-1,
            )

        # quats: (N,4)
        # rot: (4,) single quaternion
        def apply_global_rotation(quats, rot, left=True):
            rot = rot / rot.norm()
            if left:
                # q' = rot * q
                return quat_mul(rot.expand_as(quats), quats)
            else:
                # q' = q * rot
                return quat_mul(quats, rot.expand_as(quats))

        R_quat_tensor = torch.tensor(
            [R_quat.w, R_quat.x, R_quat.y, R_quat.z],
            device=self.device,
            dtype=quats.dtype,
        )
        quats = apply_global_rotation(quats, R_quat_tensor)
        self.splats["quats"] = quats
        # No scale change for now
        # Set the scene transform for reference (eg. camera transformation if a dataset is loaded)
        self.translation = t_tensor.cpu().numpy()
        self.rotation_wxyz = R_quat.elements

    def store_original_geometry(self):
        if "original_means" not in self.splats:
            self.splats["original_means"] = torch.nn.Parameter(
                self.splats["means"].detach().clone().requires_grad_(False),
                requires_grad=False,
            )
        if "original_quats" not in self.splats:
            self.splats["original_quats"] = torch.nn.Parameter(
                self.splats["quats"].detach().clone().requires_grad_(False),
                requires_grad=False,
            )
        if "original_scales" not in self.splats:
            self.splats["original_scales"] = torch.nn.Parameter(
                self.splats["scales"].detach().clone().requires_grad_(False),
                requires_grad=False,
            )
        if "original_opacities" not in self.splats:
            self.splats["original_opacities"] = torch.nn.Parameter(
                self.splats["opacities"].detach().clone().requires_grad_(False),
                requires_grad=False,
            )

    def get_scene_extent(self):
        center, axes = self.compute_bounding_ellipsoid(
            selected_only=False, percentile=0.9, enlarge_factor=1.0
        )
        lengths = axes.norm(dim=1)
        max_extent = lengths.max().item()
        return center.cpu().numpy(), max_extent

    def export_to_ply(self, ply_path: str):
        from gsplat.utils import save_ply

        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        save_ply(self.splats, ply_path)

    @classmethod
    def load_from_ckpt(
        cls,
        ckpt_path,
        require_grad=False,
        device="cuda",
        renormalize_coordinates=False,
        flip_z=False,
        scene_trafo: Optional[Tuple[torch.Tensor, Quaternion]] = None,
        dataparser_trafo: Optional[Tuple[np.ndarray, float]] = None,
        z_up=False,
    ):
        ckpt_dir = resolve_path(ckpt_path)
        if not ckpt_dir.exists():
            print(f"Checkpoint path {ckpt_dir} does not exist.")
            return None

        if ckpt_dir.suffix == ".ply":
            # use load from ply instead
            scene = cls.from_ply(
                ply_path=ckpt_dir, require_grad=require_grad, device=device
            )
        else:
            scene = cls.from_pt(
                pt_path=ckpt_dir, require_grad=require_grad, device=device
            )

        scene.ckpt_dir = ckpt_dir

        try:
            scene.load_config()
        except Exception as e:
            print(f"Failed to load config for checkpoint: {e}")

        if (
            renormalize_coordinates
            and scene_trafo is None
            and (scene.rotation_wxyz is None or scene.translation is None)
        ):
            scene.renormalize_splats(z_up=z_up)
        elif scene_trafo is not None:
            t_vec, R_quat = scene_trafo
            scene.apply_scene_transform(t_vec, R_quat)
        elif dataparser_trafo is not None:
            T, scale = dataparser_trafo
            # apply the transform
            scene.apply_scene_transform(matrix=T)
            # scale the splats
            scene.splats["means"] = scene.splats["means"] / float(scale)
            scales = torch.exp(scene.splats["scales"])  # [N, 3]
            scaled_scales = scales / float(scale)
            scene.splats["scales"] = torch.log(scaled_scales)

        if flip_z:
            # rotate 180 degrees around the X axis to flip the Z axis
            scene.apply_scene_transform(
                matrix=np.diag([1.0, -1.0, -1.0, 1.0])
            )  

        # center, extent = scene.get_scene_extent()
        # print(f"Scene center: {center}, extent: {extent}")

        scene.device = device
        return scene

    @classmethod
    def from_pt(cls, pt_path: str, require_grad: bool = False, device="cuda"):
        scene = cls()

        try:
            ckpt = torch.load(pt_path, map_location=device)
        except Exception as e:
            print(f"Failed to load checkpoint from {pt_path}: {e}")
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)

        # Build as ParameterDict from the start
        param_dict = {}
        # gsplat / TASE style checkpoints
        if "splats" in ckpt:
            for k, v in ckpt["splats"].items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                v = v.to(device)
                # Set requires_grad if requested and key is learnable
                v.requires_grad_(bool(require_grad))
                # Wrap as Parameter so optimizers will pick it up
                param_dict[k] = torch.nn.Parameter(v, requires_grad=v.requires_grad)
        elif "pipeline" in ckpt:
            pipeline = ckpt["pipeline"]
            # DreamCatalyst edit checkpoints use _model.<param> directly;
            # nerfstudio splatfacto (base models) use _model.gauss_params.<param>.
            if "_model.means" in pipeline:
                nerfstudio_keys = {
                    "means": "_model.means",
                    "quats": "_model.quats",
                    "scales": "_model.scales",
                    "opacities": "_model.opacities",
                    "sh0": "_model.features_dc",
                    "shN": "_model.features_rest",
                }
            else:
                nerfstudio_keys = {
                    "means": "_model.gauss_params.means",
                    "quats": "_model.gauss_params.quats",
                    "scales": "_model.gauss_params.scales",
                    "opacities": "_model.gauss_params.opacities",
                    "sh0": "_model.gauss_params.features_dc",
                    "shN": "_model.gauss_params.features_rest",
                }
            for k, ckpt_key in nerfstudio_keys.items():
                if ckpt_key in pipeline:
                    v = pipeline[ckpt_key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    v = v.detach().clone().to(device)
                    if k == "sh0" and v.ndim == 2:
                        # (N, 3) → (N, 1, 3)
                        v = v.unsqueeze(1)
                    if k == "opacities" and v.ndim == 2:
                        v = v.squeeze(-1)
                    param_dict[k] = torch.nn.Parameter(v, requires_grad=bool(require_grad))
                else:
                    print(
                        f"WARNING: Key {ckpt_key} not found in nerfstudio checkpoint."
                    )
        elif (
            isinstance(ckpt, (list, tuple))
            and len(ckpt) == 2
            and isinstance(ckpt[0], (list, tuple))
            and len(ckpt[0]) >= 7
            and isinstance(ckpt[0][0], int)
            and isinstance(ckpt[0][1], torch.Tensor)
        ):
            # Vanilla 3DGS .pth format: (model_params_tuple, iteration)
            # model_params[0] = active_sh_degree
            # model_params[1] = _xyz        (N, 3)
            # model_params[2] = _features_dc (N, 1, 3)
            # model_params[3] = _features_rest (N, sh_rest, 3)
            # model_params[4] = _scaling    (N, 3)  log-scale
            # model_params[5] = _rotation   (N, 4)  wxyz quaternions
            # model_params[6] = _opacity    (N, 1)  logit
            model_params = ckpt[0]
            active_sh_degree = int(model_params[0])
            for k, v in [
                ("means", model_params[1]),
                ("sh0", model_params[2]),
                ("shN", model_params[3]),
                ("scales", model_params[4]),
                ("quats", model_params[5]),
                ("opacities", model_params[6].squeeze(-1)),
            ]:
                v = v.detach().clone().to(device)
                param_dict[k] = torch.nn.Parameter(v, requires_grad=bool(require_grad))
            scene.sh_degree = active_sh_degree
        else:
            print(f"WARNING: Checkpoint format not recognized. Expected keys 'splats' or 'pipeline'. Found: ")
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    print(f"Dictionary with:")
                    for k, v in ckpt.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: Tensor of shape {v.shape}")
                        else:
                            print(f"  {k}: {v}")
                elif isinstance(ckpt, torch.Tensor):
                    print(f"Tensor of shape {ckpt.shape}")
                elif isinstance(ckpt, list | tuple):
                    print(f"List/Tuple of length {len(ckpt)}")
                    for i, item in enumerate(ckpt):
                        if isinstance(item, torch.Tensor):
                            print(f"  [{i}]: Tensor of shape {item.shape}")
                        else:
                            print(f"  [{i}]: {item}")
            raise RuntimeError("Unknown checkpoint format.")

        scene.splats = torch.nn.ParameterDict(param_dict)

        if "rotation_wxyz" in ckpt:
            scene.rotation_wxyz = ckpt["rotation_wxyz"].cpu().numpy()
        if "translation" in ckpt:
            scene.translation = ckpt["translation"].cpu().numpy()

        # Background models
        def _prep_bg(m):
            if m is None:
                return None
            m.to(device)
            for p in m.parameters():
                p.requires_grad = bool(require_grad)
            m.train() if require_grad else m.eval()
            return m

        scene.background_model = None
        if "background_MLP" in ckpt:
            try:
                scene.background_model = MLP_BackgroundModel()
                scene.background_model.load_state_dict(ckpt["background_MLP"])
            except Exception as e:
                print(f"Failed to load background model: {e}")
                scene.background_model = None
        scene.background_model = _prep_bg(scene.background_model)

        scene.feature_bg_model = None
        if "feature_background_MLP" in ckpt and "features" in scene.splats:
            try:
                scene.feature_bg_model = MLP_BackgroundModel(
                    output_dim=scene.splats["features"].shape[-1]
                )
                scene.feature_bg_model.load_state_dict(ckpt["feature_background_MLP"])
            except Exception as e:
                print(f"Failed to load feature background model: {e}")
                scene.feature_bg_model = None
        scene.feature_bg_model = _prep_bg(scene.feature_bg_model)

        print("Number of Gaussians:", int(scene.splats["means"].shape[0]))

        return scene

    @classmethod
    def from_ply(cls, ply_path: str, require_grad: bool = False, device="cuda"):
        scene = cls()
        scene.ckpt_dir = Path(ply_path)

        from plyfile import PlyData

        plydata = PlyData.read(ply_path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scene.splats = torch.nn.ParameterDict()
        scene.splats["means"] = torch.nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(
                require_grad
            ),
            requires_grad=require_grad,
        )
        scene.splats["sh0"] = torch.nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(require_grad),
            requires_grad=require_grad,
        )
        scene.splats["shN"] = torch.nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(require_grad),
            requires_grad=require_grad,
        )
        scene.splats["opacities"] = torch.nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(
                require_grad
            )[:, 0],
            requires_grad=require_grad,
        )
        scene.splats["scales"] = torch.nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(
                require_grad
            ),
            requires_grad=require_grad,
        )
        scene.splats["quats"] = torch.nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(
                require_grad
            ),
            requires_grad=require_grad,
        )
        scene.sh_degree = max_sh_degree

        return scene

    def save_ckpt(self, path: str, info: dict | None = None):
        """
        Save the current scene to a checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        split_scene = False
        if self.frozen_splats is not None and len(self.frozen_splats) > 0:
            self.merge_frozen_splats()
            split_scene = True

        data = {"splats": self.splats.state_dict()}
        if info is not None:
            data["info"] = info
        if self.background_model is not None:
            data["background_MLP"] = self.background_model.state_dict()
        if self.feature_bg_model is not None:
            data["feature_background_MLP"] = self.feature_bg_model.state_dict()

        if self.rotation_wxyz is not None:
            data["rotation_wxyz"] = torch.tensor(self.rotation_wxyz)
        if self.translation is not None:
            data["translation"] = torch.tensor(self.translation)

        torch.save(data, path)

        if split_scene:
            self.split_frozen_splats()

    @property
    def transform(self):
        return self.rotation_wxyz, self.translation

    @property
    def means(self):
        return self.splats.get("means")

    @property
    def quats(self):
        return self.splats.get("quats")

    @property
    def scales(self):
        return self.splats.get("scales")

    @property
    def opacities(self):
        return self.splats.get("opacities")

    @property
    def sh0(self):
        return self.splats.get("sh0")

    @property
    def shN(self):
        return self.splats.get("shN")

    @property
    def features(self):
        return self.splats.get("features")

    @property
    def classes(self):
        return self.splats.get("classes")

    @property
    def selection(self):
        return self.splats.get("selection")

    @property
    def num_splats(self):
        return (
            self.splats.get("means").shape[0]
            if self.splats and "means" in self.splats
            else 0
        )

    @property
    def colors(self):
        if "sh0" in self.splats and "shN" in self.splats:
            return torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        return None

    def load_config(self):
        if self.ckpt_dir is None:
            print("No checkpoint path provided!")
            return

        conf_path = self.ckpt_dir.parent.parent / "cfg.yml"

        if not conf_path.exists():
            print(f"Config path {conf_path} does not exist.")
            return

        self.cfg = Config.from_yaml(conf_path)

    def sample_road_camera_poses(
        self,
        num_poses: int,
        min_z: float = 1.0,
        max_z: float = 2.0,
        min_tilt: float = -0.3,
        max_tilt: float = 0.3,
        device: str = "cuda",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Samples camera poses from a semantic point cloud. Poses are derived from "road" points in the point cloud.
        DEPRICATED: Needs to be updated to use the new _rng
        Args:
            num_poses: The number of camera poses to sample.
            points: The point cloud data.
            cls: The class labels for the points.
            device: The device to use for the camera poses.
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing the quaternion and translation for each sampled camera pose.
        """
        if self.semantic_pc is None:
            raise ValueError(
                "No semantic point cloud loaded. Please load a semantic point cloud before sampling camera poses."
            )

        points = self.semantic_pc[:, :3]  # (N, 3)
        cls = self.semantic_pc[:, 3]  # (N,)

        device = torch.device(device)

        # get the valid class indices for the road classes
        road_idxs = []
        for name in [
            "road",
            "flat.driveable_surface",
        ]:  # NOTE: only road for now, add more if needed
            for key in self.name2idx:
                if name in key.lower():
                    road_idxs.append(self.name2idx[key])
                    break
        road_idxs = np.array(
            road_idxs, dtype=np.int32
        )  # convert to numpy array for torch.isin

        if len(road_idxs) == 0:
            raise ValueError(
                "Could not retrieve the relevant class indices. Check class mappings."
            )

        # convert numpy arrays to torch tensors
        points = torch.tensor(points, dtype=torch.float32, device=device)  # (N, 3)
        cls = torch.tensor(cls, dtype=torch.int32, device=device)
        road_idxs = torch.tensor(road_idxs, dtype=torch.int32, device=device)  # (M,)

        mask = torch.isin(cls, road_idxs)  # (N,) boolean mask
        filtered_points = points[mask]  # only those points
        if filtered_points.shape[0] == 0:
            raise ValueError(
                "No points found in the point cloud that match the road class. Check the point cloud data."
            )

        # get random subset of points
        num_points = filtered_points.shape[0]
        if num_points < num_poses:
            raise ValueError(
                f"Not enough points in the point cloud to sample {num_poses} poses. Found only {num_points} points."
            )
        sampled_idxs = np.random.choice(num_points, num_poses, replace=False)
        sampled_points = filtered_points[sampled_idxs]  # (num_poses, 3)

        # random z offset per pose
        z_offsets = (
            torch.rand(num_poses, device=device) * (max_z - min_z) + min_z
        )  # (num_poses,)
        sampled_points[:, 2] += z_offsets  # add z offset to the sampled points

        # random rho and theta per pose
        tilt = np.random.uniform(min_tilt, max_tilt, size=(num_poses))
        pan = np.random.uniform(0, 2 * np.pi, size=(num_poses))

        # convert to rotation and translation
        rot_trans = []
        for pan_i, tilt_i, trans in zip(pan, tilt, sampled_points):
            q = Scene.camera_quaternion_from_pan_tilt(
                pan=float(pan_i), tilt=float(tilt_i)
            )
            rot_trans.append((q.elements, trans.cpu().numpy()))

        return rot_trans

    def sample_random_camera_poses(
        self, num_poses: int, min_tilt: float = -0.3, max_tilt: float = 0.3
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample random camera poses for the scene.
        """
        samples = []
        # normal distribution within the aabb of the splats
        for i in range(3):
            mean = self.splats["means"][:, i].mean().item()
            mins = self.splats["means"][:, i].min().item()
            maxs = self.splats["means"][:, i].max().item()
            extent = maxs - mins
            std = extent / 3.0
            samples_i = torch.normal(mean=mean, std=std, size=(num_poses,))
            samples.append(samples_i)
        sampled_points = torch.stack(samples, dim=-1)  # (num_samples, 3)

        # random rho and theta per pose
        tilt = np.random.uniform(min_tilt, max_tilt, size=(num_poses))
        pan = np.random.uniform(0, 2 * np.pi, size=(num_poses))

        # convert to rotation and translation
        rot_trans = []
        for pan_i, tilt_i, trans in zip(pan, tilt, sampled_points):
            q = Scene.camera_quaternion_from_pan_tilt(
                pan=float(pan_i), tilt=float(tilt_i)
            )
            rot_trans.append((q.elements, trans.cpu().numpy()))

        return rot_trans

    @staticmethod
    def pan_tilt_roll_delta_local(
        pan: float, tilt: float, roll: float = 0.0
    ) -> Quaternion:
        """
        Intrinsic (camera-local) rotations:
        - pan  : about local up    (y)
        - tilt : about local right (x)
        - roll : about local fwd   (z)

        Returns a quaternion q_delta such that: q_new = q_old * q_delta
        """
        q_pan = Quaternion(axis=[0, 1, 0], angle=pan)  # yaw about local up
        q_tilt = Quaternion(axis=[1, 0, 0], angle=tilt)  # pitch about local right
        q_roll = Quaternion(axis=[0, 0, 1], angle=roll)  # roll about local forward
        return (q_pan * q_tilt * q_roll).normalised

    def set_seed(self, seed: int):
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

    def from_original_poses(
        self,
        num_samples: int,
        original_rot_trans,
        perturb: bool = True,
        rotation_std: float = 0.3,
        translation_std: float = 0.5,
        verbose: bool = True,
        seed: int | None = None,
    ):
        if not original_rot_trans:
            raise ValueError("No camera poses available to perturb.")

        N = len(original_rot_trans)

        # Sample indices with torch to stay on the same RNG stream
        idxs = torch.randint(low=0, high=N, size=(num_samples,), generator=self._rng)
        if verbose:
            print(f"sampled indices: {idxs.cpu().numpy()}")

        out = []
        for i in idxs.tolist():
            quat_wxyz, trans = original_rot_trans[i]
            q = Quaternion(quat_wxyz)
            t = np.asarray(trans, dtype=np.float32)

            if perturb:
                # Use the same generator for all randomness in this call
                pan = (torch.randn((), generator=self._rng) * rotation_std).item()
                tilt = (torch.randn((), generator=self._rng) * rotation_std).item()

                q_pert = Scene.pan_tilt_roll_delta_local(pan=pan, tilt=tilt, roll=0.0)
                q = q * q_pert

                t = (
                    (
                        torch.from_numpy(t)
                        + torch.randn(3, generator=self._rng) * translation_std
                    )
                    .cpu()
                    .numpy()
                )

            out.append((q.elements, t))

        return out

    def sample_object_centric_camera_poses(
        self,
        num_poses: int | Tuple[int, int],
        bounding_ellipsoid: Tuple[torch.Tensor, torch.Tensor],
        min_distance: float = 1.0,
        max_distance: float = 2.0,
        min_pan: float = -math.pi,
        max_pan: float = math.pi,
        pan_std: float = 0.5,
        min_tilt: float = -math.pi * 0.2,
        max_tilt: float = math.pi * 0.05,
        mode: str = "object_centric",  # "grid", "random", "continuous" or "orbit"
    ):
        """
        Sample camera poses on (or just outside) a bounding ellipsoid, all looking at its center.

        bounding_ellipsoid: (center, axes)
            center: (3,) tensor
            axes  : (3,3) tensor, each row a scaled principal axis vector (half-length * direction)

        Pan (yaw)  : rotation about world_up (z-up assumed)
        Tilt (pitch): rotation about camera-right after yaw. 0 = level, +tilt looks upward.

        mode:
            random: uniform random pan & tilt in ranges
            grid  : deterministic grid sweep (approx even coverage in rectangular param space)

        Returns:
            List of (quat_wxyz, position_xyz)
        """
        if bounding_ellipsoid is None:
            raise ValueError("Bounding ellipsoid required.")

        center_t, axes_t = bounding_ellipsoid
        center = center_t.detach().cpu().float()
        axes = axes_t.detach().cpu().float()  # (3,3)

        # Ellipsoid quadratic form (x-c)^T Q^{-1} (x-c)=1
        dirs_unit = F.normalize(axes, dim=1)
        lengths = axes.norm(dim=1)
        R = dirs_unit.T
        inv_l2 = 1.0 / (lengths.clamp_min(1e-8) ** 2)
        Q_inv = (R * inv_l2) @ R.T

        # Sample pan / tilt
        if mode == "grid":
            if isinstance(num_poses, Tuple):
                n_tilt, n_pan = num_poses
                num_poses = n_tilt * n_pan
            else:
                n_pan = int(math.ceil(math.sqrt(num_poses)))
                n_tilt = int(math.ceil(num_poses / n_pan))
            tilts = torch.linspace(min_tilt, max_tilt, n_tilt)
            pans = torch.linspace(min_pan, max_pan, n_pan + 1)[
                :-1
            ]  # avoid duplicating 2pi=0
            pans, tilts = torch.meshgrid(pans, tilts, indexing="ij")
            pans = pans.flatten()
            tilts = tilts.flatten()
            if len(pans) > num_poses:
                pans = pans[:num_poses]
                tilts = tilts[:num_poses]
        elif mode == "continuous":
            pans = torch.randn(num_poses, generator=self._rng) * pan_std + self.cur_pan
            self.cur_pan += pan_std
            tilts = (
                torch.rand(num_poses, generator=self._rng) * (max_tilt - min_tilt)
                + min_tilt
            )
        elif mode == "orbit":
            t = torch.linspace(0.0, 1.0, num_poses)
            pan_mid, pan_amp = 0.5 * (min_pan + max_pan), 0.5 * (max_pan - min_pan)
            tilt_mid, tilt_amp = 0.5 * (min_tilt + max_tilt), 0.5 * (
                max_tilt - min_tilt
            )

            pans = pan_mid + pan_amp * torch.cos(2 * math.pi * t)
            tilts = tilt_mid + tilt_amp * torch.sin(2 * math.pi * t)
        else:  # random
            pans = (
                torch.rand(num_poses, generator=self._rng) * (max_pan - min_pan)
                + min_pan
            )
            tilts = (
                torch.rand(num_poses, generator=self._rng) * (max_tilt - min_tilt)
                + min_tilt
            )

        print(f"pans: {pans}\ntilts: {tilts}")

        # Forward direction (camera looking direction) for each (pan, tilt):
        # Choose convention: forward(pan=0,tilt=0) = +Y
        # forward = [sin(pan) cos(tilt), cos(pan) cos(tilt), sin(tilt)]
        sin_pan, cos_pan = torch.sin(pans), torch.cos(pans)
        sin_tilt, cos_tilt = torch.sin(tilts), torch.cos(tilts)
        forward = torch.stack(
            [sin_pan * cos_tilt, cos_pan * cos_tilt, sin_tilt], dim=-1
        )  # (N,3)
        forward = F.normalize(forward, dim=-1)

        # Camera position direction from center = -forward
        out_dir = -forward

        # Distance to ellipsoid surface along out_dir
        denom = torch.einsum("ni,ij,nj->n", out_dir, Q_inv, out_dir).clamp_min(1e-12)
        t_surface = 1.0 / torch.sqrt(denom)

        if max_distance < min_distance:
            max_distance = min_distance
        extra = torch.rand(num_poses) * (max_distance - min_distance) + min_distance
        dist = t_surface + extra

        cam_positions = center[None, :] + dist[:, None] * out_dir  # (N,3)

        # Quaternions: directly from pan/tilt
        rot_trans = []
        for pan, tilt, pos, fwd in zip(pans, tilts, cam_positions, forward):
            q = Scene.camera_quaternion_from_pan_tilt(
                float(-pan), float(tilt)
            )  # your existing helper
            # (Optional sanity align: ensure q rotates -Z close to forward; usually fine)
            rot_trans.append((q.elements, pos.cpu().numpy().astype(np.float32)))

        return rot_trans

    @staticmethod
    def camera_quaternion_from_pan_tilt(pan: float, tilt: float) -> Quaternion:
        # Base: look down +Z (Z is up in world coords, forward in camera coords)
        base = Quaternion(axis=[1, 0, 0], angle=-np.pi / 2)

        # 1) Apply base, then yaw (about world Z)
        q = Quaternion(axis=[0, 0, 1], angle=pan) * base  # base first, then yaw

        # 2) Tilt about the *camera right* after yaw+base
        right_axis_world = q.rotate([1, 0, 0])  # camera's right in world coords
        q = Quaternion(axis=right_axis_world, angle=tilt) * q  # then tilt

        return q

    def sample_new_cam_poses(
        self,
        num_samples: int | Tuple[int, int],
        camera_sample_strategy: Literal[
            "random", "road", "perturb", "original", "object_centric", "orbit", "grid"
        ] = "perturb",
        rotation_std: float = 0.3,
        translation_std: float = 0.5,
        min_cam_dist: float = 1.0,
        max_cam_dist: float = 2.0,
        bounding_ellipsoid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_rot_trans: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = True,
    ) -> List[CameraState]:
        """
        Sample new camera poses for the scene.
        """
        rot_trans = []
        if camera_sample_strategy == "random":
            rot_trans = self.sample_random_camera_poses(num_samples)
        elif camera_sample_strategy == "road":
            rot_trans = self.sample_road_camera_poses(num_samples)
        elif camera_sample_strategy == "perturb":
            rot_trans = self.from_original_poses(
                num_samples,
                original_rot_trans,
                rotation_std=rotation_std,
                translation_std=translation_std,
                verbose=verbose,
            )
        elif camera_sample_strategy == "original":
            rot_trans = self.from_original_poses(
                num_samples, original_rot_trans, perturb=False, verbose=verbose
            )
        elif (
            camera_sample_strategy == "object_centric"
            or camera_sample_strategy == "orbit"
            or camera_sample_strategy == "grid"
        ):
            if bounding_ellipsoid is None:
                raise ValueError(
                    "Bounding ellipsoid must be provided for object-centric sampling."
                )
            rot_trans = self.sample_object_centric_camera_poses(
                num_samples,
                bounding_ellipsoid,
                min_distance=min_cam_dist,
                max_distance=max_cam_dist,
                mode=camera_sample_strategy,
            )
        else:
            raise ValueError(
                f"Unknown camera sample strategy: {camera_sample_strategy}"
            )

        cams = []
        for rot, trans in rot_trans:
            # build the c2w matrix from translation and rotation
            c2w = rotation_translation_to_pose(rot, trans)
            cams.append(c2w)

        return cams

    @torch.no_grad()
    def compute_bounding_ellipsoid(
        self,
        selected_only: bool = True,
        percentile: float = 1.0,
        enlarge_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a weighted bounding ellipsoid to currently SELECTED splats.

        center:
            Weighted center of mass (weights ~ volume * opacity).
        axes:
            Three (scaled) axis vectors a1,a2,a3 (rows of shape (3,)) obtained from the
            weighted covariance (principal components). Each axis length is the
            given percentile of |projection| of points onto that eigenvector,
            then multiplied by enlarge_factor.

        percentile:
            Percentile (0–1] used to define half-length along each principal axis.
            (e.g. 0.95 ≡ covers 95% (by distance) of points along that axis)
        enlarge_factor:
            Post scaling inflation factor for all three half-lengths.
        """

        means = self.means
        scales = self.scales
        opacities = self.opacities

        if selected_only:
            selection = self.selection
            if selection is None:
                self._log("No selection tensor present.")
                return
            selected = (selection > 0).squeeze(-1)
            if selected.sum() == 0:
                self._log("No splats selected. Cannot set bounding ellipsoid.")
                return

            means = means[selected]
            scales = scales[selected]
            opacities = opacities[selected]

        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)

        # Weights ~ (approx volume) * opacity
        weights = (scales.prod(dim=1) * opacities).clamp_min(1e-8)
        weights = weights / weights.sum()

        # Weighted center of mass
        center = (weights[:, None] * means).sum(dim=0)  # (3,)

        # Centered coordinates
        X = means - center[None, :]  # (M,3)

        # Weighted covariance (no 1/(sum w) needed beyond normalization above)
        # cov = Σ w_i x_i x_i^T
        cov = (X * weights[:, None]).T @ X  # (3,3)

        # Eigen decomposition (principal axes). eigh -> ascending order.
        evals, evecs = torch.linalg.eigh(cov)  # evals asc
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]  # columns are eigenvectors

        # For each principal direction, compute percentile of absolute projections
        axes = []
        for k in range(3):
            v = evecs[:, k]
            # Normalize (guard against numerical drift)
            v = v / v.norm().clamp_min(1e-12)
            proj = X @ v  # (M,)
            # Use absolute distance from center along this axis
            if percentile >= 1.0:
                half_len = proj.abs().max().item()
            else:
                half_len = torch.quantile(proj.abs(), percentile).item()
            # Fallback if degenerate
            if not np.isfinite(half_len) or half_len < 1e-9:
                half_len = evals[k].clamp_min(0).sqrt().item() + 1e-6
            half_len *= enlarge_factor
            axes.append(v * half_len)

        axes = torch.stack(axes, dim=0)  # (3,3) rows are scaled axis vectors

        return center.to(device=means.device, dtype=means.dtype), axes.to(
            device=means.device, dtype=means.dtype
        )


def norm_channel_wise(
    features: Union[torch.Tensor, np.ndarray], eps: float = 1e-6
) -> Union[torch.Tensor, np.ndarray]:
    """
    Channel-wise normalize the features to [0,1], for either
    a PyTorch tensor or a NumPy array.

    Args:
        features: (B, H, W, C) or (H, W, C) as torch.Tensor or np.ndarray
        eps: small constant to avoid divide-by-zero

    Returns:
        Normalized features of the same type and shape as input.
    """
    # dispatch based on type
    if isinstance(features, torch.Tensor):
        fn_amin = features.amin
        fn_amax = features.amax
    elif isinstance(features, np.ndarray):
        fn_amin = lambda **kwargs: np.amin(features, **kwargs)
        fn_amax = lambda **kwargs: np.amax(features, **kwargs)
    else:
        raise TypeError("features must be a torch.Tensor or np.ndarray")

    dim = features.ndim
    if dim == 4:
        # (B, H, W, C)
        axes = (1, 2)
        keep = True
    elif dim == 3:
        # (H, W, C)
        axes = (0, 1)
        keep = True
    else:
        raise ValueError("Features must be 3D or 4D (H,W,C) or (B,H,W,C)")

    min_vals = fn_amin(axis=axes, keepdims=keep)
    max_vals = fn_amax(axis=axes, keepdims=keep)

    # normalize
    normalized = (features - min_vals) / (max_vals - min_vals + eps)
    return normalized


def norm_globally(
    features: Union[torch.Tensor, np.ndarray], eps: float = 1e-6
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the features globally to [0,1], for either
    a PyTorch tensor or a NumPy array.

    Args:
        features: (B, H, W, C) or (H, W, C) as torch.Tensor or np.ndarray
        eps: small constant to avoid divide-by-zero

    Returns:
        Normalized features of the same type and shape as input.
    """
    # dispatch based on type
    if isinstance(features, torch.Tensor):
        fn_amin = features.amin
        fn_amax = features.amax
    elif isinstance(features, np.ndarray):
        fn_amin = lambda **kwargs: np.amin(features, **kwargs)
        fn_amax = lambda **kwargs: np.amax(features, **kwargs)
    else:
        raise TypeError("features must be a torch.Tensor or np.ndarray")

    min_val = fn_amin()
    max_val = fn_amax()

    # normalize
    normalized = (features - min_val) / (max_val - min_val + eps)
    return normalized


def norm_across_channels_old(
    features: Union[torch.Tensor, np.ndarray], eps: float = 1e-6
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the features across channels to [0,1], for either
    a PyTorch tensor or a NumPy array.

    Args:
        features: (B, H, W, C) or (H, W, C) as torch.Tensor or np.ndarray
        eps: small constant to avoid divide-by-zero

    Returns:
        Normalized features of the same type and shape as input.
    """
    # dispatch based on type
    if isinstance(features, torch.Tensor):
        fn_amin = features.amin
        fn_amax = features.amax
    elif isinstance(features, np.ndarray):
        fn_amin = lambda **kwargs: np.amin(features, **kwargs)
        fn_amax = lambda **kwargs: np.amax(features, **kwargs)
    else:
        raise TypeError("features must be a torch.Tensor or np.ndarray")

    min_vals = fn_amin(axis=-1, keepdims=True)
    max_vals = fn_amax(axis=-1, keepdims=True)

    # normalize
    normalized = (features - min_vals) / (max_vals - min_vals + eps)
    return normalized


def norm_across_channels(
    features: Union[torch.Tensor, np.ndarray], eps: float = 1e-6
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the features across channels to [0,1], for either
    a PyTorch tensor or a NumPy array.

    Args:
        features: (B, H, W, C) or (H, W, C) as torch.Tensor or np.ndarray
        eps: small constant to avoid divide-by-zero

    Returns:
        Normalized features of the same type and shape as input.
    """
    # dispatch based on type
    if isinstance(features, torch.Tensor):
        min_vals = features.min(dim=-1, keepdim=True).values
        max_vals = features.max(dim=-1, keepdim=True).values
    elif isinstance(features, np.ndarray):
        min_vals = np.min(features, axis=-1, keepdims=True)
        max_vals = np.max(features, axis=-1, keepdims=True)
    else:
        raise TypeError("features must be a torch.Tensor or np.ndarray")

    # normalize
    normalized = (features - min_vals) / (max_vals - min_vals + eps)
    return normalized


from TASE.splatting.trainer.TASE_dataparser import TASE, TASE_DataParserConfig
import time


def _resolve_dataset_for_parser(
    scene: Scene, dataset_override: Optional[str] = None
) -> tuple[TASE, Path]:
    """
    Build a TASE parser pointing to the dataset that will receive new samples.
    Priority:
      1) --dataset passed in
      2) scene.cfg.data_dir (from loaded checkpoint)
    """
    if dataset_override is not None:
        data_dir = resolve_path(dataset_override)
    elif scene.cfg is not None:
        data_dir = resolve_path(scene.cfg.data_dir)
    else:
        print("[WARN] No dataset override or scene.cfg.data_dir found.")
        return TASE(TASE_DataParserConfig()), "unknown_scene"

    scene_name = (
        scene.cfg.scene
        if not scene.cfg.scene.endswith(".json")
        else scene.cfg.scene[:-5]
    )
    scene_transform = scene.transform
    parser_cfg = TASE_DataParserConfig(
        data_dir=str(data_dir),
        metadata=scene_name,
        new_metadata=f"{scene_name}_EDIT_{int(time.time())}",
        scene_transform=scene_transform,
    )
    parser = TASE(parser_cfg)
    # Access to poses / camera meta if needed by SampleGenerator
    try:
        parser.get_poses()
    except Exception as e:
        print(f"[WARN] parser.get_poses() failed or not needed: {e}")

    return parser, scene_name


def _load_ckpt_and_dataset(
    ckpt: str,
    old_data_dir: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
    renormalize_coordinates: bool = False,
    flip_z: bool = False,
    dataparser_transform: Optional[Path | str] = None,
    dataset_split: Optional[str] = None,
):
    # ---- Load the data parser transform if provided
    if dataparser_transform is not None:
        dataparser_transform = resolve_path(dataparser_transform)
        if not dataparser_transform.exists():
            print(
                f"[WARN] dataparser transform path does not exist: {dataparser_transform}"
            )
        else:
            print(f"[Edit] Loading dataparser transform from: {dataparser_transform}")
            with open(dataparser_transform, "r") as f:
                transform_json = json.load(f)
            T_world_to_dataparser = transform_json["transform"]
            dataparser_scale = transform_json.get("scale", 1.0)
            dataparser_T = np.linalg.inv(T_world_to_dataparser)
            # dataparser_T = np.array(T_world_to_dataparser).astype(np.float32)
    else:
        dataparser_scale = 1.0
        dataparser_T = np.eye(4)

    # ---- Load scene from checkpoint (learnable)
    print(f"[Edit] Loading scene from: {ckpt}")
    scene = Scene.load_from_ckpt(
        ckpt,
        device=device,
        require_grad=True,
        renormalize_coordinates=renormalize_coordinates,
        flip_z=flip_z,
        dataparser_trafo=(dataparser_T, dataparser_scale),
    )
    if scene is None:
        print(f"Could not load scene from {ckpt}")

    # ---- Dataset / parser for saving edited samples
    try:
        TASE_parser, scene_name = _resolve_dataset_for_parser(scene, old_data_dir)
        print(f"[Edit] Scene name: {scene_name}")
        scene.name = scene_name  # ensure scene carries the name
    except Exception as ex:
        print(f"[WARN] Could not set up dataset parser: {ex}")
        TASE_parser, scene_name = TASE(TASE_DataParserConfig()), "unknown_scene"

    # ---- Load samples into the viewer if available
    try:
        samples = load_TASE_samples(
            TASE_parser.config.metadata,
            TASE_parser.datadir,
            device=device,
            scene_transform=scene.transform,
        )
    except Exception as ex:
        samples = None
        print(f"[WARN] Could not load samples from the dataset: {ex}")

    return scene, TASE_parser, scene_name, samples
