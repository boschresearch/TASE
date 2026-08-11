# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from nerfview V 0.0.3
#   (https://github.com/nerfstudio-project/nerfview/blob/main/nerfview/viewer.py)
# Copyright [not published upstream - see 3rd-party-licenses.txt], licensed
# under the Apache-2.0 license, cf. 3rd-party-licenses.txt file in the root
# directory of this source tree.

import dataclasses
import time
from threading import Lock, Timer
from typing import Callable, Literal, Optional, Tuple, Union, List
from pathlib import Path
import os
from datetime import datetime

import torch
import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8
import colorsys

from nerfview._renderer import Renderer, RenderTask

from TASE.splatting.trainer.trainer_cfg import Config
from TASE.splatting.util.utils import CameraOptModule

from TASE.splatting.util.data_util import *
from TASE.splatting.common.rendering import *
from TASE.splatting.common.segmentation import SegmentationModule3D
from TASE.splatting.common.rendering import Scene, _load_ckpt_and_dataset


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal["rendering", "preparing", "training", "paused", "completed"] = (
        "training"
    )


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        scene: Scene,
        cfg: Optional[Config] = None,
        mode: Literal["rendering", "training", "samples"] = "rendering",
        initial_pose: Optional[
            np.ndarray
        ] = None,  # A transformation matrix that defines the initial camera pose.
        colormapper: Optional[
            ColorMapper
        ] = None,  # pass the colormapper that is used for depth rendering so that the range of the colormap can be adjusted
        samples: Optional[
            Union[Path, Dict]
        ] = None,  # pass any additional elements like cameras that are supposed to be added to the visualization
        color_correct: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # pass the color correction matrix and offset
        pose_opt: Optional[
            CameraOptModule
        ] = None,  # pass the camera pose optimization module
        pose_perturb: Optional[
            CameraOptModule
        ] = None,  # pass the camera pose perturbation module
        gt_cam_poses: Optional[
            List[np.ndarray]
        ] = None,  # ground truth camera poses for pose optimization
        render_path: Optional[Path] = Path(
            "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/TASE/renders"
        ),  # path to save the rendered images
        segmentation_module: Optional[bool] = False,
        radius_clip: Optional[
            float
        ] = 1.0,  # clip gaussians smaller than this radius for better rendering speed
    ):
        # Public states.
        self.initialized = False
        self.server = server
        self.render_path = render_path
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        self.cfg = cfg
        if self.mode == "rendering":
            self.state.status = "rendering"
        self.elements = None
        self.radius_clip = radius_clip

        # Private GUI states.
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0
        self._num_gs = 0
        self._bg_color = torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"
        )

        if self.elements and "colormap_info" in self.elements:
            colormap_info = self.elements["colormap_info"]
        else:
            colormap_info = None

        self.set_scene(scene, attach_segmentation=False)

        # Rendering functions
        self._render_modes = ["RGB", "DEPTH", "NORMAL", "FEATURE"]
        self._renderers: dict[int, Renderer] = {}
        self._render_mode_idx = 0

        # per image color correction
        if color_correct is not None:
            self.color_correct_m = color_correct[0]
            self.color_correct_b = color_correct[1]
        else:
            self.color_correct_m = None
            self.color_correct_b = None

        # camera pose optimization
        self.pose_opt = pose_opt
        self.pose_perturb = pose_perturb

        # cancel flag that can be used to stop long-running operations like training or generation
        self.cancel = False

        # set the initial pose
        self.set_reset_pose(initial_pose)

        # add any additional scene elements like cameras
        if samples is None and self.cfg is not None and self.cfg.result_dir:
            samples = os.path.join(
                self.cfg.result_dir, "training_samples", self.cfg.scene
            )
        if isinstance(samples, str) or isinstance(samples, Path):
            self.elements = load_samples(samples)
        elif isinstance(samples, dict):
            self.elements = samples
        self._add_scene_elements()

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis(colormap_info)
        if segmentation_module and scene is not None:
            self.segmentation_module = SegmentationModule3D(
                self.server,
                self,
                scene.splats,
            )
            self.segmentation_module.attach()
        else:
            self.segmentation_module = segmentation_module

        self.scene_ckpt_dir = ""
        self.initialized = True

    def set_mode(self, mode: Literal["rendering", "training", "samples"]):
        self.mode = mode
        if self.mode == "rendering":
            self.state.status = "rendering"
        elif self.mode == "training":
            self.state.status = "training"
        elif self.mode == "samples":
            self.state.status = "rendering"

        if self._stats_folder is not None:
            self._stats_folder.visible = self.mode == "training"
        if self._training_folder is not None:
            self._training_folder.visible = self.mode == "training"

        self._last_update_step = 0

    def _load_cameras_json_btn_fn(self, event=None):
        path = self._cameras_json_text.value.strip()
        if not path:
            self._log("No cameras.json path provided.")
            return
        try:
            cameras = load_cameras_json(path)
        except Exception as e:
            self._log(f"Failed to load cameras.json: {e}")
            return
        self.add_samples(cameras)
        self._log(f"Loaded {len(cameras)} cameras from {path}.")

    def _load_scene_button_fn(self, event=None):
        path = self._load_scene_text.value.strip()
        load_dataset = self._load_dataset_cbx.value
        self.load_scene(path, load_dataset)

    def _export_ply_button_fn(self, event=None):
        if self.server.scene is None:
            self._log("No scene to export!")
            return
        path = self._load_scene_text.value.strip()
        if path == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"exported_scene_{timestamp}.ply"
        if self.scene is None:
            self._log("No scene to export!")
            return
        self.scene.export_to_ply(path)
        self._log(f"Exported scene to {path}.")

    def load_scene(
        self,
        ckpt_dir: str,
        load_dataset: bool = False,
        device="cuda",
        renormalize_coordinates: bool = False,
        dataparser_transform: Optional[Union[Path, str]] = None,
    ) -> None:
        ckpt_dir = str(ckpt_dir).strip()
        self._log(f"Loading scene from original ({ckpt_dir}).")
        if not load_dataset:
            ############ Quick reload ############
            # preserve the segmentation anchors
            if isinstance(self.segmentation_module, SegmentationModule3D):
                anchors = self.segmentation_module.get_anchors()
            else:
                anchors = None
            # only reload the scene, parser and samples stay the same
            scene = Scene.load_from_ckpt(
                ckpt_dir,
                device=device,
                require_grad=True,
                renormalize_coordinates=renormalize_coordinates,
            )
            parser, samples = None, None
        else:
            anchors = None
            scene, parser, data_dir, samples = _load_ckpt_and_dataset(
                ckpt_dir,
                None,
                device=device,
                renormalize_coordinates=renormalize_coordinates,
                dataparser_transform=dataparser_transform,
            )
        if scene is None:
            self._log(f"Failed to load scene from {ckpt_dir}.")
            return
        self.scene_ckpt_dir = ckpt_dir
        self._load_scene_text.value = ckpt_dir
        self.set_scene(scene)
        if samples is not None:
            self.elements = samples
            self._add_scene_elements()
        if anchors is not None and isinstance(
            self.segmentation_module, SegmentationModule3D
        ):
            self.segmentation_module.set_anchors(anchors)
        # keep the colormap range slider in sync with the freshly loaded scene
        if self.colormapper is not None and hasattr(self, "_cmap_range_slider"):
            if self.colormapper.vmax is None:
                self.colormapper.vmax = 5.0
            self._cmap_range_slider.value = self.colormapper.vmax
        self.rerender(None)

    def set_scene(self, scene: Scene, attach_segmentation: bool = True):
        if scene is None:
            self._render_fn = None
            self._depth_render_fn = None
            self._normal_render_fn = None
            self._feature_render_fn = None
            self._sel_render_fn = None
            self._render_none = None
            self.colormapper = None
            self.render_fn = self.meta_render_fn
            return

        self.scene = scene

        self._render_fn = scene.render_fn
        self._depth_render_fn = scene.depth_render_fn
        self._normal_render_fn = scene.normal_render_fn
        self._feature_render_fn = scene.feature_render_fn
        self._sel_render_fn = scene.selection_render_fn
        self._render_none = scene.dummy_render_fn
        self.base_render_fn = self._render_fn
        self.render_fn = self.meta_render_fn

        self.colormapper = scene.depth_colormapper

        if attach_segmentation and self.segmentation_module:
            if isinstance(self.segmentation_module, SegmentationModule3D):
                self.segmentation_module.detach()
            self.segmentation_module = SegmentationModule3D(
                self.server,
                self,
                scene.splats,
            )
            self.segmentation_module.attach()

    @torch.no_grad()
    def meta_render_fn(self, camera_state: CameraState, img_wh: Tuple[int, int]):
        try:
            show_seg = (
                self.segmentation_module is not None
                and isinstance(self.segmentation_module, SegmentationModule3D)
                and self.segmentation_module.show == True
            )
            background_color = self._bg_color if not self._bg_model_cbx.value else None
            img = self.base_render_fn(
                camera_state,
                img_wh,
                radius_clip=self.radius_clip,
                background_color=background_color,
            )
            if show_seg:
                overlay = self._sel_render_fn(
                    camera_state, img_wh, radius_clip=self.radius_clip
                )
                if (
                    overlay is not None
                    and img is not None
                    and overlay.shape == img.shape
                ):
                    alpha = 0.5
                    img = (1 - alpha) * img + alpha * overlay
                    img = np.clip(img, 0, 1)
        except Exception as e:
            self._log(f"Rendering error: {e}")
            img = np.zeros((img_wh[1], img_wh[0], 3), dtype=np.float32)
        return img

    @staticmethod
    def index_to_rgb(idx: int, n: int = 10) -> np.ndarray:
        """
        Map an integer index to an RGB color in [0,1].
        Colors cycle smoothly and repeat after n steps.
        """
        # compute hue in [0,1)
        hue = (idx % n) / float(n)
        # fixed saturation and value for vivid colors
        sat, val = 0.9, 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return np.array([r, g, b], dtype=np.float32)

    def _update_samples(self, keys=None):
        # Get the samples and scene for rendering
        scene = self.server.scene
        cameras = self.elements["cameras"] if "cameras" in self.elements else None
        if not cameras:
            self._log("No cameras to render!")
            return

        # Helper: safe to numpy
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        # Default color
        default_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # If keys is not provided, update all the cameras
        if keys is None:
            keys = cameras.keys()

        show_loss_masks = (
            self._loss_mask_cbx.value if hasattr(self, "_loss_mask_cbx") else False
        )
        show_bg_masks = (
            self._bg_mask_cbx.value if hasattr(self, "_bg_mask_cbx") else False
        )

        for i, key in enumerate(keys):
            cam = cameras[key]
            if cam is None or not isinstance(cam, dict):
                self._log(f"Camera {i}: invalid camera data — skipping.")
                continue
            # Normalize possible torch tensors to numpy
            cam2world = _to_np(cam.get("cam2world"))
            K = _to_np(cam.get("K"))
            dim = _to_np(cam.get("dim"))  # [H, W]
            idx = int(cam.get("idx", i))
            pose_type = cam.get("pose_type", "unknown")
            gen = int(cam.get("gen", cam.get("generation", 0)))
            filename = cam.get("filename")
            if not filename or not isinstance(filename, str) or filename.strip() == "":
                # unique fallback name
                filename = f"cam_{i}_idx{idx}"

            # Required fields sanity
            if cam2world is None or K is None or dim is None:
                self._log(f"Camera {i}: missing intrinsics/extrinsics/dims — skipping.")
                continue

            # Compute FOV/aspect (keep the original behavior)
            H, W = int(dim[0]), int(dim[1])
            fy = K[1, 1]
            fov = 2.0 * np.arctan(W / (2.0 * fy))  # matches the original code
            aspect = W / max(H, 1)

            # Optional per-sample maps (all are H×W×C in new format)
            img = _to_np(cam.get("img"))
            depth = _to_np(cam.get("depth"))
            # depth_conf= _to_np(cam.get("depth_conf"))
            normal = _to_np(cam.get("normal"))
            # normal_conf=_to_np(cam.get("normal_conf"))
            bg_mask = _to_np(cam.get("bg_mask"))
            feature = _to_np(cam.get("feature"))

            # Pose optimization (kept identical semantics)
            if self.pose_perturb is not None:
                cam2world_pert = (
                    self.pose_perturb(
                        torch.tensor(
                            cam2world,
                            device=self.pose_perturb.identity.device,
                            dtype=torch.float32,
                        )[None],
                        torch.tensor(
                            idx,
                            device=self.pose_perturb.identity.device,
                            dtype=torch.int64,
                        )[None],
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                wxyz, position = pose_to_rotation_translation(cam2world_pert)
                orig_cam_name = f"cameras_pert/{gen}/{filename}"
                scene.remove_by_name(orig_cam_name)  # remove old camera if exists
                scene.add_camera_frustum(
                    name=orig_cam_name,
                    fov=fov,
                    aspect=aspect,
                    position=position,
                    wxyz=wxyz,
                    image=None,
                    scale=0.1,
                    color=np.array([0.5, 0.5, 0.0], dtype=np.float32),
                )

            if self.pose_opt is not None:
                wxyz, position = pose_to_rotation_translation(cam2world)
                orig_cam_name = f"cameras_orig/{gen}/{filename}"
                scene.remove_by_name(orig_cam_name)  # remove old camera if exists
                scene.add_camera_frustum(
                    name=orig_cam_name,
                    fov=fov,
                    aspect=aspect,
                    position=position,
                    wxyz=wxyz,
                    image=None,
                    scale=0.1,
                    color=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                )
                cam2world = (
                    self.pose_opt(
                        torch.tensor(
                            cam2world,
                            device=self.pose_opt.identity.device,
                            dtype=torch.float32,
                        )[None],
                        torch.tensor(
                            idx,
                            device=self.pose_opt.identity.device,
                            dtype=torch.int64,
                        )[None],
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )

            # Final pose to place frustum
            wxyz, position = pose_to_rotation_translation(cam2world)

            # Choose the image to display by render mode
            if len(self._render_modes) > 0:
                mode = self._render_modes[self._render_mode_idx]
            else:
                mode = "RGB"
            if mode == "DEPTH" and (
                (self.colormapper is not None and depth is not None)
                or bg_mask is not None
            ):
                # Expect depth shape (H,W,1)
                if depth is not None:
                    d = (
                        depth[..., 0]
                        if depth.ndim == 3 and depth.shape[-1] == 1
                        else depth
                    )
                    img_to_show = self.colormapper.apply_colormap(d)
                else:
                    img_to_show = np.zeros((H, W, 3), dtype=np.float32)
            elif mode == "NORMAL" and normal is not None:
                img_to_show = normal / 2.0 + 0.5  # map [-1,1] -> [0,1]
                if bg_mask is not None:
                    mask = bg_mask[..., 0] if bg_mask.ndim == 3 else bg_mask
                    img_to_show[mask <= 0] = 0.0
            elif mode == "FEATURE":
                if feature is not None:
                    # take first 3 channels if present
                    img_to_show = feature[..., :3] if feature.ndim == 3 else feature
                    img_to_show = norm_channel_wise(
                        img_to_show
                    )  # normalize the features across channels to [0, 1]
                    # img_to_show = img_to_show / 2.0 + 0.5
                    img_to_show = np.clip(img_to_show, 0.0, 1.0)
                else:
                    self._log("No features available for rendering!")
                    img_to_show = np.zeros((H, W, 3), dtype=np.float32)
            else:
                # Optional per-image color correction (index by sample idx)
                img_to_show = img
                if (
                    self.color_correct_m is not None
                    and self.color_correct_b is not None
                    and img_to_show is not None
                ):
                    # img: (H,W,3), apply affine correction (x - b) @ (I + M)^T
                    img_t = torch.from_numpy(img_to_show).to(
                        self.color_correct_b.device
                    )
                    img_b = self.color_correct_b[idx]
                    img_m = self.color_correct_m[idx] + torch.eye(
                        3, device=self.color_correct_m.device
                    )
                    img_t = (img_t - img_b) @ img_m.transpose(0, 1)
                    img_to_show = torch.clamp(img_t, 0, 1).detach().cpu().numpy()
            if img_to_show is None:
                img_to_show = np.zeros((H, W, 3), dtype=np.float32)
            if show_loss_masks and "loss_mask" in cam:
                loss_mask = _to_np(cam.get("loss_mask"))
                if loss_mask is not None:
                    mask = loss_mask
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    if mask.ndim == 2:  # (H,W) -> (H,W,1)
                        mask = mask[..., None]
                    # ensure float32 in [0,1]
                    mask = mask.astype(np.float32)
                    if mask.max() > 1.0:
                        mask = mask / mask.max()

                    # Option A: additive tint
                    img_to_show = np.clip(
                        img_to_show
                        + mask
                        * np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, None, :],
                        0.0,
                        1.0,
                    )
            if show_bg_masks and bg_mask is not None:
                mask = bg_mask
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                if mask.ndim == 2:  # (H,W) -> (H,W,1)
                    mask = mask[..., None]
                # ensure float32 in [0,1]
                mask = mask.astype(np.float32)
                if mask.max() > 1.0:
                    mask = mask / mask.max()

                # Option A: additive tint
                img_to_show = np.clip(
                    img_to_show
                    + mask * np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :],
                    0.0,
                    1.0,
                )

            # Color by pose_type
            """
            if pose_type == "nuScenes":
                color = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif pose_type == "CAMA":
                color = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            elif pose_type == "TASE":
                color = np.array([1.0, 0.5, 0.0], dtype=np.float32)
            else:
                color = default_color
            """
            # color by sample generation
            color = self.index_to_rgb(
                gen
            )  # use the generation index to color the cameras

            # Build name: use gen & filename (with fallbacks as requested)
            cam_name = f"cameras/{gen}/{filename}"
            cameras[key]["viewer_name"] = cam_name  # store for later removal

            scene.remove_by_name(cam_name)  # remove old camera if exists
            cam_obj = scene.add_camera_frustum(
                name=cam_name,
                fov=fov,
                aspect=aspect,
                position=position,
                wxyz=wxyz,
                image=img_to_show,
                scale=0.1,
                color=color,
            )
            if self._camera_clicked_callback is not None:
                cam_obj.on_click(self._camera_clicked_callback)

    def add_samples(self, cameras):
        """
        Add some additional samples to the viewer during runtime.
        """
        # Update the elememnts dict
        if self.elements is None:
            self.elements = {}
        if "cameras" not in self.elements:
            self.elements["cameras"] = {}
        self.elements["cameras"].update(cameras)

        # update only the newly added cameras
        self._update_samples(cameras.keys())

    def remove_samples(self, keys):
        """
        Remove some samples from the viewer during runtime.
        """
        if self.elements is None or "cameras" not in self.elements:
            return
        cameras = self.elements["cameras"]
        for key in keys:
            viewer_name = key
            if key in cameras:
                viewer_name = cameras[key].get("viewer_name", key)
                del cameras[key]
            try:
                self.server.scene.remove_by_name(viewer_name)
            except Exception:
                self._log(f"Could not remove camera {viewer_name} from viewer.")

    def _add_scene_elements(self, event=None):
        elements = self.elements
        if elements is None:
            return
        scene = self.server.scene
        # Only do this if not called as gui callback
        if event is None:
            # clear the scene
            scene.reset()
            # Handle point clouds
            if "pointclouds" in elements and elements["pointclouds"] is not None:
                point_clouds = elements["pointclouds"]
                for key in point_clouds:
                    point_cloud = point_clouds[key]

                    if isinstance(point_cloud, Dict):
                        points = point_cloud["points"]
                        if "normals" in point_cloud:
                            normals = point_cloud["normals"]
                        if "colors" in point_cloud:
                            colors = point_cloud["colors"]
                    elif isinstance(point_cloud, torch.Tensor):
                        if point_cloud.shape[1] == 9:
                            points = point_cloud[:, :3].cpu().numpy()
                            colors = point_cloud[:, 3:6].cpu().numpy()
                            normals = point_cloud[:, 6:9].cpu().numpy()
                        elif point_cloud.shape[1] == 6:
                            points = point_cloud[:, :3].cpu().numpy()
                            colors = point_cloud[:, 3:6].cpu().numpy()
                            normals = None
                        elif point_cloud.shape[1] == 3:
                            points = point_cloud[:, :3].cpu().numpy()
                            colors = None
                            normals = None
                        else:
                            self._log(f"Unknown point cloud shape: {point_cloud.shape}")
                            continue

                    if colors is None:
                        if "init" in key:
                            colors = np.ones_like(points) * np.array([1.0, 0, 0])
                        elif "random" in key:
                            colors = np.ones_like(points) * np.array([0.3, 0.3, 0.3])
                        else:
                            colors = np.ones_like(points) * np.random.rand(3)

                    scene.add_point_cloud(
                        name=f"pointclouds/pc_{key}",
                        points=points,
                        colors=colors,
                        point_shape="circle",
                        point_size=0.1,
                        visible=False,
                    )

                    if normals is not None:
                        """
                        Note: nerfstudio requires viser version 0.2.7, add_line_segments however is based on a newer viser version (tested with 0.2.23).
                        I will try to make my code compatible with newer version, but I might have to keep the old version.
                        """
                        normals = np.stack((points, points + normals * 0.2), axis=1)
                        scene.add_line_segments(
                            name=f"normals/pc_{key}",
                            points=normals,
                            colors=colors,
                            line_width=2,
                            visible=False,
                        )

        # Handle bounding boxes
        if "bounding_boxes" in elements and elements["bounding_boxes"] is not None:
            bounding_boxes = elements["bounding_boxes"]
            for key in bounding_boxes:
                bbx = bounding_boxes[key]
                if isinstance(bbx, torch.Tensor):
                    bbx = bbx.cpu().numpy()
                if bbx.shape == (10,):
                    center = bbx[:3]
                    wlh = bbx[3:6]
                    rotation = bbx[6:10]
                else:
                    self._log(f"Unknown bounding box shape: {bbx.shape}")
                    continue
                scene.add_box(
                    name=f"bounding_boxes/bbox_{key}",
                    position=center,
                    dimensions=wlh,
                    wxyz=rotation,
                    color=np.array([0.0, 0.0, 0.0]),
                )

        # Handle cameras
        if "cameras" in elements and elements["cameras"] is not None:
            self.add_samples(elements["cameras"])

    def _get_legend_markdown(
        self, color2names: Dict[Tuple[int, int, int], List[str]]
    ) -> str:
        """Generates a pure-Markdown legend table based on the provided colormap dictionary.
        Args:
            color2names (dict): A dictionary mapping RGB tuples to lists of class names.
        Returns:
            str: A Markdown-formatted table representing the legend for the Viser GUI.
        """
        # Header for the legend table
        lines = ["Color | Classes ", "--- | ---"]
        # Populate rows
        for color, names in color2names.items():
            names = [
                name.split(".")[-1] for name in names
            ]  # Use only the last part of the class name
            # Join multiple class names with commas
            label = ", ".join(names)
            # Convert RGB tuple to hex string
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            # Add a table row with hex code and class names
            lines.append(f"| `{hex_color}` | {label} |")
        # Combine into single Markdown string
        return "\n".join(lines)

    def _define_guis(self, colormap_info: Optional[dict] = None):
        with self.server.gui.add_folder(
            "Stats", visible=self.mode == "training"
        ) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step:_}({self._last_update_step:_}) \n
                Num GS: {self._num_gs:_}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Update freq.", min=1, max=100, step=1, initial_value=20
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)
            self._fov_slider = self.server.gui.add_slider(
                "FOV", min=0.5, max=1.5, step=0.1, initial_value=0.7
            )
            self._fov_slider.on_update(self._update_fov_fn)
            # Depth colormap range slider. Created unconditionally so that it
            # remains available even if the scene is loaded later on via
            # `load_scene` (e.g. when the viewer is started without a scene).
            initial_vmax = 5.0
            if self.colormapper is not None:
                if self.colormapper.vmax is None:
                    self.colormapper.vmax = 5.0
                initial_vmax = self.colormapper.vmax
            self._cmap_range_slider = self.server.gui.add_slider(
                "Cmap range",
                min=0.0,
                max=100.0,
                step=1.0,
                initial_value=initial_vmax,
            )
            self._cmap_range_slider.on_update(self._cmap_range_slider_fn)
            # Checkbox to toggle rendering of background models
            self._bg_model_cbx = self.server.gui.add_checkbox(
                "Render background", initial_value=True
            )
            # Overlay the lossmasks to the viewer if available
            self._loss_mask_cbx = self.server.gui.add_checkbox(
                "Show loss masks", initial_value=False
            )
            # Overlay the background masks to the viewer if available
            self._bg_mask_cbx = self.server.gui.add_checkbox(
                "Show bg masks", initial_value=False
            )
            # Render mode button
            if len(self._render_modes) > 1:
                self._render_mode_drop = self.server.gui.add_dropdown(
                    f"Mode",
                    self._render_modes,
                    initial_value=self._render_modes[self._render_mode_idx],
                )
                self._render_mode_drop.on_update(self._toggle_render_mode_fn)
            if self.color_correct_m is not None or self.colormapper is not None:
                self.camera_update = self.server.gui.add_button("Update cameras")
                self.camera_update.on_click(self._add_scene_elements)
            # Save image button
            self._save_image_btn = self.server.gui.add_button(f"Save image")
            self._save_image_btn.on_click(self._save_image_fn)
            # Load scene button
            self._load_scene_btn = self.server.gui.add_button(f"Load scene")
            self._load_scene_btn.on_click(self._load_scene_button_fn)
            # Export scene to PLY button
            self._export_ply_btn = self.server.gui.add_button(f"Export scene to PLY")
            self._export_ply_btn.on_click(self._export_ply_button_fn)
            # Text field for the scene path
            self._load_scene_text = self.server.gui.add_text(
                "Scene path",
                initial_value="",
            )
            # Checkbox wether to try loading the dataset along with the scene
            self._load_dataset_cbx = self.server.gui.add_checkbox(
                "Load dataset", initial_value=False
            )
            # cameras.json loader
            self._cameras_json_text = self.server.gui.add_text(
                "cameras.json path",
                initial_value="",
            )
            self._load_cameras_json_btn = self.server.gui.add_button("Load cameras.json")
            self._load_cameras_json_btn.on_click(self._load_cameras_json_btn_fn)

        with self.server.gui.add_folder(
            "Navigation", expand_by_default=False
        ) as self._navigation_folder:
            # Add a textbox and a button to explicitly set the pose
            self._pose_text_input = self.server.gui.add_text(
                "Pose", initial_value=str(self.reset_pose)
            )
            self._reset_view_button = self.server.gui.add_button("Set Pose")
            self._reset_view_button.on_click(self._set_view_button_fn)
            self._max_img_res_slider.on_update(self.rerender)
            # Add buttons to incrementally turn the camera
            self._nav_panel_1 = self.server.gui.add_button_group("", ["Up", "Down"])
            self._nav_panel_1.value = ""
            self._nav_panel_2 = self.server.gui.add_button_group("", ["Left", "Right"])
            self._nav_panel_2.value = ""
            self._nav_panel_1.on_click(self._nav_panel_fn)
            self._nav_panel_2.on_click(self._nav_panel_fn)
            self._nav_get_pose_btn = self.server.gui.add_button("Get Current Pose")
            self._nav_get_pose_btn.on_click(self._get_current_pose_fn)

        if colormap_info is not None and len(colormap_info) > 0:
            with self.server.gui.add_folder(
                "Colormap Legend", expand_by_default=False
            ) as legend_folder:
                # Add a legend for the colormap
                legend_html = self._get_legend_markdown(colormap_info)
                self.server.gui.add_markdown(legend_html, "Colormap Legend")

    def _update_fov_fn(self, event):
        event.client.camera.fov = self._fov_slider.value
        self.rerender(event)

    def _camera_clicked_callback(self, event):
        cam = event.target
        client = event.client
        client.camera.position = cam.position
        client.camera.wxyz = cam.wxyz
        # Print the camera name
        self._log(f"Camera clicked: {cam.name}")
        # update the pose textbox
        c2w = rotation_translation_to_pose(cam.wxyz, cam.position)
        self._pose_text_input.value = str(c2w.tolist())

    def _cmap_range_slider_fn(self, event):
        vmax = event.target.value
        if self.colormapper is not None:
            self.colormapper.vmax = vmax
        self.rerender(None)

    def _set_view_button_fn(self, event):
        client = event.client
        # Try to read the text field
        self.set_reset_pose(self._pose_text_input.value)
        # set the initial camera pose if applicable
        if self.reset_pose is not None:
            self._log(f"Setting the pose to: \n{self.reset_pose}")
            self.set_camera_state(client, self.reset_pose)

    def _toggle_render_mode_fn(self, event):
        mode_name = event.target.value
        if mode_name not in self._render_modes:
            raise self._log(
                f"Unknown render mode: {mode_name}. Available modes: {self._render_modes}"
            )
        self._render_mode_idx = self._render_modes.index(mode_name)
        self._render_mode_idx = self._render_mode_idx % len(self._render_modes)
        mode = self._render_modes[self._render_mode_idx]

        # set the button label
        event.target.label = f"Mode: {mode}"

        if mode == "None":
            self.base_render_fn = self._render_none
        elif mode == "RGB":
            self.base_render_fn = self._render_fn
        elif mode == "DEPTH":
            self.base_render_fn = self._depth_render_fn
        elif mode == "NORMAL":
            self.base_render_fn = self._normal_render_fn
        elif mode == "FEATURE":
            self.base_render_fn = self._feature_render_fn
        elif mode == "SELECTION":
            self.base_render_fn = self._sel_render_fn
        else:
            raise ValueError(f"Unknown render mode: {mode}")

        self._log(f"Setting render mode to {mode}.")
        # update the view
        self._add_scene_elements(event)
        self.rerender(None)

    def _save_image_fn(self, event):
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_img_name_str = "render"
        if self.render_path is None:
            self.render_path = Path(__file__).parent / "renders"
        renders = self.render_path / time_str
        if not renders.exists():
            os.makedirs(renders, exist_ok=True)

        # save all the modalities
        camera_state = self.get_camera_state(event.client)
        # aspect = 16.0 / 9.0  # default aspect ratio
        aspect = camera_state.aspect
        resolution = (
            int(self._max_img_res_slider.value * aspect),
            int(self._max_img_res_slider.value),
        )
        img_names = []
        if self._render_fn is not None:
            img = self._render_fn(camera_state, resolution)
            img_name = f"{base_img_name_str}_rgb.png"
            img_path = renders / img_name
            if self.color_correct_m is not None and self.color_correct_b is not None:
                img = torch.from_numpy(img).to(self.color_correct_b.device)
                img_b = self.color_correct_b[0]
                img_m = self.color_correct_m[0] + torch.eye(3).to(
                    self.color_correct_m.device
                )
                img = (img - img_b) @ img_m.transpose(0, 1)
                img = torch.clamp(img, 0, 1).detach().cpu().numpy()
            # rgb to bgr
            img = img[..., ::-1]  # Convert RGB to BGR
            cv2.imwrite(str(img_path), img * 255)
            img_names.append(img_name)
        if self._depth_render_fn is not None:
            depth_img, alpha_img = self._depth_render_fn(
                camera_state, resolution, raw=True
            )
            depth_name = f"{base_img_name_str}_depth.npy"
            depth_path = renders / depth_name
            np.save(depth_path, depth_img)
            img_names.append(depth_name)
            alpha_name = f"{base_img_name_str}_alpha.npy"
            alpha_path = renders / alpha_name
            np.save(alpha_path, alpha_img)
            img_names.append(alpha_name)
        if self._normal_render_fn is not None:
            normal_img = self._normal_render_fn(camera_state, resolution, raw=True)
            normal_name = f"{base_img_name_str}_normal.npy"
            normal_path = renders / normal_name
            np.save(normal_path, normal_img)
            img_names.append(normal_name)
        if self._feature_render_fn is not None:
            feature_img = self._feature_render_fn(camera_state, resolution, raw=True)
            feature_name = f"{base_img_name_str}_feature.npy"
            feature_path = renders / feature_name
            np.save(feature_path, feature_img)
            img_names.append(feature_name)
        with open(renders / "render_info.txt", "w") as f:
            f.write(f"Camera state: {camera_state}\n")
            f.write(f"Image resolution: {resolution}\n")
            f.write(f"Color correction matrix: {self.color_correct_m}\n")
            f.write(f"Color correction offset: {self.color_correct_b}\n")
            f.write(f"Image names: {', '.join(img_names)}\n")

        self._log(f"Saved images {img_names} to {renders}.")

    def rotate_vector(self, v, axis, angle):
        axis = axis / np.linalg.norm(axis)
        return (
            v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * (np.dot(axis, v)) * (1 - np.cos(angle))
        )

    def _get_current_pose_fn(self, event):
        client = event.client
        camera = client.camera
        q = camera.wxyz
        t = camera.position
        c2w = rotation_translation_to_pose(q, t)
        print(f"Rotation (wxyz): {q}")
        print(f"Translation: {t}")
        print(f"Camera-to-world pose:\n{c2w}")
        self._pose_text_input.value = str(c2w).replace("\n", " ")

    def _nav_panel_fn(self, event):
        # TODO: This doesn't work as expected
        if event.client is None:
            return
        camera = event.client.camera
        button = event.target.value
        angle = 10 / 180 * 3.14  # small rotation angle in radians
        # Compute the right vector from current look_at and up_direction.
        right = np.cross(camera.look_at, camera.up_direction)

        if button == "Left":
            camera.look_at = self.rotate_vector(
                camera.look_at, camera.up_direction, angle
            )
        elif button == "Right":
            camera.look_at = self.rotate_vector(
                camera.look_at, camera.up_direction, -angle
            )
        elif button == "Up":
            camera.look_at = self.rotate_vector(camera.look_at, right, angle)
        elif button == "Down":
            camera.look_at = self.rotate_vector(camera.look_at, right, -angle)
        else:
            self._log("Invalid button.")

        event.target.value = ""

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):

        if not self.initialized:
            self._log("Viewer not initialized yet, waiting.")
            while not self.initialized:
                time.sleep(0.1)

        client_id = client.client_id

        # Set the fov
        self.set_camera_state(client, pose=self.reset_pose, fov=self._fov_slider.value)

        # set the near render distance
        client.camera.near = 0.25

        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            # Maybe I can change the way the camera is manipulated somewhere here
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

        self._log(f"connected client {client.client_id}")

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    # sets the camera state of the passed viewer
    def set_camera_state(
        self,
        client: viser.ClientHandle,
        pose: Optional[Float32[np.ndarray, "4 4"]],
        fov: Optional[float] = None,
    ):
        camera = client.camera
        if pose is not None:
            q, t = pose_to_rotation_translation(pose)
            camera.wxyz = q
            camera.position = t
        if fov is not None:
            camera.fov = fov

    # sets the pose to which the camera gets initialized when a client connects
    def set_reset_pose(self, pose: np.ndarray):

        if pose is None:
            self.reset_pose = None
            return

        if type(pose) == str:
            clean_str = pose.replace("[", "").replace("]", "").replace(",", " ")
            pose = np.fromstring(clean_str, sep=" ")
            if pose.size == 12:
                pose = pose.reshape(3, 4)
            elif pose.size == 16:
                pose = pose.reshape(4, 4)
            else:
                self._log("Invalid reset pose.")

        if pose.shape == (3, 4):
            pose = np.concatenate([pose, [0, 0, 0, 1]], axis=0)

        if pose.shape != (4, 4):
            self._log("Invalid pose shape!")
            self.reset_pose = None
            return

        self.reset_pose = pose

    def _log(self, msg: str):
        print(f"[Viewer] {msg}")

    def update(self, step: int, num_train_rays_per_step: int, num_gaussians: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        self._num_gs = num_gaussians
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                # train_util * view_time / (train_time - train_util * train_time)
            )
            update_every = int(train_util)
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(
                        RenderTask("update", camera_state)
                    )
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
