# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple, Literal, List, Optional, Sequence, Union
from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers import (
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
)
from diffusers.models.controlnet import zero_module
from diffusers.models.modeling_utils import load_state_dict
from pathlib import Path
import torch.nn.functional as F

from TASE.splatting.common.rendering import Scene
from TASE.splatting.trainer.TASE_dataparser import TASE
from TASE.splatting.util.data_util import pose_to_rotation_translation, resolve_path

from nerfstudio.cameras.cameras import Cameras
from nerfview import CameraState
from datetime import datetime
import os


# ------------- Helper functions -------------
def cameras_to_states(cameras: Cameras) -> List[CameraState]:
    """
    Convert a Cameras object to a list of CameraState objects.
    Assumes fx/fy are per-camera focal lengths in pixels.
    """
    num_cams = cameras.camera_to_worlds.shape[0]
    cam_states = []

    for i in range(num_cams):
        # Get width and height as Python ints
        W = int(cameras.width[i].item())
        H = int(cameras.height[i].item())

        # Get fy for vertical FOV
        fy_val = float(cameras.fy[i].item())
        fov = 2.0 * np.arctan(H / (2.0 * fy_val))  # vertical FOV in radians

        # Compute aspect ratio
        aspect = W / H

        # Convert c2w to 4x4 matrix
        if cameras.camera_to_worlds.shape[1] == 3:
            c2w_3x4 = cameras.camera_to_worlds[i].cpu().numpy()
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :4] = c2w_3x4

        cam_states.append(
            CameraState(
                fov=fov,
                aspect=aspect,
                c2w=c2w_4x4.astype(np.float32),
            )
        )

    return cam_states


class SampleGenerator:
    """
    A class to generate samples for scene generation.
    """

    parser: TASE
    scene: Scene
    ctrl_encoder: EmbeddingModel

    background_color_options: dict[str, Tuple[float, float, float]] = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "gray": (0.5, 0.5, 0.5),
        "dark_gray": (0.25, 0.25, 0.25),
        "light_gray": (0.75, 0.75, 0.75),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
        "purple": (0.5, 0.0, 0.5),
        "pink": (1.0, 0.75, 0.8),
        "navy": (0.0, 0.0, 0.5),
        "beige": (0.96, 0.96, 0.86),
    }

    def __init__(
        self,
        scene: Scene = None,
        parser: TASE = None,
        load_models: bool = True,
        ctrl_net_path: str = None,
        ctrl_net_lora_path: str = None,
        mrl_model_path: str = None,
        pca_path: str = None,
        use_pca: bool = False,
        raw_dino: bool = False,
        transformer_path: str = None,
        transformer_lora_path: str = None,
        max_ctrl_channels: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        feature_hacks: bool = True,
    ):
        self.device = device
        self.dtype = dtype

        self.max_ctrl_channels = max_ctrl_channels
        self.use_pca = use_pca
        self.raw_dino = raw_dino

        if load_models:
            (
                self.pipe,
                self.controlnet,
                self.noise_scheduler,
                self.ctrl_encoder,
                self.depth_estimation_model,
            ) = self.load_models(
                ctrl_net_path=ctrl_net_path,
                ctrl_net_lora_path=ctrl_net_lora_path,
                mrl_model_path=mrl_model_path,
                pca_path=pca_path,
                transformer_path=transformer_path,
                transformer_lora_path=transformer_lora_path,
                device=device,
                dtype=dtype,
            )
            self.transformer = self.pipe.transformer

        self.update_intrinsics()

        self.feature_upsampling_mode = "rendering"  # "bilinear" or "rendering"

        self.max_pred_depth = (
            180.0 * 0.95
        )  # Maximum predicted depth in metric canonical camera space. This should be constant for all samples.

        self.prompt_encoding = None  # cache the prompt encoding if needed
        self.current_prompt = None
        self.feature_hacks = feature_hacks

        self._rng = torch.Generator()

        self.scene = scene
        if parser is None:
            print("WARNING: No parser provided, cannot save samples to dataset.")
        self.parser = parser
        self.tmp_names = []  # keep track of temporary sample names to remove them later

    def tensor_to_camera_state(self, poses: torch.Tensor) -> List[CameraState]:
        """
        Convert a tensor of camera poses to a list of CameraState objects.
        Args:
            poses (torch.Tensor): Tensor of shape (N, 4, 4) representing camera poses.
        Returns:
            List[CameraState]: List of CameraState objects.
        """
        cam_states = []

        # Calculate FOV and aspect ratio
        H, W = self.H, self.W
        aspect = W / H

        for pose in poses:
            # Convert to numpy array
            c2w = pose.cpu().numpy().astype(np.float32)
            fov = self.fov  # Use the fov from the SampleGenerator instance
            cam_states.append(CameraState(fov=fov, aspect=aspect, c2w=c2w))

        return cam_states

    def update_intrinsics(self, height: int = 512, width: int = 512, fov: float = 0.7):

        self.H = height
        self.W = width
        self.fov = fov

        focal = self.H / (
            2.0 * np.tan(self.fov / 2.0)
        )  # vertical focal only, derived from vertical fov. Horizontal focal is the same (square pixels assumed). Horizontal fov is derived from aspect ratio.

        self.intrinsics = {
            "fov": self.fov,
            "fx": focal,
            "fy": focal,
            "cx": self.W / 2,
            "cy": self.H / 2,
            "width": self.W,
            "height": self.H,
        }

        # Update K matrix
        self.K = np.array(
            [
                [self.intrinsics["fx"], 0, self.intrinsics["cx"]],
                [0, self.intrinsics["fy"], self.intrinsics["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def get_metric3d_depth(
        self, rgb_origin: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Copied code from metric3d repo (https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L145)
        Coordinates for the normal maps:
            - x: right
            - y: down
            - z: forward
        """
        # convert the rgb image to the correct format
        model = self.depth_estimation_model
        device = self.device
        # dtype = self.dtype
        #### ajust input size to fit pretrained model
        # keep ratio resize
        input_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(
            rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        # remember to scale intrinsic, hold depth
        self.canonical_to_real_scale = (
            self.intrinsics["fy"] * scale / 1000.0
        )  # The scaling factor caused by the difference between the canonical camera and the real camera.

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].to(device)

        ###################### canonical camera space ######################
        # inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({"input": rgb})
        pred_normal = output_dict["prediction_normal"][
            :, :3, :, :
        ]  # only available for Metric3Dv2 i.e., ViT models
        normal_confidence = output_dict["prediction_normal"][
            :, 3, :, :
        ]  # see https://arxiv.org/abs/2109.09881 for details

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1],
            pad_info[2] : pred_depth.shape[1] - pad_info[3],
        ]
        confidence = torch.clamp(
            confidence, 0, 1
        )  # confidence might be negative otherwise
        confidence = confidence.squeeze()
        confidence = confidence[
            pad_info[0] : confidence.shape[0] - pad_info[1],
            pad_info[2] : confidence.shape[1] - pad_info[3],
        ]
        pred_normal = pred_normal.squeeze()  # 3, H, W
        pred_normal = pred_normal[
            :,
            pad_info[0] : pred_normal.shape[1] - pad_info[1],
            pad_info[2] : pred_normal.shape[2] - pad_info[3],
        ]
        normal_confidence = torch.clamp(
            normal_confidence, 0, 1
        )  # confidence might be negative otherwise
        normal_confidence = normal_confidence.squeeze()
        normal_confidence = normal_confidence[
            pad_info[0] : normal_confidence.shape[0] - pad_info[1],
            pad_info[2] : normal_confidence.shape[1] - pad_info[3],
        ]

        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        confidence = torch.nn.functional.interpolate(
            confidence[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        pred_normal = torch.nn.functional.interpolate(
            pred_normal[None, :, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        pred_normal /= (
            pred_normal.norm(dim=0, keepdim=True) + 1e-8
        )  # renormalize the normal
        pred_normal = pred_normal.permute(1, 2, 0)  # (H, W, 3)
        normal_confidence = torch.nn.functional.interpolate(
            normal_confidence[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        # print(f"max depth: {pred_depth.max():.2f} m, min depth: {pred_depth.min():.2f} m, scale: {canonical_to_real_scale:.2f}")    # Should be range 0.3 to 150.0
        pred_depth = (
            pred_depth * self.canonical_to_real_scale
        )  # now the depth is metric

        return (
            pred_depth.cpu().numpy(),
            confidence.cpu().numpy(),
            pred_normal.cpu().numpy(),
            normal_confidence.cpu().numpy(),
        )

    def get_feature_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get the feature embedding for the given image using the control encoder.
        Args:
            image (torch.Tensor): The input image tensor of shape (C, H, W).
        Returns:
            torch.Tensor: The feature embedding tensor.
        """
        # Ensure the image is in the correct format
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if self.feature_hacks:
            # replicate padding to avoid artifacts
            ph = self.ctrl_encoder.patch_size
            image = F.pad(image, (ph, ph, ph, ph), mode="replicate")
        # Get the feature embedding
        with torch.no_grad():
            feature_embedding = self.ctrl_encoder.encode(
                image.to(self.device).to(self.dtype),
                use_pca=getattr(self, "use_pca", False),
            )
        if self.feature_hacks:
            # Crop out the outer most patches to avoid artifacts
            feature_embedding = feature_embedding[
                :, 1:-1, 1:-1, :
            ]  # assume square patches
        return feature_embedding

    def load_models(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ctrl_net_path: str = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/CtrlNet/20250922_121309/ckpts/stage_1_res_512/latest/diffusion_pytorch_model.safetensors",
        ctrl_net_lora_path: Optional[str] = None,
        mrl_model_path: str = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/MRL/dinov3/FINAL_AE_GRAM_EQV_MOMENTS/ckpt/best.pth",
        pca_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        transformer_lora_path: Optional[str] = None,
        load_depth_model: bool = False,
    ) -> Tuple[
        FluxControlNetPipeline,
        FluxControlNetModel,
        FlowMatchEulerDiscreteScheduler,
        EmbeddingModel,
        Optional[torch.nn.Module],
    ]:
        """
        Load the ControlNet models for scene generation.
        """
        # get the FLUX pipeline components
        print(f"Loading FLUX pipeline from black-forest-labs/FLUX.1-dev")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
        )
        vae, transformer = pipe.vae, pipe.transformer
        for m in (vae, transformer, pipe.text_encoder, pipe.text_encoder_2):
            m.requires_grad_(False)
            m.eval()
        pipe.to(device).to(dtype)

        # load the DINO model and the dimensionality reduction model
        ctrl_encoder = None
        if self.raw_dino:
            print("Using raw DINOv3 features (identity mode, no adapter)")
            ctrl_encoder = EmbeddingModel.from_backbone("dinov3", device=device, dtype=dtype)
        else:
            mrl_model_path = resolve_path(mrl_model_path)
            print(f"Loading MRL model from checkpoint {mrl_model_path}")
            ctrl_encoder = EmbeddingModel.load_ckpt(mrl_model_path, pca_path=pca_path, device=device)
        for p in ctrl_encoder.parameters():
            p.requires_grad_(False)

        # prepare the controlnet
        ctrl_net_path = resolve_path(ctrl_net_path)
        print(f"Loading controlnet from checkpoint {ctrl_net_path}")
        controlnet = FluxControlNetModel.from_transformer(transformer)
        controlnet.controlnet_x_embedder = zero_module(
            torch.nn.Linear(self.max_ctrl_channels * 4, controlnet.inner_dim)
        )
        controlnet.config.x_embed_in = self.max_ctrl_channels * 4
        controlnet.config.x_embed_out = controlnet.inner_dim
        state = load_state_dict(ctrl_net_path)
        controlnet.load_state_dict(state, strict=False)
        controlnet.to(device).to(dtype)
        for p in controlnet.parameters():
            p.requires_grad_(False)

        # load the controlnet LoRA if provided
        if ctrl_net_lora_path is not None:
            ctrl_net_lora_path = resolve_path(ctrl_net_lora_path)
            print(f"Loading controlnet LoRA from {ctrl_net_lora_path}")
            controlnet.load_lora_adapter(
                ctrl_net_lora_path, adapter_name="splat_ctrl_lora", prefix=None
            )

        # load the transformer if provided (otherwise use the one from FLUX)
        if transformer_path is not None:
            transformer_path = resolve_path(transformer_path)
            print(f"Loading transformer from {transformer_path}")
            state = load_state_dict(transformer_path)
            transformer.load_state_dict(state, strict=False)
            transformer.to(device).to(dtype)
            for p in transformer.parameters():
                p.requires_grad_(False)
            pipe.transformer = transformer

        # load the transformer LoRA if provided
        if transformer_lora_path is not None:
            transformer_lora_path = resolve_path(transformer_lora_path)
            print(f"Loading transformer LoRA from {transformer_lora_path}")
            transformer.load_lora_adapter(
                transformer_lora_path, adapter_name="splat_ctrl_lora", prefix=None
            )

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        # load the depth estimation model
        if load_depth_model:
            # Doesn't appear to work with the current env!
            try:
                depth_estimation_model = (
                    torch.hub.load(
                        "yvanyin/metric3d",
                        "metric3d_vit_large",
                        pretrain=True,
                        map_location=device,
                    )
                    .to(device)
                    .eval()
                )
                for p in depth_estimation_model.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print(f"[WARN] Could not load depth estimation model: {e}")
                depth_estimation_model = None
        else:
            depth_estimation_model = None

        return pipe, controlnet, noise_scheduler, ctrl_encoder, depth_estimation_model

    def sample_view_poses(
        self,
        num_samples: int | Tuple[int, int] = 10,
        camera_sample_strategy: Literal[
            "random",
            "road",
            "perturb",
            "original",
            "object_centric",
            "orbit",
            "grid",
        ] = "original",
        rotation_std: float = 0.2,
        translation_std: float = 0.2,
        min_cam_dist: float = 1.0,
        max_cam_dist: float = 2.0,
        bounding_ellipsoid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        viewer=None,
        name_prefix: str = "gen",
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Samples a list of camera poses and (optionally) previews them in the viewer as frustums.

        Returns:
            c2ws:      list of (4,4) np.float32 camera-to-world matrices
            names:     list of viewer names used for the temporary preview frustums
        """
        scene = self.scene
        parser = self.parser
        rot_trans = parser.cam_rot_trans if hasattr(parser, "cam_rot_trans") else None

        # Remove the old temporary samples from the viewer
        if viewer is not None and len(self.tmp_names) > 0:
            viewer.remove_samples(self.tmp_names)
            self.tmp_names = []

        # 1) Sample camera poses
        c2ws = scene.sample_new_cam_poses(
            num_samples=num_samples,
            camera_sample_strategy=camera_sample_strategy,
            bounding_ellipsoid=bounding_ellipsoid,
            original_rot_trans=rot_trans,
            rotation_std=rotation_std,
            min_cam_dist=min_cam_dist,
            max_cam_dist=max_cam_dist,
            translation_std=translation_std,
            verbose=verbose,
        )
        if verbose:
            print(
                f"Sampled {len(c2ws)} new camera poses (strategy='{camera_sample_strategy}')."
            )

        # 2) Preview frustums (no images yet)
        names: List[str] = []
        if viewer is not None:
            tmp_samples = {}
            for i, c2w in enumerate(c2ws):
                # ensure (4,4) np.float32
                if isinstance(c2w, torch.Tensor):
                    c2w = c2w.detach().cpu().numpy()
                if c2w.shape == (3, 4):
                    c = np.eye(4, dtype=np.float32)
                    c[:3, :4] = c2w
                    c2w = c
                c2w = c2w.astype(np.float32)

                name = f"{name_prefix}_{i:04d}"
                names.append(name)
                tmp_samples[name] = {
                    "cam2world": c2w,
                    "K": self.K,
                    "dim": (self.H, self.W),
                    "gen": -1,
                    "filename": name,
                }
            viewer.add_samples(tmp_samples)

        self.tmp_names = names  # store for later removal
        return c2ws, names

    def cache_promt_encoding(self, prompt: str | List[str]):
        """
        Cache the prompt encoding for the given prompt.
        Args:
            prompt (str): The input text prompt.
        """
        if self.current_prompt is not None and self.current_prompt == prompt:
            return

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        with torch.no_grad():
            self.prompt_encoding = self.pipe.encode_prompt(prompts, prompt_2=prompts)
        self.current_prompt = prompt

    # ------------- Pseudo view generation function -------------
    @torch.no_grad()
    def generate_conditional_image(
        self,
        prompts=None,
        ctrl: torch.Tensor | None = None,  # (B,C,H,W) or (C,H,W)
        mask: torch.Tensor | None = None,  # (B,1,H,W) 1=change, 0=keep (image space)
        mask_scheduling: bool = False,
        mask_dilation: float = 0.0,
        mask_amplify: float = 1.5,
        rgb_guide_image: torch.Tensor | None = None,  # (B,3,H,W) in [0,1] or [-1,1]
        assumed_noise_level: float = 0.2,  # If noisy splat renders are passed here (0=no clean, 1=full noise)
        noise_frac: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 3.5,
        ctrl_guidance_scale: float = 0.75,
        ctrl_dims: int | None = None,
        dynamic_q: float | None = None,
        dynamic_q_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = 42,
    ) -> torch.Tensor:
        images = generate_conditional_image(
            pipe=self.pipe,
            transformer=self.pipe.transformer,
            controlnet=self.controlnet,
            noise_scheduler=self.noise_scheduler,
            prompt_encoding=self.prompt_encoding,
            prompts=prompts,
            ctrl=ctrl,
            mask=mask,
            mask_scheduling=mask_scheduling,
            mask_dilation=mask_dilation,
            mask_amplify=mask_amplify,
            rgb_guide_image=rgb_guide_image,
            assumed_noise_level=assumed_noise_level,
            noise_frac=noise_frac,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            ctrl_guidance_scale=ctrl_guidance_scale,
            ctrl_dims=ctrl_dims,
            dynamic_q=dynamic_q,
            dynamic_q_alpha=dynamic_q_alpha,
            device=device if device is not None else self.device,
            dtype=dtype,
            seed=seed,
        )
        return images

    @torch.no_grad()
    def generate_sample(
        self,
        cam: CameraState = None,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        text_ctrl_strength: float = 3.5,
        rgb_guidance_noise_frac: float | None = None,
        dynamic_q: float | None = None,
        prompt: str = "",
        seed: int = 42,
        predict_depth: bool = False,
        predict_features: bool = True,
        masked: bool = False,
        debug: bool = False,
        artifact_removal: float = 0.0,
        mask_scheduling: bool = False,
        mask_dilation: float = 0.0,
        mask_amplify: float = 1.5,
    ):
        scene = self.scene
        feature = None
        if feature_ctrl_strength is not None:
            feat = scene.feature_render_fn(
                camera_state=cam,
                img_wh=(self.W, self.H),
                raw=True,
                downsample_factor=(
                    8
                    if self.feature_upsampling_mode == "rendering"
                    else self.ctrl_encoder.patch_size
                ),
            )
            feature = feat.to(dtype=self.dtype, device=self.device).permute(0, 3, 1, 2)
            if feature.shape[1] > feature_ctrl_dims:
                feature = mask_suffix_channels(
                    feature, feature.shape[1] - feature_ctrl_dims
                )
            feature = feature[:, : self.max_ctrl_channels, :, :]  # crop to max channels
            feature = torch.clamp(feature, -1.0, 1.0)

        if masked:
            mask = scene.selection_render_fn(
                camera_state=cam,
                img_wh=(self.W, self.H),
                raw=True,
            )
            mask = mask.to(dtype=self.dtype, device=self.device).permute(0, 3, 1, 2)
            # amplify the mask a bit (to avoid artifacts on the edges)
            # mask = mask * 1.5
            mask = mask.clamp(0.0, 1.0)
        else:
            mask = None

        # do the artifact removal with prob artifact_removal
        artifact_removal = torch.rand(1, generator=self._rng).item() < artifact_removal

        if artifact_removal:
            if mask is None:
                print(
                    "WARNING: artifact_removal is True but no mask provided, skipping artifact removal."
                )
            else:
                # get a random background color
                background_colors = list(self.background_color_options.keys())
                background_color_name = background_colors[
                    torch.randint(
                        0, len(background_colors), (1,), generator=self._rng
                    ).item()
                ]
                background_color = self.background_color_options[background_color_name]
                background_color_t = torch.tensor(
                    background_color, device=self.device
                ).view(1, 3, 1, 1)
                background_color_t = background_color_t.repeat(1, 1, self.H, self.W)
                # encode and blend with the ctrl image
                bg_feature = self.get_feature_embedding(background_color_t).permute(
                    0, 3, 1, 2
                )
                if feature is not None:
                    bg_feature = F.interpolate(
                        bg_feature,
                        size=feature.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    bg_feature = bg_feature[
                        :, : feature.shape[1], :, :
                    ]  # crop to match
                    feature_mask = F.interpolate(
                        mask, size=feature.shape[2:], mode="area"
                    )
                    feature = feature * feature_mask + bg_feature * (1 - feature_mask)
                if prompt is None:
                    prompt = ""
                prompt += (
                    f" in front of a plain, uniform {background_color_name} background"
                )

        if (
            masked
            or rgb_guidance_noise_frac is not None
            and rgb_guidance_noise_frac < 1.0
        ):
            rgb, _ = scene.render_fn(
                camera_state=cam,
                img_wh=(self.W, self.H),
                raw=True,
                background_color=background_color if artifact_removal else None,
            )
            rgb = rgb.to(dtype=self.dtype, device=self.device).permute(0, 3, 1, 2)
            rgb = torch.clamp(rgb, 0.0, 1.0)
        else:
            rgb = None

        # cache the prompt encoding
        self.cache_promt_encoding(prompt)

        # 2a) Diffusion generation
        image = self.generate_conditional_image(
            prompts=None,  # use cached prompt encoding
            text_guidance_scale=text_ctrl_strength,
            ctrl=feature,
            height=self.H,
            width=self.W,
            ctrl_guidance_scale=feature_ctrl_strength,
            rgb_guide_image=rgb,
            mask=mask,
            mask_scheduling=mask_scheduling,
            mask_dilation=mask_dilation,
            mask_amplify=mask_amplify,
            noise_frac=rgb_guidance_noise_frac,
            dynamic_q=dynamic_q,
            seed=seed,
            device=self.device,
            dtype=self.dtype,
        )

        # Convert to uint8 BGR for saving
        np_image = image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)

        # 2b) Build sample dict
        rotation_wxyz, translation = pose_to_rotation_translation(cam.c2w)
        sample = {
            "intrinsics": self.intrinsics,
            "rotation_wxyz": rotation_wxyz,
            "translation": translation,
            "prompt": prompt,
            "artifact_removal": artifact_removal,
            "background_color": background_color if artifact_removal else [0, 0, 0],
            "image": np_image,
        }

        sample["datamap"] = {}

        if masked:
            # also save the inpainting mask
            sample["datamap"]["inpainting_mask"] = (
                mask.squeeze(0).float().cpu().numpy() * 255
            ).astype(np.uint8)

        # 2c) Datamap generation (on-the-fly)
        if predict_features:
            feature_emb = self.get_feature_embedding(
                image
            )  # expects tensor from pipeline
            sample["datamap"]["reduced_embeddings"] = (
                feature_emb.squeeze(0).float().cpu().numpy()
            )

        if predict_depth:
            depth, conf, normal, norm_conf = self.get_metric3d_depth(np_image)
            bg_mask = depth < (self.max_pred_depth)
            datamap = {
                "depth": depth.astype(np.float32),
                "depth_confidence": conf.astype(np.float32),
                "normals": normal.astype(np.float32),
                "normal_confidence": norm_conf.astype(np.float32),
                "bg_mask": bg_mask.astype(np.bool_),
            }
            sample["datamap"].update(datamap)

        if debug:
            # dump all the rendered and generated images for debugging
            debug_sample = {}
            if feature is not None:
                debug_sample["feature_render"] = (
                    feature.squeeze(0).float().cpu().numpy()
                )
            if rgb is not None:
                debug_sample["rgb_render"] = (
                    rgb.squeeze(0).float().cpu().numpy()
                )  # in [0,1]
            if mask is not None:
                debug_sample["inpainting_mask"] = mask.squeeze(0).float().cpu().numpy()
            if artifact_removal:
                debug_sample["background_color"] = np.array(
                    background_color, dtype=np.float32
                ).reshape(1, 1, 3)
            if np_image is not None:
                debug_sample["generated_image"] = np_image
            if predict_depth:
                debug_sample["predicted_depth"] = depth.astype(np.float32)
                debug_sample["depth_confidence"] = conf.astype(np.float32)
                debug_sample["predicted_normals"] = normal.astype(np.float32)
                debug_sample["normal_confidence"] = norm_conf.astype(np.float32)
            if feature_emb is not None:
                debug_sample["predicted_features"] = (
                    feature_emb.squeeze(0).float().cpu().numpy()
                )
            # timestamp to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("debug", exist_ok=True)
            np.savez_compressed(f"debug/debug_sample_{timestamp}.npz", **debug_sample)

        return sample, np_image

    def generate_samples_from_poses(
        self,
        c2ws: List[np.ndarray],
        pose_names: Optional[List[str]] = None,
        *,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        text_ctrl_strength: float = 3.5,
        rgb_guide_noise_frac: float | None = None,
        dynamic_q: float | None = None,        
        masked: bool = False,
        mask_scheduling: bool = False,
        mask_dilation: float = 0.0,
        mask_amplify: float = 1.5,
        segmentation_anchors: torch.Tensor | None = None,  # (N,C)
        segmentation_dims: int = 16,
        predict_depth: bool = False,
        predict_features: bool = True,
        artifact_removal: float = 0.0,
        debug: bool = False,
        seed: int = 42,
        new_data_dir: str | None = None,
        prompt: str = "",
        viewer=None,
    ) -> str:
        """
        Generates images for the provided list of poses, saves them immediately to the dataset,
        and (optionally) updates the viewer frustums with the produced images.

        Returns:
            new_scene (str): parser.new_metadata identifier
        """
        parser = self.parser

        # Output dirs
        datadir = (
            Path(new_data_dir) if new_data_dir is not None else Path(parser.datadir)
        )
        image_dir = datadir / parser.metadata["subdirs"]["image_dir"]
        datamap_dir = datadir / parser.metadata["subdirs"]["datamap_dir"]
        image_dir.mkdir(parents=True, exist_ok=True)
        datamap_dir.mkdir(parents=True, exist_ok=True)

        # New generation (matches add_samples semantics)
        parser.gen_cnt += 1

        # Camera intrinsics for states
        fov = self.fov

        n = 0

        # prepare segmentation anchors
        if segmentation_anchors is not None:
            pos_anchors, neg_anchors = segmentation_anchors
            # truncate to max channels
            pos_anchors = pos_anchors[:, :segmentation_dims]
            neg_anchors = neg_anchors[:, :segmentation_dims]
            pos_anchors = F.normalize(pos_anchors, dim=-1).to(device=self.device)
            neg_anchors = F.normalize(neg_anchors, dim=-1).to(device=self.device)

        with tqdm(total=len(c2ws), desc="Generating samples") as pbar:
            for i, (c2w, tmp_name) in enumerate(zip(c2ws, pose_names)):
                # Ensure ndarray (4,4)
                if isinstance(c2w, torch.Tensor):
                    c2w = c2w.detach().cpu().numpy()
                if c2w.shape == (3, 4):
                    c = np.eye(4, dtype=np.float32)
                    c[:3, :4] = c2w
                    c2w = c
                c2w = c2w.astype(np.float32)

                cam = CameraState(fov=fov, aspect=1.0, c2w=c2w)

                # Generate image
                sample, np_image = self.generate_sample(
                    feature_ctrl_strength=feature_ctrl_strength,
                    feature_ctrl_dims=feature_ctrl_dims,
                    text_ctrl_strength=text_ctrl_strength,
                    rgb_guidance_noise_frac=rgb_guide_noise_frac,
                    dynamic_q=dynamic_q,
                    masked=masked,
                    artifact_removal=artifact_removal,
                    mask_scheduling=mask_scheduling,
                    mask_dilation=mask_dilation,
                    mask_amplify=mask_amplify,
                    cam=cam,
                    prompt=prompt,
                    seed=seed,
                    predict_depth=predict_depth,
                    predict_features=predict_features,
                    debug=debug,
                )

                # Create an segmentation alpha mask if the anchors are provided
                if segmentation_anchors is not None and sample.get("datamap", None):
                    reduced_embeddings = sample["datamap"].get(
                        "reduced_embeddings", None
                    )  # (H, W, C)
                    if reduced_embeddings is not None:
                        emb = (
                            torch.from_numpy(reduced_embeddings)
                            .float()
                            .to(device=self.device)
                        )  # (H,W,C)
                        # truncate to max channels
                        emb = emb[..., :segmentation_dims]
                        emb = emb.unsqueeze(0)  # (1,H,W,C)
                        emb = F.normalize(emb, dim=-1)
                        pos_sim = torch.einsum(
                            "bhwc,nc->bhwn", emb, pos_anchors
                        )  # (1,H,W,N)
                        neg_sim = torch.einsum(
                            "bhwc,nc->bhwn", emb, neg_anchors
                        )  # (1,H,W,N)
                        if pos_sim.shape[-1] == 0:
                            pos_sim = torch.ones(
                                (1, self.H, self.W), device=self.device
                            )
                        else:
                            pos_sim, _ = pos_sim.max(dim=-1)  # (1,H,W)
                        if neg_sim.shape[-1] == 0:
                            neg_sim = torch.zeros_like(pos_sim)
                        else:
                            neg_sim, _ = neg_sim.max(dim=-1)  # (1,H,W)
                        sim = pos_sim - neg_sim  # (1,H,W)
                        sim = sim.squeeze(0)  # (H,W)
                        sim = torch.clamp(sim, min=0.0)
                        # min max normalize to [0,1]
                        # seg_mask = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
                        seg_mask = sim
                        seg_mask = (
                            F.interpolate(
                                seg_mask.unsqueeze(0).unsqueeze(0),
                                size=(self.H, self.W),
                                mode="bilinear",
                                align_corners=False,
                            )
                            .squeeze(0)
                            .squeeze(0)
                        )  # (H,W)
                        seg_mask = seg_mask * 255
                        seg_mask = seg_mask.cpu().numpy().astype(np.uint8)
                        sample["datamap"]["loss_mask"] = seg_mask

                # Persist single sample
                final_name = parser.add_sample(sample=sample, idx=i, datadir=datadir)
                pbar.set_description(f"Generating ({i+1}/{len(c2ws)})")
                pbar.set_postfix_str(final_name)
                pbar.update(1)
                n += 1

                # Update viewer frustum image (replace preview)
                if viewer is not None:
                    img = np_image[:, :, ::-1] / 255.0  # BGR->RGB normalized
                    # add the stuff from the datamap if available
                    depth = sample.get("datamap", {}).get("depth", None)
                    normal = sample.get("datamap", {}).get("normal", None)
                    feature = sample.get("datamap", {}).get("reduced_embeddings", None)
                    bg_mask = sample.get("datamap", {}).get("bg_mask", None)
                    loss_mask = sample.get("datamap", {}).get("loss_mask", None)
                    viz_sample = {
                        "cam2world": c2w,
                        "K": self.K,
                        "dim": (self.H, self.W),
                        "img": img,
                        "depth": depth,
                        "bg_mask": bg_mask,
                        "loss_mask": loss_mask,
                        "normal": normal,
                        "feature": feature,
                        "filename": final_name,
                        "gen": parser.gen_cnt,
                    }
                    try:
                        viewer.remove_samples([tmp_name])
                    except Exception:
                        pass
                    viewer.add_samples({final_name: viz_sample})

                # check for cancel flag
                if viewer is not None and viewer.cancel:
                    print("Generation cancelled by user.")
                    break

        # Save metadata once
        parser.save_metadata(datadir=datadir)
        print(
            f"Generated {n} samples. New scene metadata saved as {parser.new_metadata}"
        )
        return parser.new_metadata, parser.gen_cnt

    def generate_new_samples(
        self,
        num_samples: int,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        depth_ctrl_strength: float = None,  # kept for BC; ignored
        semseg_ctrl_strength: float = None,  # kept for BC; ignored
        rgb_guide_noise_frac: float | None = None,
        rgb_extra_noise_frac: float | None = None,
        camera_sample_strategy: Literal[
            "random", "road", "perturb", "original"
        ] = "original",
        rotation_std: float = 0.2,
        translation_std: float = 0.2,
        predict_depth: bool = False,
        predict_features: bool = True,
        debug: bool = False,
        seed: int = 42,
        new_data_dir: str | None = None,
        prompt: str = "A high resolution, realistic DSLR image of a street scene with vehicles, people and structures.",
        viewer=None,
    ) -> str:
        """
        Backward-compatible wrapper:
        1) Samples poses (+ previews in viewer if provided)
        2) Generates images for those poses and saves them immediately.

        Returns:
            new_scene (str): parser.new_metadata identifier
        """
        # 1) Sample poses (and preview as red frustums)
        c2ws, names = self.sample_view_poses(
            num_samples=num_samples,
            camera_sample_strategy=camera_sample_strategy,
            rotation_std=rotation_std,
            translation_std=translation_std,
            viewer=viewer,
            name_prefix="gen",
        )

        # 2) Generate images for sampled poses
        new_scene = self.generate_samples_from_poses(
            c2ws=c2ws,
            pose_names=names,
            feature_ctrl_strength=feature_ctrl_strength,
            feature_ctrl_dims=feature_ctrl_dims,
            rgb_guide_noise_frac=rgb_guide_noise_frac,
            rgb_extra_noise_frac=rgb_extra_noise_frac,
            predict_depth=predict_depth,
            predict_features=predict_features,
            debug=debug,
            seed=seed,
            new_data_dir=new_data_dir,
            prompt=prompt,
            viewer=viewer,
        )

        return new_scene

    # -------- SDS loss --------
    def sds_step_loss(
        self,
        imgs: torch.Tensor,  # (B,3,H,W) in [0,1], requires_grad
        ctrl: torch.Tensor | None,  # (B,Cc,H_lat,W_lat) or None  (latent-res!)
        *,
        prompts: list[str] | None,
        text_guidance_scale: float = 3.5,
        ctrl_guidance_scale: float = 0.7,
        total_steps: int = 1000,
        n_levels: int = 1,  # K: how many timesteps per image
        bias_exponent: float = 2.0,  # >1 biases toward early/high-noise
        device: torch.device,
        dtype: torch.dtype,
        seed: int | None = None,
    ) -> torch.Tensor:
        B, _, H, W = imgs.shape
        vae = self.pipe.vae
        self.noise_scheduler.set_timesteps(total_steps, device=device)

        # --- VAE encode to latents ([-1,1] expected) ---
        x = (imgs * 2.0 - 1.0).to(device=vae.device, dtype=vae.dtype)
        lat = vae.encode(x).latent_dist.sample()
        lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
        lat = lat.to(device=device, dtype=dtype)  # (B,C,H_lat,W_lat)

        # --- Shapes / packing ---
        C_lat, H_lat, W_lat = lat.shape[1], lat.shape[2], lat.shape[3]
        packed = FluxControlNetPipeline._pack_latents(
            lat, B, C_lat, H_lat, W_lat
        )  # (B,*,*,*)
        packed = packed.repeat_interleave(n_levels, dim=0)  # (B*K,...)
        Bk = packed.shape[0]

        # --- Control (already at latent res) ---
        if ctrl is not None:
            assert ctrl.shape[:2] == (B, ctrl.shape[1]) and ctrl.shape[-2:] == (
                H_lat,
                W_lat,
            ), "ctrl must be (B,Cc,H_lat,W_lat)"
            ctrl_latents = FluxControlNetPipeline._pack_latents(
                ctrl.to(device=device, dtype=dtype), B, ctrl.shape[1], H_lat, W_lat
            )
            ctrl_latents = ctrl_latents.repeat_interleave(n_levels, dim=0)
        else:
            ctrl_latents = None

        # --- Sample biased timesteps (earlier → higher noise) ---
        T = len(self.noise_scheduler.timesteps)
        idx = torch.clamp(
            (torch.rand(B, n_levels, device=device) ** bias_exponent) * T, max=T - 1
        ).long()
        idx = idx.reshape(-1)  # (B*K,)
        t_sched = self.noise_scheduler.timesteps[idx].to(
            device=device, dtype=dtype
        )  # (B*K,)
        sigmas = self.noise_scheduler.sigmas[idx].to(
            device=device, dtype=dtype
        )  # (B*K,)

        # --- Build noisy latents x_t = (1-σ) * x0 + σ * ε (σ already in scheduler’s scale) ---
        gen = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )
        noise = torch.randn(
            packed.shape, device=packed.device, dtype=packed.dtype, generator=gen
        )
        sig = sigmas.view(Bk, *([1] * (packed.ndim - 1)))
        noisy = (1.0 - sig) * packed + sig * noise

        # --- Text encodings (broadcast to B, then repeat K) ---
        if prompts is not None:
            prompt_embeds, pooled, text_ids = self.pipe.encode_prompt(
                prompts, prompt_2=prompts
            )
        elif getattr(self, "prompt_encoding", None) is not None:
            prompt_embeds, pooled, text_ids = self.prompt_encoding
        else:
            raise ValueError(
                "Either prompts or cached prompt_encoding must be provided."
            )
        if prompt_embeds.shape[0] != B:
            assert prompt_embeds.shape[0] == 1, "Prompt batch mismatch."
            prompt_embeds = prompt_embeds.expand(B, *prompt_embeds.shape[1:])
            pooled = pooled.expand(B, *pooled.shape[1:])
            text_ids = text_ids.expand(B, *text_ids.shape[1:])
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(
            n_levels, dim=0
        )
        pooled = pooled.to(device=device, dtype=dtype).repeat_interleave(
            n_levels, dim=0
        )
        text_ids = text_ids.to(device=device)

        # --- IDs / guidance ---
        image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            batch_size=Bk,
            height=H_lat // 2,
            width=W_lat // 2,
            device=device,
            dtype=dtype,
        )
        g_vec = torch.full((Bk,), text_guidance_scale, device=device, dtype=dtype)
        t_model = t_sched / 1000.0  # (B*K,)

        # Debug print all the relevant shapes
        # print(f"SDS step shapes: latents {packed.shape}, noisy {noisy.shape}, prompt_embeds {prompt_embeds.shape}, pooled {pooled.shape}, text_ids {text_ids.shape}, ctrl_latents {ctrl_latents.shape if ctrl_latents is not None else None}, image_ids {image_ids.shape}")

        # --- ControlNet ---
        if ctrl_latents is not None:
            ctrl_samples, ctrl_singles = self.controlnet(
                hidden_states=noisy,
                timestep=t_model,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                txt_ids=text_ids,
                img_ids=image_ids,
                guidance=g_vec,
                return_dict=False,
                controlnet_cond=ctrl_latents,
            )
            ctrl_samples = (
                [s * ctrl_guidance_scale for s in ctrl_samples]
                if ctrl_samples
                else None
            )
            ctrl_singles = (
                [s * ctrl_guidance_scale for s in ctrl_singles]
                if ctrl_singles
                else None
            )
        else:
            ctrl_samples, ctrl_singles = None, None

        # --- Transformer → v_pred ---
        (v_pred,) = self.pipe.transformer(
            hidden_states=noisy,
            timestep=t_model,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            txt_ids=text_ids,
            img_ids=image_ids,
            guidance=g_vec,
            return_dict=False,
            controlnet_block_samples=(
                [s.to(dtype=dtype) for s in ctrl_samples] if ctrl_samples else None
            ),
            controlnet_single_block_samples=(
                [s.to(dtype=dtype) for s in ctrl_singles] if ctrl_singles else None
            ),
        )

        # --- SDS target + loss over all (B*K) samples ---
        v_target = noise - packed
        loss = F.mse_loss(v_pred, v_target, reduction="mean")
        return loss

    def generate_sds_loss(
        self,
        cams: List[CameraState],  # used to render features for ctrl
        prompts: list[str] | None,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        text_ctrl_strength: float = 3.5,
        timestep_bias_exponent: float = 10.0,  # >1 biases toward early/high-noise
        n_timesteps_per_image: int = 1,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Score Distillation Sampling loss (v-parameterization) w/ FLUX + ControlNet features.
        Gradients flow to `rgb`. Returns (loss, aux).
        """

        device = torch.device(self.device)
        dtype = self.dtype

        # 1) render the rgb images
        imgs = []
        ctrls = []
        for cam in cams:
            img, info = self.scene.render_fn(
                camera_state=cam, img_wh=(self.W, self.H), raw=True
            )
            img = img.permute(0, 3, 1, 2).to(
                device=device, dtype=dtype
            )  # (B=1, 3, H, W)
            imgs.append(img)

            # 2) Prepare feature control latents at latent resolution (B=1 or B)
            if feature_ctrl_strength is not None:
                # Render features from Scene at (H,W)
                # NOTE: Downsampling is fixed
                ctrl = self.scene.feature_render_fn(
                    camera_state=cam,
                    img_wh=(self.W, self.H),
                    raw=True,
                    downsample_factor=(
                        8
                        if self.feature_upsampling_mode == "rendering"
                        else self.ctrl_encoder.patch_size
                    ),
                )
                ctrl = ctrl.permute(0, 3, 1, 2).to(
                    device=device, dtype=dtype
                )  # (1, Cc, H, W)
                ctrl = (
                    mask_suffix_channels(ctrl, ctrl.shape[2] - feature_ctrl_dims)
                    if ctrl.shape[2] > feature_ctrl_dims
                    else ctrl
                )
                ctrl = torch.clamp(ctrl, -1.0, 1.0)
                ctrl = ctrl[:, : self.max_ctrl_channels, :, :]  # crop to max channels
                ctrls.append(ctrl)

        # stack to (B, C, H, W)
        imgs = torch.cat(imgs, dim=0)
        ctrl = torch.cat(ctrls, dim=0) if feature_ctrl_strength is not None else None

        # 4) Predict v at a random timestep (and get target)
        loss = self.sds_step_loss(
            imgs=imgs,
            ctrl=ctrl,
            prompts=prompts,
            text_guidance_scale=text_ctrl_strength,
            ctrl_guidance_scale=feature_ctrl_strength,
            total_steps=1000,
            n_levels=n_timesteps_per_image,
            bias_exponent=timestep_bias_exponent,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        return loss, info

    def batch_sds_loss(
        self,
        batch_size: int,
        prompts: list[str] | str,
        camera_sample_strategy: Literal[
            "random", "road", "perturb", "original"
        ] = "original",
        rotation_std: float = 0.2,
        translation_std: float = 0.2,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        text_ctrl_strength: float = 3.5,
        timestep_bias_exponent: float = 10.0,  # >1 biases toward early/high-noise
        n_timesteps_per_image: int = 1,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the average SDS loss over a batch of camera states.
        Returns (loss, aux).
        """
        # Sample some camera poses
        cam_poses, _ = self.sample_view_poses(
            num_samples=batch_size,
            camera_sample_strategy=camera_sample_strategy,
            rotation_std=rotation_std,
            translation_std=translation_std,
            viewer=None,
            name_prefix="sds",
            verbose=False,
        )

        cam_poses = self.tensor_to_camera_state(
            torch.from_numpy(np.stack(cam_poses)).to(self.device)
        )
        self.cache_promt_encoding(prompts)

        loss, info = self.generate_sds_loss(
            cams=cam_poses,
            prompts=None,  # use cached
            feature_ctrl_strength=feature_ctrl_strength,
            feature_ctrl_dims=feature_ctrl_dims,
            text_ctrl_strength=text_ctrl_strength,
            timestep_bias_exponent=timestep_bias_exponent,
            n_timesteps_per_image=n_timesteps_per_image,
            seed=seed,
        )

        info.update(
            {
                "cam_poses": cam_poses,
            }
        )

        return loss, info

    @staticmethod
    def _odd_ksize_from_radius(r: int) -> int:
        return max(1, 2 * max(0, int(r)) + 1)

    @staticmethod
    def _gaussian_1d_kernel(radius: int, sigma: float, device, dtype):
        k = SampleGenerator._odd_ksize_from_radius(radius)
        half = k // 2
        x = torch.arange(-half, half + 1, device=device, dtype=dtype)
        ker = torch.exp(-(x**2) / (2.0 * max(1e-6, sigma) ** 2))
        ker = ker / ker.sum().clamp_min(1e-12)
        return ker.view(1, 1, -1), k  # [1,1,K

    @torch.no_grad()
    def dilate_then_feather_mask(
        mask: torch.Tensor,
        dilate_rel: float = 0.5,
        blur_rel: float = 0.25,
    ):
        # Normalize to (B,1,H,W)
        if mask.dim() == 2:
            mask = mask[None, None]
        elif mask.dim() == 3:
            mask = mask[:1][None] if mask.shape[0] != 1 else mask[None]
        elif mask.dim() != 4:
            raise ValueError("mask must be (H,W), (1,H,W), or (B,1,H,W)")

        B, C, H, W = mask.shape
        device, dtype = mask.device, mask.dtype
        bin_mask = (mask > 0).to(dtype)

        max_radius = max(mask.shape[-1], mask.shape[-2])
        dilate_radius = int(max_radius * dilate_rel)
        sigma = max_radius * blur_rel
        gauss_radius = int(max(1, round(3.0 * sigma))) if sigma > 0 else 1
        ksize = SampleGenerator._odd_ksize_from_radius(gauss_radius)

        # Dilation (replicate pad)
        if dilate_radius > 0:
            pad = (dilate_radius, dilate_radius, dilate_radius, dilate_radius)
            x = F.pad(bin_mask, pad, mode="replicate")
            k = 2 * dilate_radius + 1
            dilated = F.max_pool2d(x, kernel_size=k, stride=1, padding=0)[:, :, :H, :W]
        else:
            dilated = bin_mask
        dilated_hard = (dilated > 0).to(dtype)

        # Feather (separable Gaussian with replicate pad)
        if sigma > 0:
            g1d, _ = SampleGenerator._gaussian_1d_kernel(
                gauss_radius, sigma, device=device, dtype=dtype
            )
            kx = g1d.view(1, 1, 1, ksize)
            ky = g1d.view(1, 1, ksize, 1)
            x = F.pad(
                dilated_hard, (gauss_radius, gauss_radius, 0, 0), mode="replicate"
            )
            soft = F.conv2d(x, kx)
            soft = F.pad(soft, (0, 0, gauss_radius, gauss_radius), mode="replicate")
            soft = F.conv2d(soft, ky)
            soft = soft[:, :, :H, :W].clamp(0.0, 1.0)
        else:
            soft = dilated_hard

        meta = {
            "dilate_radius_px": dilate_radius,
            "gaussian_sigma_px": sigma,
            "gaussian_kernel_size": ksize,
        }
        return dilated_hard, soft, meta


# --- Generation function ---
def mask_suffix_channels(
    x: torch.Tensor,
    num_channels: Optional[Union[int, torch.Tensor, Sequence[int]]] = None,
    channels_last: bool = False,
) -> torch.Tensor:
    """
    Zero out a suffix of channels in the input tensor.

    If num_channels is None, for each sample i we pick Ki ∈ [0..C] at random.
    If num_channels is an int K, we mask the last K channels for every sample.
    If num_channels is a length-B Sequence or Tensor, each sample i masks the last Ki channels.

    Args:
        x: Tensor of shape (B, C, H, W).
        num_channels: None | int | Tensor[B] | Sequence[int].
    Returns:
        Tensor of same shape as x, with the specified suffix channels zeroed.
    """
    no_batch = x.ndim == 3
    if no_batch:
        x = x.unsqueeze(0)  # add batch dim
    if channels_last:
        x = x.permute(0, 3, 1, 2)  # to (B, C, H, W)
    B, C, _, _ = x.shape
    device, dtype = x.device, x.dtype

    # Build the per-sample K array
    if num_channels is None:
        Ks = torch.randint(0, C + 1, (B,), device=device)
    else:
        if isinstance(num_channels, int):
            Ks = torch.full((B,), num_channels, device=device, dtype=torch.long)
        else:
            Ks = torch.as_tensor(num_channels, device=device, dtype=torch.long)
            if Ks.dim() == 0:
                Ks = Ks.view(1).expand(B)
        Ks = Ks.clamp(0, C)

    # Create a (B, C) boolean keep-mask
    arange = torch.arange(C, device=device).view(1, C)  # shape (1, C)
    thresholds = (C - Ks).view(B, 1)  # shape (B, 1)
    keep = arange < thresholds  # shape (B, C)

    # Broadcast to (B, H, W, C) and mask
    mask = keep.view(B, C, 1, 1).to(dtype)  # shape (B,1,1,C)
    ret = x * mask
    if channels_last:
        ret = ret.permute(0, 2, 3, 1)  # back to (B, H, W, C)
    if no_batch:
        ret = ret.squeeze(0)
    return ret


def _odd_ksize_from_radius(r: int) -> int:
    return max(1, 2 * max(0, int(r)) + 1)


def _gaussian_1d_kernel(radius: int, sigma: float, device, dtype):
    k = _odd_ksize_from_radius(radius)
    half = k // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    ker = torch.exp(-(x**2) / (2.0 * max(1e-6, sigma) ** 2))
    ker = ker / ker.sum().clamp_min(1e-12)
    return ker.view(1, 1, -1), k  # [1,1,K


@torch.no_grad()
def dilate_then_feather_mask(
    mask: torch.Tensor,
    dilate_rel: float = 0.5,
    blur_rel: float = 0.25,
):
    # Normalize to (B,1,H,W)
    if mask.dim() == 2:
        mask = mask[None, None]
    elif mask.dim() == 3:
        mask = mask[:1][None] if mask.shape[0] != 1 else mask[None]
    elif mask.dim() != 4:
        raise ValueError("mask must be (H,W), (1,H,W), or (B,1,H,W)")

    B, C, H, W = mask.shape
    device, dtype = mask.device, mask.dtype
    # bin_mask = (mask > 0).to(dtype)

    max_radius = max(mask.shape[-1], mask.shape[-2])
    dilate_radius = int(max_radius * dilate_rel)
    sigma = max_radius * blur_rel
    gauss_radius = int(max(1, round(3.0 * sigma))) if sigma > 0 else 1
    ksize = _odd_ksize_from_radius(gauss_radius)

    # Dilation (replicate pad)
    if dilate_radius > 0:
        pad = (dilate_radius, dilate_radius, dilate_radius, dilate_radius)
        x = F.pad(mask, pad, mode="replicate")
        k = 2 * dilate_radius + 1
        dilated = F.max_pool2d(x, kernel_size=k, stride=1, padding=0)[:, :, :H, :W]
    else:
        dilated = mask
    # dilated = (dilated > 0).to(dtype)

    # Feather (separable Gaussian with replicate pad)
    if sigma > 0:
        g1d, _ = _gaussian_1d_kernel(gauss_radius, sigma, device=device, dtype=dtype)
        kx = g1d.view(1, 1, 1, ksize)
        ky = g1d.view(1, 1, ksize, 1)
        x = F.pad(dilated, (gauss_radius, gauss_radius, 0, 0), mode="replicate")
        soft = F.conv2d(x, kx)
        soft = F.pad(soft, (0, 0, gauss_radius, gauss_radius), mode="replicate")
        soft = F.conv2d(soft, ky)
        soft = soft[:, :, :H, :W].clamp(0.0, 1.0)
    else:
        soft = dilated

    meta = {
        "dilate_radius_px": dilate_radius,
        "gaussian_sigma_px": sigma,
        "gaussian_kernel_size": ksize,
    }
    return dilated, soft, meta


@torch.no_grad()
def get_timestep_idx(noise_scheduler, noise_frac: float) -> torch.Tensor:
    """
    Map noise fraction f∈[0,1] to the scheduler index after set_timesteps(...):
    f=0 → lowest-σ (near clean) → last index
    f=1 → highest-σ (max noise)  → first index
    Works for both ascending or descending σ arrays.
    """
    f = float(max(0.0, min(1.0, noise_frac)))
    sigmas = noise_scheduler.sigmas
    n = len(sigmas) - 1
    # If σ descends: [σ_max,...,σ_min] → f=1→idx=0, f=0→idx=n
    # If σ ascends:  [σ_min,...,σ_max] → f=1→idx=n, f=0→idx=0
    descending = sigmas[0] > sigmas[-1]
    idx = ((1.0 - f) * n) if descending else (f * n)
    idx = int(round(idx))
    return torch.as_tensor(idx, device=sigmas.device).long().clamp(0, n)


@torch.no_grad()
def generate_conditional_image(
    pipe,
    transformer,
    controlnet,
    noise_scheduler,
    prompts=None,
    prompt_encoding=None,
    ctrl: torch.Tensor | None = None,  # (B,C,H,W) or (C,H,W)
    mask: torch.Tensor | None = None,  # (B,1,H,W) 1=change, 0=keep (image space)
    mask_scheduling: bool = True,  # progressively shrink mask
    mask_dilation: float = 0.0,  # relative to max(H,W)
    mask_amplify: float = 1.0,
    rgb_guide_image: torch.Tensor | None = None,  # (B,3,H,W) in [0,1] or [-1,1]
    assumed_noise_level: (
        float | None
    ) = None,  # if provided, the passed image is treated as alrady noisy at this level
    noise_frac: float = 1.0,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 28,
    text_guidance_scale: float = 3.5,
    ctrl_guidance_scale: float = 0.75,
    ctrl_dims: Optional[int] = None,
    dynamic_q: float | None = 0.9,
    dynamic_q_alpha: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int | None = 42,
) -> torch.Tensor:
    # ----------------- setup / shapes -----------------
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    B_ctrl = ctrl.shape[0] if (ctrl is not None and ctrl.ndim == 4) else 1
    B_text = (
        len(prompts)
        if (prompts is not None and isinstance(prompts, (list, tuple)))
        else 1
    )
    B = max(B_ctrl, B_text)
    if ctrl is not None:
        if ctrl.ndim == 3:
            ctrl = ctrl.unsqueeze(0)
        if ctrl.shape[0] != B:
            ctrl = ctrl.repeat(B, 1, 1, 1)
        if ctrl_dims is not None:
            ctrl = mask_suffix_channels(ctrl, num_channels=ctrl.shape[1] - ctrl_dims)

    if prompts is not None:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(prompts, prompt_2=prompts)
        if prompt_embeds.shape[0] != B:
            prompt_embeds = prompt_embeds.expand(B, *prompt_embeds.shape[1:])
            pooled = pooled.expand(B, *pooled.shape[1:])
            text_ids = text_ids.expand(B, *text_ids.shape[1:])
    elif prompt_encoding is not None:
        prompt_embeds, pooled, text_ids = prompt_encoding
        if prompt_embeds.shape[0] != B:
            prompt_embeds = prompt_embeds.expand(B, *prompt_embeds.shape[1:])
            pooled = pooled.expand(B, *pooled.shape[1:])
            text_ids = text_ids.expand(B, *text_ids.shape[1:])
    else:
        raise ValueError("Either 'prompts' or 'prompt_encoding' must be provided.")

    # VAE spatial scale
    vae_spacial_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    H_lat = height // vae_spacial_scale_factor
    W_lat = width // vae_spacial_scale_factor
    C_lat = pipe.vae.config.latent_channels

    # ----------------- Control latents -----------------
    # NOTE: If bg artifact removal is on, ctrl is expected to have the plain bg color encoded
    if ctrl is None:
        ctrl_latents = None
    else:
        if ctrl.shape[1] == 3 and ctrl.shape[-2:] == (height, width):
            ctrl_img = ctrl.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
            ctrl_x0 = pipe.vae.encode(ctrl_img).latent_dist.sample()
            ctrl_x0 = (
                ctrl_x0 - pipe.vae.config.shift_factor
            ) * pipe.vae.config.scaling_factor
        else:
            ctrl_x0 = ctrl.to(device=device, dtype=dtype)
            if ctrl_x0.shape[-2:] != (H_lat, W_lat):
                ctrl_x0 = F.interpolate(
                    ctrl_x0, size=(H_lat, W_lat), mode="bilinear", align_corners=False
                )
        ctrl_latents = FluxControlNetPipeline._pack_latents(
            ctrl_x0, B, ctrl_x0.shape[1], H_lat, W_lat
        ).to(device, dtype)

    # ----------------- Scheduler / ids -----------------
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    sigmas = noise_scheduler.sigmas
    image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
        batch_size=B, height=H_lat // 2, width=W_lat // 2, device=device, dtype=dtype
    )

    # ----------------- Inpainting prep & mask schedule -----------------
    if mask is not None:
        if mask_amplify != 1.0:
            mask = mask.float() * float(mask_amplify)
        if rgb_guide_image is None:
            raise ValueError("Mask provided without original image.")
        # ensure (B,1,H,W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != B:
            mask = mask.expand(B, *mask.shape[1:])
        if mask.shape[1] != 1:
            raise ValueError("Mask must have a single channel.")
        mask = mask.float().clamp(0, 1)

        # downsample *softly* to latent res (area) to avoid aliasing
        if mask.shape[-2:] != (H_lat, W_lat):
            mask_lat = F.interpolate(mask, size=(H_lat, W_lat), mode="area")
        else:
            mask_lat = mask

        # dilate & feather the mask
        if mask_dilation > 0.0:
            _, mask_lat, _ = dilate_then_feather_mask(
                mask_lat, dilate_rel=mask_dilation, blur_rel=0.0
            )

    # ----------------- Init latents -----------------
    if rgb_guide_image is not None:
        if rgb_guide_image.ndim == 3:
            rgb_guide_image = rgb_guide_image.unsqueeze(0)
        if rgb_guide_image.shape[0] != B:
            assert (
                rgb_guide_image.shape[0] == 1
            ), "Batch mismatch between guide image and ctrl."
            rgb_guide_image = rgb_guide_image.expand(B, *rgb_guide_image.shape[1:])
        guide = rgb_guide_image
        if guide.min() >= 0.0 and guide.max() <= 1.0:
            guide = guide * 2.0 - 1.0
        guide = guide.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        x0 = pipe.vae.encode(guide).latent_dist.sample()
        x0 = (x0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        x0 = x0.to(device=device, dtype=dtype)

        if assumed_noise_level is not None and 0.0 <= assumed_noise_level < 1.0:
            # remap the given noise level to account for the assumed noise level of the passed image if given
            sigma0 = sigmas[get_timestep_idx(noise_scheduler, noise_frac).item()].to(
                device=device, dtype=dtype
            )
            diffusion_frac = (
                noise_frac * (1.0 - assumed_noise_level) + assumed_noise_level
            )
            idx_start = get_timestep_idx(noise_scheduler, diffusion_frac).item()
        else:
            # passed image is clean
            idx_start = get_timestep_idx(noise_scheduler, noise_frac).item()
            sigma0 = sigmas[idx_start].to(device=device, dtype=dtype)
        eps = torch.randn(
            (B, C_lat, H_lat, W_lat), device=device, dtype=dtype, generator=gen
        )

        if sigma0 < 1.0:
            latents_BhwC = (1 - sigma0) * x0 + sigma0 * eps
            loop_range = range(idx_start, len(timesteps))
        else:
            latents_BhwC = eps
            loop_range = range(0, len(timesteps))
    else:
        latents_BhwC = torch.randn(
            (B, H_lat, W_lat, C_lat), device=device, dtype=dtype, generator=gen
        )
        loop_range = range(0, len(timesteps))

    latents = FluxControlNetPipeline._pack_latents(latents_BhwC, B, C_lat, H_lat, W_lat)

    # ----------------- Denoising loop -----------------
    g_vec = torch.full((B,), text_guidance_scale, device=device, dtype=dtype)
    for i in loop_range:
        t_sched = timesteps[i]
        t_model = t_sched.to(device=device, dtype=dtype).unsqueeze(0) / 1000

        # ControlNet
        if ctrl_latents is None:
            ctrl_samples, ctrl_singles = None, None
        else:
            ctrl_samples, ctrl_singles = controlnet(
                hidden_states=latents,
                timestep=t_model,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                txt_ids=text_ids,
                img_ids=image_ids,
                guidance=g_vec,
                return_dict=False,
                controlnet_cond=ctrl_latents,
            )
            if ctrl_samples:
                ctrl_samples = [s * ctrl_guidance_scale for s in ctrl_samples]
            if ctrl_singles:
                ctrl_singles = [s * ctrl_guidance_scale for s in ctrl_singles]

        # Transformer
        (v_pred,) = transformer(
            hidden_states=latents,
            timestep=t_model,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            txt_ids=text_ids,
            img_ids=image_ids,
            guidance=g_vec,
            return_dict=False,
            controlnet_block_samples=(
                [s.to(dtype=dtype) for s in ctrl_samples] if ctrl_samples else None
            ),
            controlnet_single_block_samples=(
                [s.to(dtype=dtype) for s in ctrl_singles] if ctrl_singles else None
            ),
        )

        # Dynamic quantile normalization
        # Disabled for inpainting (doesn't make sense to rescale editable region only and causes artifacts)
        if dynamic_q is not None and mask is None:
            sigma = sigmas[i].to(device=device, dtype=dtype)
            x0_pred = latents - sigma * v_pred
            flat = x0_pred.to(torch.float32).abs().view(B, -1)
            s = torch.quantile(flat, dynamic_q, dim=1, keepdim=True)
            s = torch.maximum(s, torch.ones_like(s))
            s_b = s.to(dtype).view([B] + [1] * (x0_pred.ndim - 1))
            x0_pred = x0_pred / s_b
            v_pred = (1.0 - dynamic_q_alpha) * v_pred + dynamic_q_alpha * (
                (latents - x0_pred) / (sigma + 1e-8)
            )

        # Scheduler update
        latents = noise_scheduler.step(v_pred, t_sched, latents).prev_sample

        # ---- per-step inpainting with scheduled soft mask ----
        if mask is not None:
            sigma = sigmas[i].to(device=device, dtype=dtype)  # scalar
            # dilate / feather the mask according to schedule
            if mask_scheduling:
                min_blur_rel = 0.01
                min_dilate_rel = 0.01
                mask_dilate_rel = 0.4
                mask_blur_rel = 0.4
                mask_offset = 0.0
                blur_rel = max(min_blur_rel, mask_blur_rel * sigma.item())
                dilate_rel = max(min_dilate_rel, mask_dilate_rel * sigma.item())
                _, M_t, _ = dilate_then_feather_mask(
                    mask, dilate_rel=dilate_rel, blur_rel=blur_rel
                )
                M_t = (M_t + mask_offset).clamp(0, 1)

            # unpack → blend in BHWC for your packing utilities
            latents_unpacked = FluxControlNetPipeline._unpack_latents(
                latents, height, width, vae_spacial_scale_factor
            )
            # noisy original latents at current sigma
            z_orig = (1 - sigma) * x0 + sigma * eps  # (B,C,Hl,Wl)
            M_t = mask_lat.expand(-1, z_orig.shape[1], -1, -1).to(dtype)
            # blend (inside mask = editable, outside = original)
            latents_unpacked = M_t * latents_unpacked + (1.0 - M_t) * z_orig
            # repack
            latents = FluxControlNetPipeline._pack_latents(
                latents_unpacked, B, C_lat, H_lat, W_lat
            )

    # ----------------- Decode -----------------
    latents = FluxControlNetPipeline._unpack_latents(
        latents, height, width, vae_spacial_scale_factor
    )
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    images = pipe.vae.decode(latents).sample  # [-1,1]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images
