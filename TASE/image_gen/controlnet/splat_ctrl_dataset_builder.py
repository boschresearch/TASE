# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import os

# Has to be run from the AnySplat repo root!
from src.model.model.anysplat import AnySplat
from src.model.types import Gaussians

import numpy as np
from pyquaternion import Quaternion
from torch.nn import functional as F
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import json, yaml
from tqdm import tqdm
import cv2
import argparse

from TASE.splatting.util.data_util import resize_intrinsics
from TASE.image_gen.dimensionality_reduction.embedding_model import EmbeddingModel
from TASE.splatting.common.rendering import Scene
from TASE.image_gen.common.util import ImageNetDataset


# -------------------- CLI helpers --------------------
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean")


# -------------------- Config --------------------
@dataclass
class Config:
    # user-set parameters
    output_dir: str = "$TASE_ROOT/DATA/SplatCtrl"
    imagenet_root: str = "$IMGNET_ROOT"
    dataset_split: str = "train"
    output_res: int = 512
    feature_hack: bool = False

    # model / encoder settings
    max_feature_dim: int = 64
    downsample_factor: int = 8
    embedding_model_path: str = None
    raw_dino: bool = False
    anysplat_model_path: str = "lhjiang/anysplat"
    inpainting_iters: int = 100
    feature_encoder_ps: int = 16
    anysplat_ps: int = 14

    # perturbation probabilities
    back_and_forth_prob: float = 0.0
    global_perturb_prob: float = 0.5
    local_perturb_prob: float = 0.5

    # perturbation configs
    camera_perturb: Dict[str, float] = field(
        default_factory=lambda: {"rot_range": 0.02, "trans_range": 0.05}
    )
    global_perturb: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "means": {"std": 0.01},
            "scales": {"std": 1.0, "offset": 0.4},
            "opacities": {"std": 2.0},
            "quats": {"std": 2.0},
        }
    )
    local_perturb: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "means": {"std": 0.02},
            "scales": {"std": 2.0, "offset": 0.8},
            "opacities": {"std": 4.0},
            "quats": {"std": 4.0},
        }
    )
    local_perturb_max_fraction: float = 0.5

    # generation parameters
    max_samples: int = 50_000
    batch_size: int = 8

    # runtime flags
    debug: bool = False

    # derived fields
    meta_path: str = field(init=False)
    anysplat_res: int = field(init=False)
    feature_res: int = field(init=False)
    dataset_res: int = field(init=False)

    # -------- CLI factory --------
    @staticmethod
    def from_cli() -> "Config":
        p = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        p.add_argument(
            "--mrl-model-path",
            type=str,
            default=None,
            help="Path to the MRL embedding model checkpoint (not needed with --raw-dino)",
        )
        p.add_argument(
            "--raw-dino",
            action="store_true",
            default=False,
            help="Use raw DINOv3 backbone features (768 dims) without adapter",
        )
        p.add_argument(
            "--output-dir",
            type=str,
            default=Config.output_dir,
            help="Output base directory",
        )
        p.add_argument(
            "--num-samples",
            type=int,
            default=Config.max_samples,
            help="Number of samples to generate",
        )
        p.add_argument(
            "--imagenet-root",
            type=str,
            default=Config.imagenet_root,
            help="ImageNet root",
        )
        p.add_argument(
            "--output-res",
            type=int,
            default=Config.output_res,
            help="Final output resolution",
        )
        p.add_argument(
            "--feature-hack",
            type=str2bool,
            default=Config.feature_hack,
            help="Enable feature edge-crop hack",
        )
        p.add_argument(
            "--max-feature-dim",
            type=int,
            default=Config.max_feature_dim,
            help="Max feature dim, None=all",
        ),
        p.add_argument(
            "--p-back-forth",
            type=float,
            default=Config.back_and_forth_prob,
            help="Back-and-forth perturb prob",
        )
        p.add_argument(
            "--p-global",
            type=float,
            default=Config.global_perturb_prob,
            help="Global perturb prob",
        )
        p.add_argument(
            "--p-local",
            type=float,
            default=Config.local_perturb_prob,
            help="Local perturb prob",
        )
        p.add_argument(
            "--dataset-split",
            type=str,
            default="train",
            help="ImageNet dataset split to use (train/val)",
        )
        p.add_argument(
            "--debug", type=str2bool, default=Config.debug, help="Debug mode"
        )
        args = p.parse_args()

        if not args.raw_dino and args.mrl_model_path is None:
            p.error("--mrl-model-path is required unless --raw-dino is specified")

        # Build instance overriding selected fields
        cfg = Config(
            output_dir=args.output_dir,
            max_samples=args.num_samples,
            imagenet_root=args.imagenet_root,
            output_res=args.output_res,
            feature_hack=args.feature_hack,
            back_and_forth_prob=args.p_back_forth,
            global_perturb_prob=args.p_global,
            local_perturb_prob=args.p_local,
            debug=args.debug,
            dataset_split=args.dataset_split,
            max_feature_dim=768 if args.raw_dino else args.max_feature_dim,
        )
        cfg.raw_dino = args.raw_dino
        if not args.raw_dino:
            cfg.embedding_model_path = args.mrl_model_path
        return cfg

    def dump(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=4)

    def __post_init__(self):
        # expand env vars
        self.output_dir = os.path.join(
            os.path.expandvars(self.output_dir), self.dataset_split
        )
        self.imagenet_root = os.path.expandvars(self.imagenet_root)

        # timestamp or debug suffix
        if self.debug:
            self.output_dir = os.path.join(self.output_dir, "debug")
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.output_dir, timestamp)

        self.meta_path = os.path.join(self.output_dir, "meta.yml")

        # compute resolutions
        self.anysplat_res = (
            self.output_res
            if self.output_res % self.anysplat_ps == 0
            else (self.output_res // self.anysplat_ps + 1) * self.anysplat_ps
        )
        self.feature_res = (
            self.output_res
            if self.output_res % self.feature_encoder_ps == 0
            else (self.output_res // self.feature_encoder_ps + 1)
            * self.feature_encoder_ps
        )
        if self.feature_hack:
            self.feature_res += 2 * self.feature_encoder_ps
        self.dataset_res = max(self.anysplat_res, self.feature_res)

        print(f"Output directory: {self.output_dir}")
        print(
            f"Using output resolution {self.output_res}, AnySplat resolution {self.anysplat_res}, "
            f"feature resolution {self.feature_res}, dataset resolution {self.dataset_res}"
        )

        self.dump()


@torch.no_grad()
def get_gaussians(
    cfg: Config,
    img_batch: torch.Tensor,
    splat_model: AnySplat,
    *,
    encoder_model: EmbeddingModel | None = None,
    feature_batch: torch.Tensor | None = None,
):
    """
    Get the gaussians and camera poses from the input image batch.
    img_batch: [B, 3, H, W] in [-1, 1]
    splat_model: AnySplat model
    encoder_model: EmbeddingModel to extract features
    feature_batch: [B, C_f, H_f, W_f] in arbitrary range, are prefered over using encoder_model
    Returns: scenes: List[Scene]
             cams: List[CameraState]
             (w, h): image size
    """
    # checks and reshaping
    B, C, H, W = img_batch.shape
    assert H == W, "Only square images are supported."

    # handle feature extraction
    if feature_batch is None and encoder_model is not None:
        # Resize image to feature resolution
        if H != cfg.feature_res or W != cfg.feature_res:
            img_batch = F.interpolate(
                img_batch,
                size=(cfg.feature_res, cfg.feature_res),
                mode="bilinear",
                align_corners=False,
            )
            H, W = cfg.feature_res, cfg.feature_res
        # Extract features using the encoder model
        with torch.no_grad():
            feature_batch = encoder_model.encode(img_batch)
        # Apply feature hacks if needed
        feature_batch = (
            feature_batch[..., : cfg.max_feature_dim]
            if cfg.max_feature_dim is not None
            else feature_batch
        )
        if cfg.feature_hack:
            # crop out the outer patches of the feature map to remove border artifacts
            H, W = feature_batch.shape[1], feature_batch.shape[2]
            crop_h = 1
            crop_w = 1
            feature_batch = feature_batch[
                :, crop_h : H - crop_h, crop_w : W - crop_w, :
            ]
            # apply the equivalent crop to the input image as well
            ps = encoder_model.patch_size
            img_h, img_w = img_batch.shape[2], img_batch.shape[3]
            crop_img_w = int(crop_w * ps)
            crop_img_h = int(crop_h * ps)
            img_batch = img_batch[
                :, :, crop_img_h : img_h - crop_img_h, crop_img_w : img_w - crop_img_w
            ]

    if feature_batch is not None:
        B_f, H_f, W_f, C_f = feature_batch.shape
        assert B_f == B, "Batch size of feature batch must match image batch."
        assert H_f == W_f, "Only square feature maps are supported."
        # Interpolate to anysplat resolution if needed
        if H_f != cfg.anysplat_res or W_f != cfg.anysplat_res:
            feature_batch = feature_batch.permute(0, 3, 1, 2)  # [B, C_f, H_f, W_f]
            feature_batch = F.interpolate(
                feature_batch,
                size=(cfg.anysplat_res, cfg.anysplat_res),
                mode="bilinear",
                align_corners=False,
            )
            feature_batch = feature_batch.permute(0, 2, 3, 1)  # [B, H_f, W_f, C_f]

    if H != cfg.anysplat_res or W != cfg.anysplat_res:
        img_batch_as = F.interpolate(
            img_batch,
            size=(cfg.anysplat_res, cfg.anysplat_res),
            mode="bilinear",
            align_corners=False,
        )
    else:
        img_batch_as = img_batch

    img_batch_as = img_batch_as.unsqueeze(1)  # [B, 1, 3, H, W]
    img_batch_as = (img_batch_as + 1) * 0.5  # to [0, 1]
    img_batch_as = img_batch_as.clamp(0, 1)

    # Disable the voxelization so that I can project the features to the gaussians directly
    if splat_model.encoder_cfg.voxelize:
        print("Warning: Disabling voxelization in the AnySplat model.")
        splat_model.encoder_cfg.voxelize = False

    # Inference splatting model to get gaussians
    with torch.no_grad():
        gaussians, pred_context_pose = splat_model.inference(img_batch_as)

    scenes = gaussians_to_scene(gaussians)
    cams = split_cams_along_batch(pred_context_pose)

    # Inject the features to the gaussians
    if feature_batch is not None:
        for b in range(B):
            features = feature_batch[b].reshape(-1, C_f)  # [H*W, C_f]
            # Check the shapes
            N = scenes[b].splats["means"].shape[0]
            N_f = features.shape[0]
            assert (
                N == N_f
            ), f"Number of gaussians {N} must match number of features {N_f}. Feature map shape: {feature_batch.shape}"
            scenes[b].splats["features"] = features.float()  # [N, C_f]

    return scenes, cams, img_batch


def gaussians_to_scene(gaussians: Gaussians, max_sh_degree=3):
    """
    Save the gaussians to a scene checkpoint file.
    """
    B = gaussians.means.shape[0]
    scenes = []

    for b in range(B):
        # Change the scales to log scale and opacities to logits
        opacities = gaussians.opacities[b].detach().clone().float()
        opacities = torch.logit(opacities.clamp(1e-10, 1 - 1e-10))
        scales = gaussians.scales[b].detach().clone().float()
        scales = torch.log(scales)

        splats = {
            "means": gaussians.means[b].detach().clone().float(),
            "scales": scales,
            "opacities": gaussians.opacities[b].detach().clone().float(),
            "quats": gaussians.rotations[b].detach().clone().float(),
            "sh0": gaussians.harmonics[b, ..., 0:1]
            .detach()
            .clone()
            .permute(0, 2, 1)
            .float()
            .contiguous(),
            "shN": gaussians.harmonics[b, ..., 1 : max_sh_degree * 4 + 1]
            .detach()
            .clone()
            .permute(0, 2, 1)
            .float()
            .contiguous(),
        }

        scene = Scene()
        scene.cfg.antialiased = False  # Antialiased mode causes issues with the splats predicted by AnySplat
        scene.splats = splats

        scenes.append(scene)

    return scenes


def split_cams_along_batch(pred_context_pose):
    """
    Split the cameras in the batch dimension to a list of CameraState.
    """
    extrinsics = pred_context_pose["extrinsic"]  # [B, 4, 4]
    intrinsics = pred_context_pose["intrinsic"]  # [B, 3,

    B = extrinsics.shape[0]
    cams = []

    for b in range(B):
        c2w = extrinsics[b, 0].unsqueeze(0)
        K = intrinsics[b, 0].unsqueeze(0)

        cam = {
            "extrinsic": c2w,
            "intrinsic": K,
        }
        cams.append(cam)

    return cams


@torch.no_grad()
def laplacian_inpaint(img, alpha, iters=10):
    """
    img:   [B,3,H,W] in [0,1]
    alpha: [B,1,H,W] 1=known, 0=hole
    Returns: inpainted image [B,3,H,W]
    """
    B, C, H, W = img.shape
    known = alpha.clamp(0, 1)
    hole = 1.0 - known

    # first set the background color to the average color of known pixels
    avg_color = (img * known).sum(dim=(2, 3), keepdim=True) / (
        known.sum(dim=(2, 3), keepdim=True) + 1e-8
    )
    img = img * known + avg_color * hole

    # ---- init unknowns by replicated-avg blur (helps convergence) ----
    # Pick an odd kernel size in [3, 15] scaling with image size.
    base = min(H, W)
    k_size = max(15, 2 * (base // 32) + 1)  # always odd, >=15
    p = (k_size - 1) // 2

    u = img.clone()
    u_pad = F.pad(u, (p, p, p, p), mode="replicate")  # (L,R,T,B)
    u = F.avg_pool2d(u_pad, kernel_size=k_size, stride=1, padding=0)
    u = known * img + hole * u

    # ---- Jacobi iterations with replicate padding (4-neighborhood) ----
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)
    k_expanded = kernel.expand(C, 1, 3, 3)  # depthwise per-channel

    for _ in range(iters):
        u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")  # correct replicate pad
        neigh_sum = F.conv2d(u_pad, k_expanded, padding=0, groups=C)
        u_new = neigh_sum * 0.25  # average of 4 neighbors
        u = known * img + hole * u_new  # Dirichlet BC

    return u.clamp(0, 1)


@torch.no_grad()
def render_fn(
    self,
    pred_context_pose,
    img_wh,
    cfg: Config,
    **kwargs,
):
    """Custom render fucntion just for the dataset creation."""

    W, H = img_wh
    c2w = pred_context_pose["extrinsic"].float().to(self.device)
    K = pred_context_pose["intrinsic"].float().to(self.device)

    # scale the normalized intrinsics to the image size
    K = K.clone()
    K[:, 0, 0] *= W
    K[:, 0, 2] *= W
    K[:, 1, 1] *= H
    K[:, 1, 2] *= H

    render_colors, render_alphas, _, info = self.rasterize_splats(
        camtoworlds=c2w,
        Ks=K,
        width=W,
        height=H,
        sh_degree=self.sh_degree,  # active all SH degrees
        render_mode="RGB",
        **kwargs,
    )  # [1, H, W, 3]

    if cfg.inpainting_iters and cfg.inpainting_iters > 0:
        # inpaint the holes with laplacian inpainting
        alpha = render_alphas.permute(0, 3, 1, 2)  # [1, 1, H, W]
        color = render_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        render_colors = laplacian_inpaint(color, alpha, cfg.inpainting_iters)
        render_colors = render_colors.permute(0, 2, 3, 1)  # [1, H, W, 3]

    img_rgb = render_colors.clamp(0, 1)

    # render the feature map
    feat_W, feat_H = W // cfg.downsample_factor, H // cfg.downsample_factor
    scale_x = feat_W / W
    scale_y = feat_H / H
    K = resize_intrinsics(
        K=K,
        scale=(scale_x, scale_y),
    )
    W, H = feat_W, feat_H

    render_colors, render_alphas, normals, info = self.rasterize_splats(
        camtoworlds=c2w,
        Ks=K,
        width=W,
        height=H,
        sh_degree=None,
        near_plane=self.cfg.near_plane,
        far_plane=self.cfg.far_plane,
        render_mode="RGB",
        color_mode="Features",
        max_feature_dim=None,
        **kwargs,
    )

    if cfg.inpainting_iters and cfg.inpainting_iters > 0:
        # inpaint the holes with laplacian inpainting
        alpha = render_alphas.permute(0, 3, 1, 2)  # [1, 1, H, W]
        color = render_colors.permute(0, 3, 1, 2)  # [1, C, H, W]
        render_colors = laplacian_inpaint(color, alpha, cfg.inpainting_iters)
        render_colors = render_colors.permute(0, 2, 3, 1)  # [1, H, W, C]

    # img_features = render_colors.clamp(-1, 1)     # Should I clamp the features?
    img_features = render_colors

    return img_rgb, img_features


def _sample_signed(max_mag: float) -> float:
    """Uniform magnitude in [0, max_mag] with ± sign."""
    if max_mag <= 0:
        return 0.0
    mag = float(np.random.uniform(0.0, max_mag))
    sgn = -1.0 if np.random.rand() < 0.5 else 1.0
    return sgn * mag


def _sample_params(max_cfg: Dict[str, float]) -> Dict[str, float]:
    """
    From a dict like {'std': 0.01, 'offset': 0.4, 'scale': 1.0} sample
    concrete values. All entries are interpreted as maximums.
    - std:  U(0, max_std)          (non-negative)
    - offset: ±U(0, max_offset)    (additive)
    - scale:  ±U(0, max_scale)     (multiplicative delta around 1.0)
    """
    out = {}
    if "std" in max_cfg:
        m = float(max_cfg["std"])
        out["std"] = 0.0 if m <= 0 else float(np.random.uniform(0.0, m))
    if "offset" in max_cfg:
        out["offset"] = _sample_signed(float(max_cfg["offset"]))
    if "scale" in max_cfg:
        out["scale"] = _sample_signed(float(max_cfg["scale"]))
    return out


@torch.no_grad()
def perturb_gaussians(
    scene: "Scene",
    global_perturb: Optional[dict] = None,
    local_perturb: Optional[dict] = None,
    local_gaussians_frac: Optional[float] = None,
    verbose: bool = False,
) -> Tuple["Scene", dict]:
    """
    Apply global + local perturbations using *sampled* parameters (<= maxima in cfg dicts).
    Returns (perturbed_scene, perturb_log) where perturb_log is JSON-serializable.
    """
    splats = scene.splats
    perturbed_splats = {}

    g_cfg = global_perturb or {}
    l_cfg = local_perturb or {}

    # ---- GLOBAL pass (all splats) ----
    global_log = {}
    for key, value in splats.items():
        v = value.detach().clone()
        if key in g_cfg:
            # sample concrete parameters for this attribute
            params = _sample_params(g_cfg[key])
            std = params.get("std", 0.0)
            offset = params.get("offset", 0.0)
            scale = params.get("scale", 0.0)  # interpreted as delta around 1.0
            noise = torch.randn_like(v) * float(std) if std > 0 else 0.0
            v = v * (1.0 + scale) + offset + noise
            global_log[key] = {"std": std, "offset": offset, "scale_delta": scale}
        perturbed_splats[key] = v

    # ---- LOCAL pass (subset of splats around random center) ----
    local_log = {"enabled": False}
    mask = None
    if (
        local_gaussians_frac is not None
        and local_gaussians_frac > 0
        and len(splats["means"]) > 0
    ):
        frac = float(np.clip(local_gaussians_frac, 0.0, 1.0))
        means = splats["means"]
        k = max(1, int(frac * means.shape[0]))

        center_idx = int(torch.randint(0, means.shape[0], (1,)).item())
        center = means[center_idx : center_idx + 1]
        dists = (means - center).norm(dim=1)  # [N]
        # kth smallest radius
        thresh, _ = torch.kthvalue(dists, k)
        mask = dists <= thresh

        if verbose:
            print(
                f"Perturbing {int(mask.sum())}/{mask.numel()} splats within local radius {float(thresh):.4f}"
            )

        # apply sampled local params only to masked entries
        for key, value in splats.items():
            if key in l_cfg:
                params = _sample_params(l_cfg[key])
                std = params.get("std", 0.0)
                offset = params.get("offset", 0.0)
                scale = params.get("scale", 0.0)
                noise = torch.randn_like(value) * float(std) if std > 0 else 0.0

                v = perturbed_splats[key]  # already has global effects
                v_masked = (
                    value[mask] * (1.0 + scale)
                    + offset
                    + (noise[mask] if isinstance(noise, torch.Tensor) else 0.0)
                )
                v = v.clone()
                v[mask] = v_masked
                perturbed_splats[key] = v

                # bookkeep per-attribute local params (same scalar per call)
                local_log.setdefault("attrs", {})[key] = {
                    "std": std,
                    "offset": offset,
                    "scale_delta": scale,
                }

        local_log.update(
            {
                "enabled": True,
                "center_index": center_idx,
                "radius": float(thresh),
                "selected_count": int(mask.sum().item()),
                "total": int(mask.numel()),
                "fraction_requested": frac,
                "fraction_effective": float(mask.float().mean().item()),
            }
        )

    # build new scene
    perturbed_scene = Scene()
    perturbed_scene.splats = perturbed_splats
    perturbed_scene.cfg = scene.cfg

    # final log
    perturb_log = {
        "global": global_log,  # per-attribute concrete params used
        "local": local_log,  # info incl. center, radius, counts
    }
    return perturbed_scene, perturb_log


def perturb_camera_extrinsics(rot_range=0.02, trans_range=0.05):
    """
    Same math as before; now also return a small log.
    rot_range (rad), trans_range (abs units) are maxima; actual angle is U(-rot_range, +rot_range).
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = float(np.random.uniform(-rot_range, rot_range))
    q_rot = Quaternion(axis=axis, angle=angle)
    R_perturb = torch.from_numpy(q_rot.rotation_matrix).float()

    t = np.random.uniform(-trans_range, trans_range, size=3).astype(np.float32)
    t_perturb = torch.from_numpy(t)

    T_perturb = torch.eye(4)
    T_perturb[:3, :3] = R_perturb
    T_perturb[:3, 3] = t_perturb

    cam_log = {"axis": axis.tolist(), "angle_rad": angle, "translation": t.tolist()}
    return T_perturb, cam_log


@torch.no_grad()
def generate_sample_pairs(
    cfg: Config,
    img_batch,
    splat_model: AnySplat,
    encoder_model: EmbeddingModel,
    back_and_forth=False,
    local_perturb=True,
    global_perturb=True,
    device=torch.device("cuda"),
):
    """
    Generate a sample pair of (input image, novel view image) for training.
    img_batch: [B, 3, H, W] in [-1, 1]
    Returns:
        input_img: [B, 3, H, W] in [-1, 1]
        novel_img: [B, 3, H, W] in [-1, 1]
        feature_map: [B, C, H', W'] in [-1, 1]
    """
    B = img_batch.shape[0]
    scenes, cams, img_batch = get_gaussians(
        cfg, img_batch, splat_model, encoder_model=encoder_model
    )
    per_sample_logs = [{"camera": {}, "gaussians": {}} for _ in range(B)]

    if back_and_forth:
        # First pertubation option: perturb camera, reconstruct with render, then re-render from original camera
        renders = []
        features_original = []
        cam_perturbs = []
        for i, (scene, cam) in enumerate(zip(scenes, cams)):
            # perturb camera
            cam_perturb, per_sample_logs[i]["camera"] = perturb_camera_extrinsics().to(
                device
            )
            cam_perturbs.append(cam_perturb)
            cam_ext_og = cam["extrinsic"].squeeze(0)
            cam_perturb = cam_perturb @ cam_ext_og
            cam_perturb = {
                "extrinsic": cam_perturb.unsqueeze(0),
                "intrinsic": cam["intrinsic"],
            }
            # render from perturbed camera
            render_original, feature_original = render_fn(
                scene,
                cam_perturb,
                img_wh=(cfg.anysplat_res, cfg.anysplat_res),
                cfg=cfg,
                radius_clip=0.1,
            )
            renders.append(render_original)
            features_original.append(feature_original)
        renders = torch.cat(renders, dim=0)  # [B, H, W, 3]
        renders = renders.permute(0, 3, 1, 2).clamp(0, 1)
        img_batch = renders * 2 - 1  # to [-1, 1]
        feature_batch = torch.cat(features_original, dim=0)  # [B, H', W', C]

        # reconstruct new scenes from the img_batch or the rendered images (if back_and_forth)
        scenes, cams, img_batch = get_gaussians(
            cfg,
            img_batch,
            splat_model,
            encoder_model=encoder_model,
            feature_batch=feature_batch,
        )
    else:
        cam_perturbs = None

    # perturb the gaussians
    if local_perturb:
        local_perturb_cfg = cfg.local_perturb
        local_gaussians_frac = np.random.uniform(0.0, cfg.local_perturb_max_fraction)
    else:
        local_perturb_cfg = None
        local_gaussians_frac = 0.0
    if global_perturb:
        global_perturb_cfg = cfg.global_perturb
    else:
        global_perturb_cfg = None
    if local_perturb or global_perturb:
        for i in range(len(scenes)):
            scenes[i], per_sample_logs[i]["gaussians"] = perturb_gaussians(
                scenes[i],
                global_perturb=global_perturb_cfg,
                local_perturb=local_perturb_cfg,
                local_gaussians_frac=local_gaussians_frac,
            )

    # render novel views from original camera poses
    renders_novel = []
    features_novel = []
    for i, (scene, cam) in enumerate(zip(scenes, cams)):
        if cam_perturbs is not None:
            cam_perturb = cam_perturbs[i]
            # retrieve original camera by inverting the perturbation
            cam_perturbed = cam["extrinsic"].squeeze(0)
            cam_extr = torch.linalg.inv(cam_perturb) @ cam_perturbed
            cam = {"extrinsic": cam_extr.unsqueeze(0), "intrinsic": cam["intrinsic"]}
        # render from original camera
        render_novel, feature_novel = render_fn(
            scene,
            cam,
            img_wh=(cfg.output_res, cfg.output_res),
            cfg=cfg,
            radius_clip=0.1,
        )
        renders_novel.append(render_novel)
        features_novel.append(feature_novel)

    renders_novel = torch.cat(renders_novel, dim=0)  # [B, H, W, 3]
    renders_novel = (renders_novel.permute(0, 3, 1, 2)).clamp(
        0, 1
    )  # [B, 3, H, W] in [0, 1]
    renders_novel = renders_novel * 2 - 1  # to [-1, 1]
    features_novel = torch.cat(features_novel, dim=0)  # [B, H', W', C]

    # Interpolate the original image to the output resolution if needed
    if img_batch.shape[2] != cfg.output_res or img_batch.shape[3] != cfg.output_res:
        img_batch = F.interpolate(
            img_batch,
            size=(cfg.output_res, cfg.output_res),
            mode="bilinear",
            align_corners=False,
        )

    return img_batch, features_novel, renders_novel, per_sample_logs


def main():
    cfg = Config.from_cli()

    # Load AnySplat from Hugging Face
    print("Loading AnySplat model...")
    splat_model = AnySplat.from_pretrained(cfg.anysplat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. Using CPU instead.")
    splat_model = splat_model.to(device).eval()
    for p in splat_model.parameters():
        p.requires_grad = False

    # Disable voxelization if you want to project features directly
    splat_model.encoder.cfg.voxelize = False

    # Load the encoder model
    print("Loading feature encoder model...")
    if getattr(cfg, "raw_dino", False):
        feature_encoder = EmbeddingModel.from_backbone("dinov3", device=device, dtype=torch.bfloat16)
    else:
        feature_encoder = (
            EmbeddingModel.load_ckpt(cfg.embedding_model_path, device).to(device).eval()
        )
    for p in feature_encoder.parameters():
        p.requires_grad = False

    # Dataset / loader
    print("Loading dataset...")
    train_ds = ImageNetDataset(
        cfg.imagenet_root,
        cfg.dataset_res,
        return_imagenet_filename=True,
        normalization="vae",
        split=cfg.dataset_split,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    for subdir in ["input", "novel", "feature"]:
        os.makedirs(os.path.join(cfg.output_dir, subdir), exist_ok=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    # dump the classmap
    with open(os.path.join(cfg.output_dir, "classmap.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=4)

    pbar, step = tqdm(total=cfg.max_samples, desc="Generating", disable=None), 0
    from datetime import datetime

    timestamp = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
    with open(cfg.meta_path, "w") as f:
        print(f"# Dataset metadata generated on {timestamp}", file=f)

    for batch in train_loader:
        if step >= cfg.max_samples:
            break
        img, label, path = batch
        img = img.to(device)  # [B,3,H,W] in [-1,1]
        B = img.shape[0]

        # perturbation draws
        back_and_forth = np.random.rand() < cfg.back_and_forth_prob
        local_perturb = np.random.rand() < cfg.local_perturb_prob
        global_perturb = np.random.rand() < cfg.global_perturb_prob

        target_img, feature_map, novel_img, per_sample_logs = generate_sample_pairs(
            cfg,
            img,
            splat_model,
            feature_encoder,
            back_and_forth=back_and_forth,
            local_perturb=local_perturb,
            global_perturb=global_perturb,
            device=device,
        )

        # save batch
        sample_meta = {}
        for b in range(B):
            in_img_path = f"input/{step+b:08d}.jpg"
            out_img_path = f"novel/{step+b:08d}.jpg"
            feat_path = f"feature/{step+b:08d}.npz"

            # images: convert [-1,1] or [0,1] to uint8 safely
            def to_u8(t):
                x = t.detach().float().cpu()
                x = (x + 1) * 0.5
                x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
                return x

            cv2.imwrite(
                os.path.join(cfg.output_dir, in_img_path),
                to_u8(target_img[b]).permute(1, 2, 0).numpy()[:, :, ::-1],
            )
            cv2.imwrite(
                os.path.join(cfg.output_dir, out_img_path),
                to_u8(novel_img[b]).permute(1, 2, 0).numpy()[:, :, ::-1],
            )

            np.savez(
                os.path.join(cfg.output_dir, feat_path),
                feature_map=feature_map[b].cpu().numpy(),
            )

            sample_meta[f"sample_{step+b:08d}"] = {
                "imgnet_filename": str(path[b]),
                "label": int(label[b].item()),
                "back_and_forth": bool(back_and_forth),
                "local_perturb": bool(local_perturb),
                "global_perturb": bool(global_perturb),
                "input_img_path": in_img_path,
                "novel_img_path": out_img_path,
                "feature_map_path": feat_path,
                "logs": per_sample_logs[b],
            }

        with open(cfg.meta_path, "a") as f:
            yaml.dump(sample_meta, f)

        step += B
        pbar.update(B)
        if cfg.debug and step >= 8 * cfg.batch_size:
            break


if __name__ == "__main__":
    main()


"""
##### ATTENTION #####
Place in the AnySplat dir and launch from there!

python splat_ctrl_dataset_builder.py \
    --mrl-model-path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/MRL/dinov3/Ablations/AE_EQV/ckpt/epoch_1.pth \
    --num-samples 4096 \
    --dataset-split val 

python splat_ctrl_dataset_builder.py \
    --mrl-model-path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/MRL/dinov3/Ablations/AE_EQV/ckpt/epoch_1.pth \
    --num-samples 50000 \
    --dataset-split train 
"""
