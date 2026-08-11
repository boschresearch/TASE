# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Evaluation pipeline for 3DGS editing baselines and ablations.

Renders edited scenes from multiple baselines (GaussianEditor, DGE,
DreamCatalyst, EditSplat, Ours/TASE) using common camera poses per scene,
then computes CLIP directional similarity and CLIP score metrics.

Usage:
    python -m TASE.splatting.util.evaluation --mode baselines
    python -m TASE.splatting.util.evaluation --mode ablations
    python -m TASE.splatting.util.evaluation --mode both
"""

import argparse
import logging
from typing import List, Dict, Optional, Tuple

from TASE.splatting.common.rendering import Scene
from TASE.splatting.trainer.trainer import Runner

import numpy as np
import torch
import shutil
import os
import json
import cv2
from glob import glob
from PIL import Image
from torchvision import transforms
from pathlib import Path
from pyquaternion import Quaternion

import torch.nn.functional as F

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants — all relative to WORKSPACE_ROOT
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path("/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace")

# Pose files per scene (in COLMAP frame shared by GE/DGE/ES/TASE)
POSE_DIR = WORKSPACE_ROOT / "Repos/TASE/data/poses"

# Prompts for all scenes/experiments
PROMPTS_JSON = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/ablations/prompts.json"

# Default output directory for baseline comparison renders + metrics
BASELINE_OUT_DIR = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/baseline_comparison"

# Default output directory for ablation renders + metrics
ABLATION_OUT_DIR = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/ablations"

# Scene name mapping: external name → TASE name
SCENE_NAME_MAP = {"bicycle": "bike"}
SCENE_NAME_MAP_INV = {"bike": "bicycle"}

# Images picked for paper figures per scene
PAPER_IMAGES = {
    "bear": ["rgb_013.jpg", "rgb_022.jpg"],
    "bike": ["rgb_001.jpg", "rgb_023.jpg"],
    "face": ["rgb_009.jpg", "rgb_012.jpg"],
}

PAPER_IMAGES_PCA = {
    "bear": ["rgb_001.jpg", "rgb_010.jpg"],
    "bike": ["rgb_001.jpg", "rgb_023.jpg"],
    "face": ["rgb_009.jpg", "rgb_012.jpg"],
}

# Image stems to exclude from metric computation, per dataset.
# Empty by default — populate to drop views that don't sufficiently show the object.
EXCLUDE_IMAGES: Dict[str, List[str]] = {
    "bear": [],
    #"bike": ["rgb_006", "rgb_007", "rgb_008", "rgb_009", "rgb_010", "rgb_011"],
    "bike": [],
    "face": [],
}

IMG_WH = (768, 448)

# In backwards-compat mode (render_depth=True), Ours/GE/DGE share one
# scene-level original_scene/ render (they all use the same COLMAP
# reconstruction).  DreamCatalyst and EditSplat each use their own
# per-method original_scene/ because they were trained on different base
# models or in different coordinate frames.
_BACKWARDS_COMPAT_SHARED_ORIG_METHODS = {"Ours", "GaussianEditor", "DGE"}

# (No per-scene DC calibration data needed: DCColmapDataParser reads COLMAP
# binary files directly, and dataparser_transforms.json encodes the
# COLMAP→normalised transform.  The nerfstudio OpenCV→OpenGL flip and the
# OpenGL→OpenCV flip required for TASE's rasteriser cancel, so only T_fwd
# is applied.  See _colmap_to_dc_normalized.)

# ---------------------------------------------------------------------------
# Baseline checkpoint discovery
# ---------------------------------------------------------------------------

def _find_ge_ckpts(scene: str) -> Dict[str, Path]:
    """Find GaussianEditor edit checkpoints for a scene."""
    base = WORKSPACE_ROOT / "Repos/GaussianEditor/outputs/edit-n2n/long" / scene
    return _find_ply_edits(base, "save/last.ply")


def _find_dge_ckpts(scene: str) -> Dict[str, Path]:
    """Find DGE edit checkpoints for a scene."""
    base = WORKSPACE_ROOT / "Repos/DGE/outputs/dge/long" / scene
    return _find_ply_edits(base, "save/last.ply")


def _find_ply_edits(base_dir: Path, ckpt_relative: str) -> Dict[str, Path]:
    """Discover experiment→ckpt_path for a baseline using PLY checkpoints."""
    result = {}
    if not base_dir.is_dir():
        return result
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        ckpt = exp_dir / ckpt_relative
        if ckpt.exists():
            result[exp_dir.name] = ckpt
    return result


def _find_es_ckpts(scene_ext: str) -> Dict[str, Path]:
    """Find EditSplat edit checkpoints for a scene.

    EditSplat stores edited PLY in point_cloud/iteration_*/point_cloud.ply.
    If that doesn't exist (failed edits), skip the experiment.
    """
    base = WORKSPACE_ROOT / "Results/EditSplat/edits" / scene_ext
    result = {}
    if not base.is_dir():
        return result
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        pc_dirs = sorted(glob(str(exp_dir / "point_cloud" / "iteration_*" / "point_cloud.ply")))
        if pc_dirs:
            result[exp_dir.name] = Path(pc_dirs[-1])  # latest iteration
    return result


def _find_dc_ckpts(scene_ext: str) -> Dict[str, Tuple[Path, Path]]:
    """Find DreamCatalyst edit checkpoints for a scene.

    Returns dict of experiment → (ckpt_path, dataparser_transforms_path).
    DC uses nerfstudio format; each edit may have multiple runs, pick latest.
    """
    base = WORKSPACE_ROOT / "Results/DreamCatalyst/edits" / scene_ext
    result = {}
    if not base.is_dir():
        return result
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        dc_dirs = sorted(glob(str(exp_dir / "dc_splat" / "*")))
        if not dc_dirs:
            continue
        # Pick the latest run directory
        latest_run = Path(dc_dirs[-1])
        ckpt_candidates = sorted(glob(str(latest_run / "nerfstudio_models" / "step-*.ckpt")))
        dt_path = latest_run / "dataparser_transforms.json"
        if ckpt_candidates and dt_path.exists():
            result[exp_dir.name] = (Path(ckpt_candidates[-1]), dt_path)
    return result


def _find_ge_base(scene: str) -> Optional[Path]:
    """Original unedited model for GE/DGE (shared COLMAP reconstruction).

    GE dataset uses 'bicycle' (not 'bike') for the base model directory.
    """
    ge_scene = SCENE_NAME_MAP_INV.get(scene, scene)
    p = WORKSPACE_ROOT / "Repos/GaussianEditor/dataset" / ge_scene / "point_cloud/iteration_30000/point_cloud.ply"
    return p if p.exists() else None


def _find_es_base(scene_ext: str) -> Optional[Path]:
    """Original unedited model for EditSplat."""
    p = WORKSPACE_ROOT / "Results/EditSplat/base_models" / scene_ext / "point_cloud/iteration_30000/point_cloud.ply"
    return p if p.exists() else None


def _find_dc_base(scene_ext: str) -> Optional[Tuple[Path, Path]]:
    """Original unedited model for DreamCatalyst. Returns (ckpt, dataparser_transforms)."""
    base_dir = WORKSPACE_ROOT / "Results/DreamCatalyst/base_models" / scene_ext / "splatfacto"
    if not base_dir.is_dir():
        return None
    runs = sorted(base_dir.iterdir())
    if not runs:
        return None
    latest = runs[-1]
    ckpts = sorted(glob(str(latest / "nerfstudio_models" / "step-*.ckpt")))
    dt = latest / "dataparser_transforms.json"
    if ckpts and dt.exists():
        return (Path(ckpts[-1]), dt)
    return None


def _load_dataparser_trafo(dt_path: Path) -> Tuple[np.ndarray, float]:
    """Load nerfstudio dataparser_transforms.json → (T_fwd_4x4, scale).

    T_fwd maps world (COLMAP) coordinates → nerfstudio normalised frame.
    Scale is applied after the affine: ``p_norm = scale * (T_fwd @ p_world)``.
    """
    with open(dt_path) as f:
        data = json.load(f)
    transform = np.array(data["transform"])  # 3x4
    # Expand to 4x4
    T = np.eye(4)
    T[:3, :] = transform
    scale = float(data["scale"])
    return T, scale


# ---------------------------------------------------------------------------
# DreamCatalyst: COLMAP (OpenCV) → DC normalised pose transform
# ---------------------------------------------------------------------------

def _colmap_to_dc_normalized(
    c2ws_colmap: List[np.ndarray],
    scene_ext: str,
    dt: Tuple[np.ndarray, float],
) -> List[np.ndarray]:
    """Convert COLMAP (OpenCV convention) c2ws to DreamCatalyst normalised frame.

    DCColmapDataParser (nerfstudio ColmapDataParser) applies two transforms
    before storing Gaussians:

      1. OpenCV → OpenGL flip  (c2w[:3, 1:3] *= -1)
      2. auto_orient_and_center + auto_scale, captured in dataparser_transforms.json

    However, TASE's rasteriser (gsplat) expects OpenCV c2w (camera +Z forward),
    so the final output must also be in OpenCV convention.  The two column-flip
    operations cancel in the rotation block, leaving::

        c2w_dc = T_fwd @ c2w_colmap    (direct, no flip needed)
        c2w_dc[:3, 3] *= scale

    ``T_fwd`` and ``scale`` come directly from the DC run's
    ``dataparser_transforms.json``.
    """
    T_fwd, dp_scale = dt
    result = []
    for c2w in c2ws_colmap:
        c2w_dc = np.eye(4, dtype=np.float64)
        c2w_dc[:3, :] = T_fwd[:3, :] @ c2w  # (3×4) @ (4×4) → (3×4)
        c2w_dc[:3, 3] *= dp_scale
        result.append(c2w_dc)
    return result

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _load_poses(scene: str) -> List[np.ndarray]:
    """Load camera poses for a scene from the shared pose directory."""
    poses_file = POSE_DIR / scene / "_cam_poses.npy"
    if not poses_file.exists():
        raise FileNotFoundError(f"Pose file not found: {poses_file}")
    return list(np.load(poses_file, allow_pickle=True))

def _load_video_poses(scene: str) -> List[np.ndarray]:
    """Load camera poses for video rendering, if available.

    Video poses may differ from the main evaluation poses (e.g. more views,
    or a different trajectory).  If no separate video pose file exists for
    a scene, returns None.
    """
    video_poses_file = POSE_DIR / scene / "edited_scene_cam_poses.npy"
    if not video_poses_file.exists():
        raise FileNotFoundError(f"Video pose file not found: {video_poses_file}")
    return list(np.load(video_poses_file, allow_pickle=True))


def _load_aligned_poses(scene: str, method: str) -> Optional[List[np.ndarray]]:
    """Load manually aligned test poses if available.

    Returns transformed test c2ws in the baseline's frame, or *None*
    if no alignment file exists for this (scene, method) pair.
    """
    path = POSE_DIR / scene / f"aligned_{method}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return list(data["test_c2ws"])


def _build_scene_transform(scene: str, minus_t: bool = True) -> np.ndarray:
    """Return M_actual (4x4), the COLMAP→TASE transform for a scene.

    The stored matrix has +t in the last column, but the actual transform
    applies ``p' = R @ p - t``, so M_actual = M_stored with sign-flipped
    translation column.
    """
    st = np.load(
        str(POSE_DIR / scene / "_scene_transform.npy"), allow_pickle=True
    ).item()
    M = np.array(st["matrix"], dtype=np.float64)
    if minus_t:
        M[:3, 3] *= -1  # stored +t → actual -t
    return M


def _get_baseline_poses(scene: str, c2ws: List[np.ndarray], minus_t: bool = True, use_pose_comp: bool = True) -> List[np.ndarray]:
    """Transform TASE-frame poses to the baseline COLMAP frame.

    For bear the baselines live in a slightly different COLMAP frame (different
    origin) than TASE's normalization; ``pose_comparison.json`` stores a
    matched pair of poses that lets us compute the direct TASE→baseline
    transform.  For bike/face, inverting the scene transform is sufficient.
    """
    if scene == "bear" and use_pose_comp:
        bear_json = POSE_DIR / "bear" / "pose_comparison.json"
        with open(bear_json) as f:
            pc = json.load(f)
        pose_ours = np.array(pc["pose_ours"], dtype=np.float64)
        pose_ns = np.array(pc["pose_ns"], dtype=np.float64)
        pose_ours2ns = pose_ns @ np.linalg.inv(pose_ours)
        return [pose_ours2ns @ p for p in c2ws]
    else:
        M_inv = np.linalg.inv(_build_scene_transform(scene, minus_t=minus_t))
        return [M_inv @ p for p in c2ws]


@torch.no_grad()
def render_scene_jpg(
    scene: Scene,
    c2ws: List[np.ndarray],
    out_dir: Path,
    img_wh: Tuple[int, int] = IMG_WH,
    quality: int = 95,
    render_depth: bool = False,
) -> List[Path]:
    """Render images from a Scene at given poses, save as JPEG.

    When *render_depth* is True, estimated-depth colormaps are also saved
    into *out_dir* alongside the RGB frames as ``depth_XXX.jpg``
    (same directory, matching the reference render format).

    Returns list of saved RGB image paths.
    """
    from nerfview import CameraState
    from TASE.splatting.common.rendering import ColorMapper

    W, H = img_wh
    out_dir.mkdir(parents=True, exist_ok=True)

    colormapper: Optional[object] = None
    if render_depth:
        colormapper = ColorMapper(device=scene.device)

    mode = "RGB+ED" if render_depth else "RGB"
    saved = []
    for i, c2w in enumerate(c2ws):
        camera_state = CameraState(c2w=c2w, aspect=1.0, fov=0.7)
        render_colors, render_alphas, _, _ = scene.rasterize_splats(
            camtoworlds=torch.from_numpy(c2w).float().to(scene.device)[None],
            Ks=torch.from_numpy(camera_state.get_K(img_wh)).float().to(scene.device)[None],
            width=W,
            height=H,
            sh_degree=scene.sh_degree,
            render_mode=mode,
        )
        rgb = render_colors[0, :, :, :3]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"rgb_{i:03d}.jpg"
        cv2.imwrite(str(out_path), rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved.append(out_path)

        if render_depth and colormapper is not None:
            depth_raw = render_colors[0, :, :, 3]  # (H, W)
            alpha = render_alphas[0, :, :, 0]       # (H, W)
            depth_color = colormapper.apply_colormap(depth_raw, alpha)  # (H, W, 3)
            depth_np = (depth_color.cpu().numpy() * 255).astype(np.uint8)
            depth_bgr = cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out_dir / f"depth_{i:03d}.jpg"),
                depth_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )

    return saved

def render_scene_video(
    scene: Scene,
    c2ws: List[np.ndarray],
    out_path: Path,
    img_wh: Tuple[int, int] = IMG_WH,
    fps: int = 30,
) -> None:
    """Render a video from a Scene at given poses.

    Saves an MP4 video at *out_path* with frames rendered at the given poses.
    """
    from nerfview import CameraState

    W, H = img_wh
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio
    video_writer = imageio.get_writer(str(out_path), fps=fps)

    for c2w in c2ws:
        camera_state = CameraState(c2w=c2w, aspect=1.0, fov=0.7)
        render_colors, _, _, _ = scene.rasterize_splats(
            camtoworlds=torch.from_numpy(c2w).float().to(scene.device)[None],
            Ks=torch.from_numpy(camera_state.get_K(img_wh)).float().to(scene.device)[None],
            width=W,
            height=H,
            sh_degree=scene.sh_degree,
            render_mode="RGB",
        )
        rgb = render_colors[0, :, :, :3]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_np = (rgb.cpu().detach().numpy() * 255).astype(np.uint8)
        # rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        video_writer.append_data(rgb_np)

    video_writer.close()


def _load_scene_for_method(
    method: str,
    ckpt_path: Path,
    renormalize: bool = False,
    scene_trafo: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    original_frame: bool = False,
) -> Optional[Scene]:
    """Load a Scene from a checkpoint.

    Args:
        method: Baseline method name (for logging).
        ckpt_path: Path to the checkpoint file.
        renormalize: If True, apply PCA-based coordinate renormalization
            (needed for bear scene baselines whose Gaussian clouds are
            not axis-aligned like the original COLMAP reconstruction).

    For DC the Gaussians stay in the nerfstudio normalised frame; poses are
    transformed forward into that frame at render time instead.
    """
    if scene_trafo is not None:
        rot, trans = scene_trafo
        rot = Quaternion(rot)
        trans = torch.from_numpy(trans).float()
        scene_trafo = (rot, trans)
    try:
        scene = Scene.load_from_ckpt(
            str(ckpt_path),
            renormalize_coordinates=renormalize,
            scene_trafo=scene_trafo
        )
    except Exception as e:
        logger.error("Failed to load %s checkpoint %s: %s", method, ckpt_path, e)
        return None
    # Transform back to the original COLMAP frame if requested (for original_scene renders).
    if original_frame:
        rot, trans = scene.transform
        rot = Quaternion(rot)
        # build the matrix to transform back to the original COLMAP frame
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = rot.rotation_matrix.T  # inverse rotation
        M[:3, 3] = -rot.rotation_matrix.T @ trans
        scene.apply_scene_transform(matrix = M)
    return scene

def render_baselines(
    scenes: List[str],
    methods: List[str],
    out_dir: Path,
    skip_existing: bool = True,
    render_depth: bool = False,
    render_videos: bool = False,
) -> None:
    """Render all baselines for given scenes and methods.

    Output structure:
        out_dir/{scene}/{method}/{experiment}/imgs/rgb_XXX.jpg
        out_dir/{scene}/{method}/{experiment}/imgs/depth_XXX.jpg  (if render_depth)
        out_dir/{scene}/{method}/original_scene/imgs/rgb_XXX.jpg
        out_dir/paper/{scene}/{exp}_{method}_{i}.jpg
    """
    for scene in scenes:
        scene_ext = SCENE_NAME_MAP_INV.get(scene, scene)
        c2ws = _load_poses(scene)
        video_c2ws = _load_video_poses(scene) if render_videos else None  # separate set of poses for videos (if needed)
        paper_imgs = PAPER_IMAGES.get(scene, [])
        video_out = out_dir / "videos" / scene 
        video_out.mkdir(parents=True, exist_ok=True)

        for method in methods:
            logger.info("Rendering %s / %s", method, scene)
            edits, base_ckpt, dt = _discover_method(method, scene, scene_ext)

            if not edits:
                logger.warning("No edits found for %s/%s, skipping", method, scene)
                continue

            # Transform poses into the target method's coordinate frame.
            # Priority: manually aligned poses > automatic transform chain.
            aligned_c2ws = _load_aligned_poses(scene, method)
            if aligned_c2ws is not None:
                logger.info("  Using manually aligned poses for %s/%s", method, scene)
                render_c2ws = aligned_c2ws
                if render_videos:
                    # get the transform from original to aligned frame, apply to video poses as well
                    from TASE.splatting.util.align_poses import solve_similarity_from_poses, apply_similarity_to_c2w
                    R, s, t, F = solve_similarity_from_poses(c2ws, aligned_c2ws)
                    video_render_c2ws = [apply_similarity_to_c2w(p, R, s, t, F) for p in video_c2ws] if render_videos else None
            elif method in ("Ours"):
                render_c2ws = c2ws
                video_render_c2ws = video_c2ws if render_videos else None
            elif method in ("DreamCatalyst"):
                colmap_c2ws = _get_baseline_poses(scene, c2ws)
                render_c2ws = _colmap_to_dc_normalized(colmap_c2ws, scene_ext, dt)
                video_colmap_c2ws = _get_baseline_poses(scene, video_c2ws) if render_videos else None
                video_render_c2ws = _colmap_to_dc_normalized(video_colmap_c2ws, scene_ext, dt) if render_videos else None
            else:
                render_c2ws = _get_baseline_poses(scene, c2ws)
                video_render_c2ws = _get_baseline_poses(scene, video_c2ws) if render_videos else None

            # For bear, GE/DGE/ES baselines need PCA renormalization of
            # the Gaussian cloud (the COLMAP reconstruction is not axis-aligned).
            needs_renormalize = (
                scene == "bear" and not method=="Ours" and aligned_c2ws is None
            )

            # In backwards-compat mode, Ours/GE/DGE share one scene-level
            # original_scene/ (same COLMAP base).  DC and ES get dedicated
            # per-method originals because they use different base models.
            # In normal mode every method gets its own sub-directory.
            if render_depth and method in _BACKWARDS_COMPAT_SHARED_ORIG_METHODS:
                orig_dir = out_dir / scene / "original_scene" / "imgs"
            else:
                orig_dir = out_dir / scene / method / "original_scene" / "imgs"

            # Render original scene.  For the shared slot (Ours/GE/DGE in
            # backwards-compat mode) skip if already written by a prior method.
            orig_video_path = video_out / "original" / f"{method}.mp4"
            if render_depth and method in _BACKWARDS_COMPAT_SHARED_ORIG_METHODS:
                need_orig_imgs = not _renders_complete(orig_dir, len(render_c2ws))
            else:
                need_orig_imgs = not (skip_existing and _renders_complete(orig_dir, len(render_c2ws)))
            need_orig_video = render_videos and not (skip_existing and _video_complete(orig_video_path))

            if base_ckpt is not None and (need_orig_imgs or need_orig_video):
                logger.info("  Rendering original scene for %s/%s", method, scene)
                base_scene = _load_scene_for_method(
                    method, base_ckpt, renormalize=needs_renormalize,
                )
                if base_scene is not None:
                    if need_orig_imgs:
                        render_scene_jpg(base_scene, render_c2ws, orig_dir, render_depth=render_depth)
                    if need_orig_video:
                        render_scene_video(base_scene, video_render_c2ws, orig_video_path)
                    del base_scene
                    torch.cuda.empty_cache()

                # Copy paper images
                if need_orig_imgs and paper_imgs:
                    paper_dir = out_dir / "paper" / scene
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img_name in enumerate(paper_imgs):
                        src = orig_dir / img_name
                        if src.exists():
                            shutil.copy(str(src), str(paper_dir / f"original_{method}_{idx}.jpg"))

            # Render each edit
            for exp_name, ckpt_info in edits.items():
                exp_out = out_dir / scene / method / exp_name / "imgs"
                edit_video_path = video_out / exp_name / f"{method}.mp4"
                need_edit_imgs = not (skip_existing and _renders_complete(exp_out, len(render_c2ws)))
                need_edit_video = render_videos and not (skip_existing and _video_complete(edit_video_path))

                if not need_edit_imgs and not need_edit_video:
                    logger.info("  Skipping %s/%s/%s (exists)", method, scene, exp_name)
                    continue

                logger.info("  Rendering %s/%s/%s", method, scene, exp_name)
                if method == "DreamCatalyst":
                    ckpt_path, _ = ckpt_info
                    edit_scene = _load_scene_for_method(method, ckpt_path)
                else:
                    edit_scene = _load_scene_for_method(
                        method, ckpt_info, renormalize=needs_renormalize,
                    )

                if edit_scene is None:
                    continue

                if need_edit_imgs:
                    render_scene_jpg(edit_scene, render_c2ws, exp_out, render_depth=render_depth)
                if need_edit_video:
                    render_scene_video(edit_scene, video_render_c2ws, edit_video_path)

                del edit_scene
                torch.cuda.empty_cache()

                # Copy paper images
                if need_edit_imgs and paper_imgs:
                    paper_dir = out_dir / "paper" / scene
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img_name in enumerate(paper_imgs):
                        src = exp_out / img_name
                        if src.exists():
                            shutil.copy(str(src), str(paper_dir / f"{exp_name}_{method}_{idx}.jpg"))


def _discover_method(
    method: str, scene: str, scene_ext: str,
) -> Tuple[Dict, Optional[Path], Optional[Tuple[np.ndarray, float]]]:
    """Discover edit checkpoints and base model for a method.

    Returns (edits_dict, base_ckpt_path, dataparser_trafo_or_None).
    For DC, edits dict values are (ckpt, dt_path) tuples.
    For others, edits dict values are plain Paths.
    """
    dt = None
    if method == "GaussianEditor":
        edits = _find_ge_ckpts(scene)  # GE uses TASE scene names
        base = _find_ge_base(scene)
    elif method == "DGE":
        edits = _find_dge_ckpts(scene)  # DGE uses TASE scene names
        base = _find_ge_base(scene)  # GE and DGE share same base
    elif method == "EditSplat":
        edits = _find_es_ckpts(scene_ext)  # ES uses external names (bicycle)
        base = _find_es_base(scene_ext)
    elif method == "DreamCatalyst":
        edits = _find_dc_ckpts(scene_ext)  # DC uses external names (bicycle)
        dc_base = _find_dc_base(scene_ext)
        if dc_base is not None:
            base, dt_path = dc_base
            dt = _load_dataparser_trafo(dt_path)
        else:
            base = None
    elif method == "Ours":
        edits = _find_ours_ckpts(scene)
        base = _find_ours_base(scene)
    else:
        logger.warning("Unknown method: %s", method)
        return {}, None, None

    return edits, base, dt

 
def _find_ours_ckpts(scene: str) -> Dict[str, Path]:
    """Find TASE (Ours) edit checkpoints.

    Structure: OUTPUT/editing/FULL_FT/{scene}/{experiment}/splat_ckpts/last.pt
    """
    base = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/editing/FULL_FT" / scene
    result = {}
    if not base.is_dir():
        return result
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        ckpt = exp_dir / "splat_ckpts" / "last.pt"
        if ckpt.exists():
            result[exp_dir.name] = ckpt
    return result


def _find_ours_base(scene: str) -> Optional[Path]:
    """Find TASE (Ours) original scene checkpoint."""
    base = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/editing/FULL_FT" / scene
    if not base.is_dir():
        return None
    for exp_dir in sorted(base.iterdir()):
        orig = exp_dir / "splat_ckpts" / "orig_scene.pt"
        if orig.exists():
            return orig
    return None


def _renders_complete(img_dir: Path, expected_count: int) -> bool:
    """Check if a render directory already has the expected number of JPGs."""
    if not img_dir.is_dir():
        return False
    jpgs = list(img_dir.glob("rgb_*.jpg"))
    return len(jpgs) >= expected_count


def _video_complete(video_path: Path) -> bool:
    """Check if a video file already exists and is non-empty."""
    return video_path.is_file() and video_path.stat().st_size > 0


def render_ablations(
    scenes: List[str],
    methods: List[str],
    out_dir: Path,
    skip_existing: bool = True,
    render_depth: bool = False,
    render_videos: bool = False,
) -> None:
    """Render ablation variants for TASE.

    Each ablation method (FULL, FULL_FT, NO_EQV, etc.) is an output directory
    under OUTPUT/editing/{method}/{scene}/auto_edit/{experiment}/splat_ckpts/last.pt.
    """

    for scene in scenes:
        c2ws = _load_poses(scene)
        video_c2ws = _load_video_poses(scene) if render_videos else None  # separate set of poses for videos (if needed)

        c2ws = _get_baseline_poses(scene, c2ws, use_pose_comp=False)
        video_c2ws = _get_baseline_poses(scene, video_c2ws, use_pose_comp=False) if render_videos else None

        renormalize_scene = True
        local_poses = not (scene == "face")
        paper_imgs = PAPER_IMAGES.get(scene, [])

        for method in methods:
            edit_base = WORKSPACE_ROOT / "Repos/TASE/OUTPUT/editing" / method / scene
            if (edit_base / "auto_edit").is_dir():
                edit_base = edit_base / "auto_edit"
            if not edit_base.is_dir():
                logger.warning("Ablation dir not found: %s", edit_base)
                continue
            logger.info("Rendering for ablation dir %s", edit_base)

            # Find original scene from any experiment
            orig_ckpt = None
            orig = ""
            for exp_dir in sorted(edit_base.iterdir()):
                if not exp_dir.is_dir():
                    continue
                orig = exp_dir / "splat_ckpts" / "orig_scene.pt"
                if orig.exists():
                    orig_ckpt = orig
                    break

            # load the local poses if available
            if local_poses:
                pose_dir = exp_dir / "renders" / "edited" / "_cam_poses.npy"
                if pose_dir.is_file():
                    c2ws = list(np.load(pose_dir, allow_pickle=True))
                pose_dir_video = exp_dir / "videos" / "edited_scene_cam_poses.npy"
                if render_videos and pose_dir_video.is_file():
                    video_c2ws = list(np.load(pose_dir_video, allow_pickle=True))

            if method in ("PCA", "PCA_FT"):
                paper_imgs = PAPER_IMAGES_PCA.get(scene, [])
            else:
                paper_imgs = PAPER_IMAGES.get(scene, [])

            # Render original
            orig_out = out_dir / scene / method / "original_scene" / "imgs"
            orig_video_path = out_dir / "videos" / scene / "original" / f"{method}.mp4"
            need_orig_imgs = orig_ckpt and not (skip_existing and _renders_complete(orig_out, len(c2ws)))
            need_orig_video = orig_ckpt and render_videos and not (skip_existing and _video_complete(orig_video_path))
            if need_orig_imgs or need_orig_video:
                logger.info("Rendering original for ablation %s/%s", method, scene)
                orig_scene = _load_scene_for_method("FULL_FT", orig_ckpt, renormalize=renormalize_scene, original_frame=not local_poses)
                if orig_scene is not None:
                    if need_orig_imgs:
                        render_scene_jpg(orig_scene, c2ws, orig_out, render_depth=render_depth)
                    if need_orig_video:
                        render_scene_video(orig_scene, video_c2ws, orig_video_path)
                    del orig_scene
                    torch.cuda.empty_cache()

                    # Copy paper images
                    if need_orig_imgs and paper_imgs:
                        paper_dir = out_dir / "paper" / scene
                        paper_dir.mkdir(parents=True, exist_ok=True)
                        for idx, img_name in enumerate(paper_imgs):
                            src = orig_out / img_name
                            if src.exists():
                                shutil.copy(str(src), str(paper_dir / f"original_{method}_{idx}.jpg"))
                            else:
                                logger.warning("Paper image not found: %s", src)
            else:
                logger.info("Skipping original (%s)", orig)

            # Render each experiment
            for exp_dir in sorted(edit_base.iterdir()):
                if not exp_dir.is_dir():
                    logger.warning("Experiment dir not found: %s", exp_dir)
                    continue
                ckpt = exp_dir / "splat_ckpts" / "last.pt"
                if not ckpt.exists():
                    logger.warning("Checkpoint not found: %s", ckpt)
                    continue
                exp_name = exp_dir.name
                exp_out = out_dir / scene / method / exp_name / "imgs"
                exp_video_path = out_dir / "videos" / scene / exp_name / f"{method}.mp4"
                need_exp_imgs = not (skip_existing and _renders_complete(exp_out, len(c2ws)))
                need_exp_video = render_videos and not (skip_existing and _video_complete(exp_video_path))

                if not need_exp_imgs and not need_exp_video:
                    logger.info("  Skipping ablation %s/%s/%s (exists)", method, scene, exp_name)
                    continue
                logger.info("  Rendering ablation %s/%s/%s", method, scene, exp_name)
                edit_scene = _load_scene_for_method(method, ckpt, renormalize=renormalize_scene, original_frame=not local_poses)
                if edit_scene is None:
                    continue
                if need_exp_imgs:
                    render_scene_jpg(edit_scene, c2ws, exp_out, render_depth=render_depth)
                if need_exp_video:
                    render_scene_video(edit_scene, video_c2ws, exp_video_path)
                del edit_scene
                torch.cuda.empty_cache()

                # Copy paper images
                if need_exp_imgs and paper_imgs:
                    paper_dir = out_dir / "paper" / scene
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img_name in enumerate(paper_imgs):
                        src = exp_out / img_name
                        if src.exists():
                            shutil.copy(str(src), str(paper_dir / f"{exp_name}_{method}_{idx}.jpg"))
                        else:
                            logger.warning("Paper image not found: %s", src)



# ---------------------------------------------------------------------------
# CLIP metric helpers
# ---------------------------------------------------------------------------

def _load_clip_models():
    """Load CLIP models onto CUDA. Returns (image_encoder, text_encoder, image_processor, tokenizer)."""
    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to("cuda").eval()
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to("cuda").eval()
    return image_encoder, text_encoder, image_processor, tokenizer


def _encode_image(image_encoder, image_processor, img_np):
    px = image_processor(img_np, return_tensors="pt")["pixel_values"].to("cuda")
    with torch.no_grad():
        feat = image_encoder(pixel_values=px).image_embeds
    return F.normalize(feat, dim=-1)


def _encode_text(text_encoder, tokenizer, text: str):
    toks = tokenizer(
        text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        feat = text_encoder(input_ids=toks["input_ids"].to("cuda")).text_embeds
    return F.normalize(feat, dim=-1)


def _img_to_np(path: str):
    """Load image as uint8 numpy array (C, H, W) for CLIP processor."""
    img = Image.open(path).convert("RGB")
    return (transforms.ToTensor()(img) * 255.0).to(torch.uint8).numpy()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    render_path: str,
    method_names: List[str],
    prompts: Dict,
    datasets: List[str] | None = None,
    clip_use_template: bool = True,
    exclude_images: Dict[str, List[str]] | None = None,
    depth_in_dir_sim: bool = False,
) -> Tuple[Dict, Dict]:
    """Compute CLIP directional similarity and CLIP score in a single pass.

    Args:
        render_path: Directory containing {scene}/{method}/{experiment}/imgs/ renders.
        method_names: List of method names to evaluate.
        prompts: Prompts dict (scene → {original, prompt_template, exp → description}).
        datasets: Scenes to evaluate; defaults to all in prompts.
        clip_use_template: If True (default), CLIP score uses the prompt template
            wrapped target description: ``cosim(feat_edit, encode(template.format(target)))``.
            If False, encodes the raw target description directly without the template.
            Directional similarity always uses ``cosim(feat_edit − feat_orig,
            text_edit − text_orig)`` regardless of this flag.
        exclude_images: Per-dataset image stems to exclude. Format:
            {"bike": ["rgb_003", "rgb_017"], "bear": ["rgb_005"]}.
            Defaults to the module-level ``EXCLUDE_IMAGES`` constant when None.
        depth_in_dir_sim: If True (backwards-compat mode), depth colormaps rendered
            alongside RGB images (in the sibling ``depth/`` directory) are also fed
            to the CLIP image encoder and included when averaging the directional
            similarity.  Depth images are never included in the CLIP score.

    Returns:
        dir_sim: method → dataset → experiment → float
        clip_sc: method → dataset → experiment → float
    """
    image_encoder, text_encoder, image_processor, tokenizer = _load_clip_models()

    if datasets is None:
        datasets = list(prompts.keys())

    exclude_map: Dict[str, set] = {
        ds: set(stems)
        for ds, stems in (exclude_images if exclude_images is not None else EXCLUDE_IMAGES).items()
        if stems
    }
    if exclude_map:
        logger.info("Excluding images per dataset: %s", {ds: sorted(s) for ds, s in exclude_map.items()})

    dir_sim: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in method_names}
    clip_sc: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in method_names}

    for dataset in datasets:
        if dataset not in prompts:
            logger.warning("Skipping dataset %s: no prompts found", dataset)
            continue
        scene_prompts = prompts[dataset]
        dataset_dir = os.path.join(render_path, dataset)

        for method in method_names:
            method_dir = os.path.join(dataset_dir, method)
            original_path = os.path.join(method_dir, "original_scene", "imgs")
            has_original = os.path.isdir(original_path)

            # Backwards-compat fallback: scene-level original_scene is shared by
            # Ours/GE/DGE (they use the same COLMAP base model).  DC and ES render
            # their own per-method original, so the fallback never fires for them.
            # Only active when depth_in_dir_sim is enabled so normal runs are unaffected.
            if not has_original and depth_in_dir_sim:
                scene_level_orig = os.path.join(dataset_dir, "original_scene", "imgs")
                if os.path.isdir(scene_level_orig):
                    original_path = scene_level_orig
                    has_original = True

            if not os.path.isdir(method_dir):
                logger.warning("Skipping %s/%s: directory not found", method, dataset)
                continue

            if not has_original:
                logger.warning("%s/%s has no original_scene/imgs — directional similarity unavailable", method, dataset)

            dir_sim[method][dataset] = {}
            clip_sc[method][dataset] = {}

            orig_files = set(os.listdir(original_path)) if has_original else set()
            # In backwards-compat mode, depth_* files live in the same imgs/ dir:
            # check for them once so we don't re-stat on every experiment.
            has_orig_depth = depth_in_dir_sim and has_original and any(
                f.startswith("depth_") for f in orig_files
            )

            for experiment in sorted(os.listdir(method_dir)):
                render_dir = os.path.join(method_dir, experiment, "imgs")
                if experiment not in scene_prompts or not os.path.isdir(render_dir):
                    continue

                if clip_use_template:
                    prompt_template = scene_prompts["prompt_template"]
                    original_prompt = prompt_template.format(scene_prompts["original"])
                    target_prompt = prompt_template.format(scene_prompts[experiment])
                else:
                    original_prompt = scene_prompts["original"]
                    target_prompt = scene_prompts[experiment]

                text_orig = _encode_text(text_encoder, tokenizer, original_prompt)
                text_edit = _encode_text(text_encoder, tokenizer, target_prompt)
                text_clip = text_edit

                img_files = sorted(
                    f for f in os.listdir(render_dir)
                    if f.endswith(".jpg") or f.endswith(".png")
                )
                exp_dir_scores, exp_clip_scores = [], []

                dataset_exclude = exclude_map.get(dataset, set())
                for img_file in img_files:
                    if not img_file.startswith("rgb_"):
                        continue
                    stem = Path(img_file).stem
                    if stem in dataset_exclude:
                        continue
                    edit_np = _img_to_np(os.path.join(render_dir, img_file))
                    feat_edit = _encode_image(image_encoder, image_processor, edit_np)

                    exp_clip_scores.append(F.cosine_similarity(feat_edit, text_clip).item())

                    # Match original by stem (handles jpg/png mismatch)
                    orig_match = None
                    for ext in (".jpg", ".png"):
                        if stem + ext in orig_files:
                            orig_match = stem + ext
                            break
                    if has_original and orig_match:
                        orig_np = _img_to_np(os.path.join(original_path, orig_match))
                        feat_orig = _encode_image(image_encoder, image_processor, orig_np)
                        exp_dir_scores.append(
                            F.cosine_similarity(feat_edit - feat_orig, text_edit - text_orig).item()
                        )

                # Backwards-compat: include depth colormaps (co-located in imgs/) in dir_sim
                if depth_in_dir_sim and has_orig_depth:
                    depth_edit_files = sorted(
                        f for f in os.listdir(render_dir)
                        if (f.endswith(".jpg") or f.endswith(".png")) and f.startswith("depth_")
                    )
                    for df in depth_edit_files:
                        # Derive index: "depth_007.png" → "007"
                        idx_str = Path(df).stem.split("_", 1)[1]  # e.g. "007"
                        orig_d_match = None
                        for ext in (".jpg", ".png"):
                            cand = f"depth_{idx_str}{ext}"
                            if cand in orig_files:
                                orig_d_match = cand
                                break
                        if orig_d_match is None:
                            continue
                        # Skip views excluded for their paired RGB stem
                        if f"rgb_{idx_str}" in dataset_exclude:
                            continue
                        edit_d_np = _img_to_np(os.path.join(render_dir, df))
                        orig_d_np = _img_to_np(os.path.join(original_path, orig_d_match))
                        feat_edit_d = _encode_image(image_encoder, image_processor, edit_d_np)
                        feat_orig_d = _encode_image(image_encoder, image_processor, orig_d_np)
                        exp_dir_scores.append(
                            F.cosine_similarity(feat_edit_d - feat_orig_d, text_edit - text_orig).item()
                        )

                if exp_clip_scores:
                    clip_sc[method][dataset][experiment] = sum(exp_clip_scores) / len(exp_clip_scores)
                if exp_dir_scores:
                    dir_sim[method][dataset][experiment] = sum(exp_dir_scores) / len(exp_dir_scores)

                logger.info(
                    "  %s/%s/%s: dir_sim=%.4f (%d imgs)  clip_score=%.4f (%d imgs)",
                    method, dataset, experiment,
                    dir_sim[method][dataset].get(experiment, float("nan")), len(exp_dir_scores),
                    clip_sc[method][dataset].get(experiment, float("nan")), len(exp_clip_scores),
                )

    return dir_sim, clip_sc


# ---------------------------------------------------------------------------
# Per-image sample figure generation
# ---------------------------------------------------------------------------

def generate_sample_figures(
    render_path: str,
    method_names: List[str],
    prompts: Dict,
    image_stem: str,
    datasets: List[str] | None = None,
    experiments: List[str] | None = None,
    clip_use_template: bool = True,
    out_dir: Path | None = None,
) -> None:
    """Generate per-image comparison figures with CLIP metrics annotated.

    For each scene/experiment, produces a figure showing the original scene
    render and each method's edited render for the specified image, with
    CLIP directional similarity and CLIP score values annotated per method.

    Args:
        render_path: Directory containing {scene}/{method}/{experiment}/imgs/ renders.
        method_names: List of method names to include in the figure.
        prompts: Prompts dict (scene → {original, prompt_template, exp → description}).
        image_stem: Which render to use, e.g. "rgb_001" (without extension).
        datasets: Scenes to evaluate; defaults to all in prompts.
        experiments: Experiments to generate figures for (e.g. ["enduro", "tiger"]).
            Defaults to all experiments found in prompts.
        clip_use_template: Whether to use prompt template for CLIP score text encoding.
        out_dir: Output directory for figures. Defaults to render_path/sample_figures/.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    image_encoder, text_encoder, image_processor, tokenizer = _load_clip_models()

    if datasets is None:
        datasets = list(prompts.keys())
    if out_dir is None:
        out_dir = Path(render_path) / "sample_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        if dataset not in prompts:
            continue
        scene_prompts = prompts[dataset]
        dataset_dir = os.path.join(render_path, dataset)

        for experiment in sorted(scene_prompts.keys()):
            if experiment in ("original", "prompt_template"):
                continue
            if experiments is not None and experiment not in experiments:
                continue

            if clip_use_template:
                prompt_template = scene_prompts["prompt_template"]
                original_prompt = prompt_template.format(scene_prompts["original"])
                target_prompt = prompt_template.format(scene_prompts[experiment])
            else:
                original_prompt = scene_prompts["original"]
                target_prompt = scene_prompts[experiment]

            text_orig = _encode_text(text_encoder, tokenizer, original_prompt)
            text_edit = _encode_text(text_encoder, tokenizer, target_prompt)

            # Collect data for figure: (method_name, image_array, dir_sim, clip_score)
            panels: List[Tuple[str, np.ndarray, Optional[float], float]] = []

            # Find the original image (use first method that has it)
            orig_img_array = None
            for method in method_names:
                orig_path = os.path.join(dataset_dir, method, "original_scene", "imgs")
                if not os.path.isdir(orig_path):
                    continue
                for ext in (".jpg", ".png"):
                    candidate = os.path.join(orig_path, image_stem + ext)
                    if os.path.isfile(candidate):
                        orig_img_array = np.array(Image.open(candidate).convert("RGB"))
                        orig_np = _img_to_np(candidate)
                        feat_orig = _encode_image(image_encoder, image_processor, orig_np)
                        break
                if orig_img_array is not None:
                    break

            if orig_img_array is None:
                logger.warning("No original image found for %s/%s/%s, skipping", dataset, experiment, image_stem)
                continue

            for method in method_names:
                render_dir = os.path.join(dataset_dir, method, experiment, "imgs")
                if not os.path.isdir(render_dir):
                    continue

                # Find the edited image
                edit_path = None
                for ext in (".jpg", ".png"):
                    candidate = os.path.join(render_dir, image_stem + ext)
                    if os.path.isfile(candidate):
                        edit_path = candidate
                        break
                if edit_path is None:
                    continue

                edit_img_array = np.array(Image.open(edit_path).convert("RGB"))
                edit_np = _img_to_np(edit_path)
                feat_edit = _encode_image(image_encoder, image_processor, edit_np)

                clip_score = F.cosine_similarity(feat_edit, text_edit).item()

                # Directional similarity (uses per-method original if available)
                method_orig_path = os.path.join(dataset_dir, method, "original_scene", "imgs")
                feat_orig_method = feat_orig  # fallback to shared original
                if os.path.isdir(method_orig_path):
                    for ext in (".jpg", ".png"):
                        candidate = os.path.join(method_orig_path, image_stem + ext)
                        if os.path.isfile(candidate):
                            feat_orig_method = _encode_image(
                                image_encoder, image_processor, _img_to_np(candidate)
                            )
                            break

                dir_sim_val = F.cosine_similarity(
                    feat_edit - feat_orig_method, text_edit - text_orig
                ).item()

                panels.append((method, edit_img_array, dir_sim_val, clip_score))

            if not panels:
                continue

            # Per-experiment output subdirectory for individual images
            exp_out_dir = out_dir / f"{dataset}_{experiment}"
            exp_out_dir.mkdir(parents=True, exist_ok=True)

            # Save original image
            Image.fromarray(orig_img_array).save(exp_out_dir / f"original_{image_stem}.jpg", quality=95)

            # Save each method's edited image and per-method metrics JSON
            for method, img, ds_val, cs_val in panels:
                safe_method = method.replace(" ", "_")
                Image.fromarray(img).save(
                    exp_out_dir / f"{safe_method}_{image_stem}.jpg", quality=95
                )

            # Save prompts and per-method scores as a JSON sidecar
            scores_data = {
                "dataset": dataset,
                "experiment": experiment,
                "image_stem": image_stem,
                "original_prompt": original_prompt,
                "target_prompt": target_prompt,
                "scores": {
                    method: {"dir_sim": ds_val, "clip_score": cs_val}
                    for method, _, ds_val, cs_val in panels
                },
            }
            with open(exp_out_dir / "scores.json", "w") as _f:
                json.dump(scores_data, _f, indent=4)

            # Combined overview figure
            fig_path = out_dir / f"{dataset}_{experiment}_{image_stem}.png"
            n_panels = 1 + len(panels)
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
            if n_panels == 1:
                axes = [axes]

            # Original panel
            axes[0].imshow(orig_img_array)
            axes[0].set_title("Original", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            # Method panels
            for i, (method, img, ds_val, cs_val) in enumerate(panels, start=1):
                axes[i].imshow(img)
                axes[i].set_title(method, fontsize=11, fontweight="bold")
                axes[i].set_xlabel(
                    f"Dir. Sim: {ds_val:.4f}\nCLIP Score: {cs_val:.4f}",
                    fontsize=9,
                )
                axes[i].set_xticks([])
                axes[i].set_yticks([])

            fig.suptitle(
                f"{dataset} / {experiment} — {image_stem}\n{target_prompt}",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved sample figure: %s (+ individual images in %s)", fig_path, exp_out_dir)


# ---------------------------------------------------------------------------
# Score aggregation and table generation
# ---------------------------------------------------------------------------

def summarize_scores(scores: Dict) -> Dict[str, Dict[str, float]]:
    """Compute per-dataset and overall averages from nested method→dataset→experiment scores."""
    summary: Dict[str, Dict[str, float]] = {}
    for method, dataset_dict in scores.items():
        summary[method] = {}
        all_vals = []
        for dataset, experiments in dataset_dict.items():
            if experiments:
                ds_avg = sum(experiments.values()) / len(experiments)
                summary[method][dataset] = ds_avg
                all_vals.extend(experiments.values())
        if all_vals:
            summary[method]["average"] = sum(all_vals) / len(all_vals)
    return summary


def _markdown_table(
    scores: Dict,
    methods: List[str],
    metric_label: str,
) -> Tuple[str, Dict]:
    """Build a markdown table with best-per-row bolding.

    Returns (table_string, summary_dict).
    """
    summary = summarize_scores(scores)
    all_datasets = sorted({ds for m in methods if m in scores for ds in scores[m]})
    all_experiments = {
        ds: sorted({
            exp
            for m in methods
            if m in scores and ds in scores[m]
            for exp in scores[m][ds]
        })
        for ds in all_datasets
    }

    def best_bold(row_vals: list) -> list[str]:
        valid = [v for v in row_vals if v is not None]
        best = max(valid) if len(valid) > 1 else None
        cells = []
        for v in row_vals:
            if v is None:
                cells.append("—")
            elif v == best:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        return cells

    col_sep = "|".join(["---:"] * len(methods))
    lines = [
        f"## {metric_label}",
        "",
        "| Dataset | Experiment | " + " | ".join(methods) + " |",
        f"|---|---|{col_sep}|",
    ]

    for ds in all_datasets:
        for exp in all_experiments[ds]:
            vals = [scores.get(m, {}).get(ds, {}).get(exp) for m in methods]
            lines.append(
                f"| {ds} | {exp.replace('_', ' ')} | "
                + " | ".join(best_bold(vals))
                + " |"
            )
        lines.append(f"|---|---|{col_sep}|")
        ds_vals = [summary.get(m, {}).get(ds) for m in methods]
        lines.append(f"| {ds} | **average** | " + " | ".join(best_bold(ds_vals)) + " |")

    lines.append(f"|---|---|{col_sep}|")
    avg_vals = [summary.get(m, {}).get("average") for m in methods]
    lines.append(f"| **all** | **average** | " + " | ".join(best_bold(avg_vals)) + " |")

    return "\n".join(lines), summary


def save_results(
    dir_sim: Dict,
    clip_sc: Dict,
    methods: List[str],
    out_dir: Path,
    label: str = "comparison",
    paper_dir: Optional[Path] = None,
) -> None:
    """Save metric scores as JSON and generate markdown tables.

    If *paper_dir* is given, the markdown table is additionally written
    there so that ``paper/`` contains both images and tables.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dir_sim_scores.json", "w") as f:
        json.dump(dir_sim, f, indent=4)
    with open(out_dir / "clip_scores.json", "w") as f:
        json.dump(clip_sc, f, indent=4)

    dir_table, dir_summary = _markdown_table(dir_sim, methods, "CLIP Directional Similarity")
    clip_table, clip_summary = _markdown_table(clip_sc, methods, "CLIP Score (image vs target text)")

    with open(out_dir / "dir_sim_summary.json", "w") as f:
        json.dump(dir_summary, f, indent=4)
    with open(out_dir / "clip_summary.json", "w") as f:
        json.dump(clip_summary, f, indent=4)

    full_table = dir_table + "\n\n" + clip_table
    table_path = out_dir / f"{label}_table.md"
    table_path.write_text(full_table + "\n")
    logger.info("Table saved to %s", table_path)
    print(full_table)

    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        paper_table = paper_dir / f"{label}_table.md"
        paper_table.write_text(full_table + "\n")
        logger.info("Paper table saved to %s", paper_table)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with notebooks)
# ---------------------------------------------------------------------------

def run_evaluation(recon_path: Path, step: int = 29999):
    ckpt_path = recon_path / f"ckpts/step{step}.pt"
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found at %s, skipping", ckpt_path)
        return None
    scene = Scene.load_from_ckpt(str(ckpt_path))
    scene.cfg.result_dir = str(recon_path)
    runner = Runner(scene.cfg, scene=scene, eval_mode=True)
    return runner.eval(step=step)


def reconstruction_stat_summary(reconstructions: Path, run_eval: bool = False, step: int = 29999) -> Dict[str, float]:
    stats = {}
    for method in sorted(os.listdir(reconstructions)):
        method_path = Path(reconstructions) / method
        if not method_path.is_dir():
            continue
        recon_subdirs = ["bear", "bicycle", "face"]
        stats[method] = {}
        for recon in recon_subdirs:
            recon_path = method_path / recon
            if not recon_path.is_dir():
                continue
            stats_path = recon_path / "stats" / f"val_step{step}.json"
            if not stats_path.exists() or run_eval:
                recon_stats = run_evaluation(recon_path, step=step)
            else:
                with open(stats_path) as f:
                    recon_stats = json.load(f)
            if recon_stats is None:
                continue
            for key in recon_stats:
                stats[method].setdefault(key, 0.0)
                stats[method][key] += recon_stats[key]
        for key in stats[method]:
            stats[method][key] /= len(recon_subdirs)
    with open(Path(reconstructions) / f"summary_stats_step{step}.json", "w") as f:
        json.dump(stats, f, indent=4)
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

BASELINE_METHODS = ["Ours", "GaussianEditor", "DGE", "DreamCatalyst", "EditSplat"]
ABLATION_METHODS = ["FULL_FT", "FULL", "NO_EQV_FT", "NO_EQV", "PCA_FT", "PCA"]
DEFAULT_SCENES = ["bear", "bike", "face"]


def _copy_prompts_json(dst: Path) -> Dict:
    """Ensure prompts.json exists in the output directory. Returns parsed prompts."""
    if not PROMPTS_JSON.exists():
        raise FileNotFoundError(f"Prompts file not found: {PROMPTS_JSON}")
    prompts_dst = dst / "prompts.json"
    if not prompts_dst.exists():
        shutil.copy(str(PROMPTS_JSON), str(prompts_dst))
    with open(prompts_dst) as f:
        return json.load(f)


def _load_exclude_list(spec: str) -> Dict[str, List[str]]:
    """Parse an exclude-images specification into a per-dataset dict.

    *spec* can be:
      - A path to a JSON file mapping dataset → list of stems:
          {"bike": ["rgb_003", "rgb_017"], "bear": ["rgb_005"]}
      - A path to a text file with lines of the form ``dataset:rgb_003`` or
        bare ``rgb_003`` (applied to all datasets).
      - A comma-separated string of ``dataset:stem`` pairs or bare stems.
    """
    path = Path(spec)
    if path.is_file():
        text = path.read_text()
        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {ds: list(stems) for ds, stems in data.items()}
        except json.JSONDecodeError:
            pass
        # Plain text: one entry per line, optional "dataset:stem" prefix
        result: Dict[str, List[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                ds, stem = line.split(":", 1)
                result.setdefault(ds.strip(), []).append(stem.strip())
            else:
                result.setdefault("__all__", []).append(line)
        return result
    # Inline comma-separated: "bike:rgb_003,bear:rgb_005" or "rgb_003,rgb_017"
    result = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            ds, stem = entry.split(":", 1)
            result.setdefault(ds.strip(), []).append(stem.strip())
        else:
            result.setdefault("__all__", []).append(entry)
    return result


def run_pipeline(args: argparse.Namespace) -> None:
    """Main pipeline: render then compute metrics."""
    scenes = args.scenes
    do_baselines = args.mode in ("baselines", "both")
    do_ablations = args.mode in ("ablations", "both")
    do_sample_figures = args.mode == "sample-figures"

    # Load exclude list if specified
    exclude_images: Dict[str, List[str]] | None = None
    if args.exclude_images:
        exclude_images = _load_exclude_list(args.exclude_images)
        # Expand __all__ entries to every known dataset
        if "__all__" in exclude_images:
            all_stems = exclude_images.pop("__all__")
            for ds in scenes:
                exclude_images.setdefault(ds, []).extend(all_stems)

    # Backwards-compat mode forces template encoding (matches the original reference
    # computation where prompts were always wrapped in prompt_template).
    clip_use_template = args.clip_use_template or args.backwards_compat

    # --- Sample figures (separate mode) ---
    if do_sample_figures:
        if not args.sample_image:
            raise ValueError("--sample-image is required for sample-figures mode")
        render_dir = Path(args.baseline_out)
        prompts = _copy_prompts_json(render_dir)
        sample_out = Path(args.sample_figures_out) if args.sample_figures_out else None
        sample_exps = args.sample_experiments if args.sample_experiments else None
        logger.info("=== Generating sample figures for %s ===", args.sample_image)
        generate_sample_figures(
            str(render_dir),
            args.baseline_methods,
            prompts,
            image_stem=args.sample_image,
            datasets=scenes,
            experiments=sample_exps,
            clip_use_template=clip_use_template,
            out_dir=sample_out,
        )
        return

    # --- Baselines ---
    if do_baselines:
        baseline_dir = Path(args.baseline_out)
        prompts = _copy_prompts_json(baseline_dir)

        if not args.metrics_only:
            logger.info("=== Rendering baselines ===")
            render_baselines(
                scenes=scenes,
                methods=args.baseline_methods,
                out_dir=baseline_dir,
                skip_existing=not args.force,
                render_depth=args.backwards_compat,
                render_videos=args.render_videos,
            )

        logger.info("=== Computing baseline metrics ===")
        dir_sim, clip_sc = compute_all_metrics(
            str(baseline_dir), args.baseline_methods, prompts, scenes,
            clip_use_template=clip_use_template,
            exclude_images=exclude_images,
            depth_in_dir_sim=args.backwards_compat,
        )
        save_results(
            dir_sim, clip_sc, args.baseline_methods,
            baseline_dir / "metrics",
            label="baseline_comparison",
            paper_dir=baseline_dir / "paper",
        )

    # --- Ablations ---
    if do_ablations:
        ablation_dir = Path(args.ablation_out)
        prompts = _copy_prompts_json(ablation_dir)

        if not args.metrics_only:
            logger.info("=== Rendering ablations ===")
            logger.info("Scenes: %s, Methods: %s", scenes, args.ablation_methods)
            render_ablations(
                scenes=scenes,
                methods=args.ablation_methods,
                out_dir=ablation_dir,
                skip_existing=not args.force,
                render_depth=args.backwards_compat,
                render_videos=args.render_videos,
            )

        logger.info("=== Computing ablation metrics ===")
        dir_sim, clip_sc = compute_all_metrics(
            str(ablation_dir), args.ablation_methods, prompts, scenes,
            clip_use_template=clip_use_template,
            exclude_images=exclude_images,
            depth_in_dir_sim=args.backwards_compat,
        )
        save_results(
            dir_sim, clip_sc, args.ablation_methods,
            ablation_dir / "metrics",
            label="ablation_comparison",
            paper_dir=ablation_dir / "paper",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Render and evaluate 3DGS editing baselines and ablations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["baselines", "ablations", "both", "sample-figures"],
        default="baselines",
        help="What to evaluate",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene names to evaluate",
    )
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=BASELINE_METHODS,
        help="Baseline methods to include",
    )
    parser.add_argument(
        "--ablation-methods",
        nargs="+",
        default=ABLATION_METHODS,
        help="Ablation methods to include",
    )
    parser.add_argument(
        "--baseline-out",
        type=str,
        default=str(BASELINE_OUT_DIR),
        help="Output directory for baseline renders and metrics",
    )
    parser.add_argument(
        "--ablation-out",
        type=str,
        default=str(ABLATION_OUT_DIR),
        help="Output directory for ablation renders and metrics",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip rendering, only compute metrics on existing renders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if output already exists",
    )
    parser.add_argument(
        "--clip-raw-target",
        dest="clip_use_template",
        action="store_false",
        default=True,
        help=(
            "Encode the raw target description for CLIP score, without wrapping "
            "it in the prompt template. By default, the template is used."
        ),
    )
    parser.add_argument(
        "--exclude-images",
        type=str,
        default=None,
        help=(
            "Image stems to exclude from metric computation. Can be a path to a "
            "text file (one stem per line, e.g. 'rgb_003') or a comma-separated "
            "list (e.g. 'rgb_003,rgb_017,rgb_020')."
        ),
    )
    parser.add_argument(
        "--backwards-compat",
        action="store_true",
        default=False,
        help=(
            "Backwards-compatibility mode: also render estimated-depth colormaps "
            "alongside RGB images (saved to depth/ siblings of imgs/)"
        ),
    )
    parser.add_argument(
        "--sample-image",
        type=str,
        default=None,
        help=(
            "Image stem for sample-figures mode (e.g. 'rgb_001'). "
            "Required when --mode=sample-figures."
        ),
    )
    parser.add_argument(
        "--sample-figures-out",
        type=str,
        default=None,
        help=(
            "Output directory for sample figures. Defaults to "
            "{baseline-out}/sample_figures/."
        ),
    )
    parser.add_argument(
        "--sample-experiments",
        nargs="+",
        default=None,
        help=(
            "Experiments to include in sample figures (e.g. 'enduro tiger'). "
            "Defaults to all experiments in prompts."
        ),
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render and store videos as well as images",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()