# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Callable, Dict, Generator, Optional, Tuple

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from tqdm import tqdm

from TASE.splatting.util.data_util import pose_to_rotation_translation

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".PNG"}


# ---------------------------- Path helpers ----------------------------


def _require_data_root(explicit: Optional[Path]) -> Path:
    """Return an absolute DATA_ROOT. If explicit is relative, join to $DATA_ROOT."""
    env_root = os.environ.get("TASE_ROOT") 
    if explicit is None:
        if not env_root:
            raise ValueError("TASE_ROOT is not set and no --data-root was provided.")
        return (Path(os.path.expandvars(env_root)).expanduser() / "DATA").resolve()
    # if user passed an absolute path, use it; else join with env
    p = Path(explicit)
    if p.is_absolute():
        return p.resolve()
    if not env_root:
        raise ValueError(
            f"Relative --data-root '{explicit}' provided but $TASE_ROOT is not set."
        )
    return (Path(os.path.expandvars(env_root)).expanduser() / "DATA" / p).resolve()


def _resolve_under_root(p: Path | str, root: Path) -> Path:
    """Resolve p: absolute stays, relative is joined under 'root'."""
    p = Path(p)
    return p if p.is_absolute() else (root / p)


# ---------------------------- LLFF loaders ----------------------------


def _find_images_dir(scene_dir: Path, prefer_down: Optional[int]) -> Path:
    """Pick images_{prefer_down}/ if present, else images/, else first images_*."""
    if prefer_down is not None:
        cand = scene_dir / f"images_{prefer_down}"
        if cand.is_dir():
            return cand
    cand = scene_dir / "images"
    if cand.is_dir():
        return cand
    for d in sorted(scene_dir.glob("images_*")):
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No images directory in {scene_dir}")


def _load_poses_bounds(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LLFF poses_bounds.npy -> (N,17). Returns:
      c2w (N,4,4), K (N,3,3) with fx=fy=f, cx=w/2, cy=h/2, bounds (N,2).
    """
    arr = np.load(scene_dir / "poses_bounds.npy")  # (N,17)
    N = arr.shape[0]
    pose_hwf = arr[:, :15].reshape(N, 3, 5)
    bounds = arr[:, 15:17].astype(np.float32)

    c2w_3x4 = pose_hwf[:, :, :4]  # (N,3,4)
    H = pose_hwf[:, 0, 4]
    W = pose_hwf[:, 1, 4]
    f = pose_hwf[:, 2, 4]

    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], N, axis=0)
    c2w[:, :3, :4] = c2w_3x4.astype(np.float32)

    K = np.zeros((N, 3, 3), dtype=np.float32)
    K[:, 0, 0] = f.astype(np.float32)
    K[:, 1, 1] = f.astype(np.float32)
    K[:, 0, 2] = (W / 2).astype(np.float32)
    K[:, 1, 2] = (H / 2).astype(np.float32)
    K[:, 2, 2] = 1.0
    return c2w, K, bounds


def traverse_scene(
    scene_dir: str | Path,
    load_images: bool = False,
    prefer_down: Optional[int] = 4,
    op: Optional[Callable[[np.ndarray, Dict], None]] = None,
) -> Generator[Dict, None, None]:
    """
    Iterate a single LLFF-format scene directory.
    Yields per-frame dict:
      {
        "scene": str, "index": int, "image_path": Path,
        "c2w": (4,4) torch.float32, "K": (3,3) torch.float32,
        "bounds": (2,) torch.float32, "H": int, "W": int,
        # if load_images=True: "image_bgr": np.uint8 (H,W,3) or passed to op(...)
      }
    If `op` is provided, it's called as op(image_bgr, meta) per frame.
    """
    scene_dir = Path(scene_dir)
    if not (scene_dir / "poses_bounds.npy").exists():
        raise FileNotFoundError(f"poses_bounds.npy not found in {scene_dir}")

    images_dir = _find_images_dir(scene_dir, prefer_down)
    c2w_np, K_np, bounds_np = _load_poses_bounds(scene_dir)

    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix in IMG_EXTS])
    if len(img_paths) != c2w_np.shape[0]:
        print(
            f"[WARN] {scene_dir.name}: {len(img_paths)} images vs {c2w_np.shape[0]} poses."
        )
    N = min(len(img_paths), c2w_np.shape[0])

    pbar = tqdm(range(N), file=sys.stdout, desc=f"Traversing {scene_dir.name}")
    for i in pbar:
        meta = {
            "scene": scene_dir.name,
            "index": i,
            "image_path": img_paths[i],
            "c2w": torch.from_numpy(c2w_np[i]),
            "K": torch.from_numpy(K_np[i]),
            "bounds": torch.from_numpy(bounds_np[i]),
            "H": int(round(2 * (K_np[i, 1, 2]))),
            "W": int(round(2 * (K_np[i, 0, 2]))),
        }

        if load_images:
            img = cv2.imread(str(img_paths[i]), cv2.IMREAD_COLOR)  # BGR uint8
            if img is None:
                pbar.write(f"[WARN] Failed to read {img_paths[i]}")
                continue
            H, W = img.shape[:2]
            if op is not None:
                op(img, meta)  # side-effect function
            else:
                meta["image_bgr"] = img
            # rescale K if needed
            H_guess = int(round(2 * (K_np[i, 1, 2])))
            W_guess = int(round(2 * (K_np[i, 0, 2])))
            if H != H_guess or W != W_guess:
                # rescale K to match actual image size
                sx, sy = (W / max(1, W_guess)), (H / max(1, H_guess))
                K_np[i, 0, :] *= sx
                K_np[i, 1, :] *= sy
                # pbar.write(f"[WARN] Rescaling K from {H_guess}x{W_guess} to {H}x{W}")
                meta["K"] = torch.from_numpy(K_np[i])
            meta["H"] = H
            meta["W"] = W

        yield meta


# ---------------------------- COLMAP readers ----------------------------


def _read_fmt(fid, fmt: str):
    size = struct.calcsize(fmt)
    data = fid.read(size)
    if len(data) != size:
        raise EOFError("Unexpected EOF")
    return struct.unpack(fmt, data)


def _read_points3D_txt(path: Path):
    ids, xyz, rgb, err = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            # Format: POINT3D_ID, X Y Z, R G B, ERROR, TRACK_LENGTH, (image_id, point2D_idx)...
            ids.append(int(toks[0]))
            xyz.append([float(toks[1]), float(toks[2]), float(toks[3])])
            rgb.append([int(toks[4]), int(toks[5]), int(toks[6])])
            err.append(float(toks[7]))
    return {
        "id": np.asarray(ids, dtype=np.int64),
        "xyz": np.asarray(xyz, dtype=np.float64),
        "rgb": np.asarray(rgb, dtype=np.uint8),
        "error": np.asarray(err, dtype=np.float64),
    }


def _read_points3D_bin(path: Path):
    with open(path, "rb") as fid:
        ids, xyzs, rgbs, errs = [], [], [], []

        head = fid.read(8)
        if len(head) < 8:
            return {
                "id": np.empty(0, np.int64),
                "xyz": np.empty((0, 3), np.float64),
                "rgb": np.empty((0, 3), np.uint8),
                "error": np.empty(0, np.float64),
            }
        (maybe_count,) = struct.unpack("<Q", head)

        def read_one_point():
            (pid,) = _read_fmt(fid, "<Q")
            x, y, z = _read_fmt(fid, "<3d")
            r, g, b = _read_fmt(fid, "<3B")
            (err,) = _read_fmt(fid, "<d")
            (track_len,) = _read_fmt(fid, "<Q")
            fid.seek(8 * track_len, 1)  # skip track
            ids.append(pid)
            xyzs.append([x, y, z])
            rgbs.append([r, g, b])
            errs.append(err)

        LEGACY_THRESHOLD = 1_000_000
        if 0 < maybe_count < LEGACY_THRESHOLD:
            for _ in range(maybe_count):
                read_one_point()
        else:
            fid.seek(0)
            try:
                while True:
                    read_one_point()
            except EOFError:
                pass

    return {
        "id": np.asarray(ids, dtype=np.int64),
        "xyz": np.asarray(xyzs, dtype=np.float64),
        "rgb": np.asarray(rgbs, dtype=np.uint8),
        "error": np.asarray(errs, dtype=np.float64),
    }


def _find_sparse_dir(scene_dir: Path) -> Path:
    """Locate the COLMAP sparse model dir inside a Mip-NeRF 360 scene."""
    cand = [scene_dir / "sparse" / "0", scene_dir / "sparse"]
    for c in cand:
        if (c / "points3D.bin").exists() or (c / "points3D.txt").exists():
            return c
    for p in scene_dir.rglob("points3D.*"):
        return p.parent
    print(f"No COLMAP sparse model found in {scene_dir}")
    return None


def load_mipnerf360_pointcloud(
    scene_dir: str | Path,
) -> Optional[Dict[str, np.ndarray]]:
    """Load COLMAP sparse point cloud for a single Mip-NeRF 360 scene."""
    scene_dir = Path(scene_dir)
    sparse_dir = _find_sparse_dir(scene_dir)
    bin_path = sparse_dir / "points3D.bin"
    txt_path = sparse_dir / "points3D.txt"
    if bin_path.exists():
        return _read_points3D_bin(bin_path)
    if txt_path.exists():
        return _read_points3D_txt(txt_path)
    print(f"[WARN] No points3D.bin or points3D.txt in {sparse_dir}")
    return None


# ---------------------------- Utils ----------------------------


def convert_cam_mipNerf360(rot: np.ndarray) -> np.ndarray:
    """Convert from mipNerf360 camera to TASE camera (both w.r.t. OpenCV coord sys)."""
    q = Quaternion(rot)
    # rotate 90 deg around z axis, then 180 deg around y axis
    q_z90 = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
    q = q * q_z90
    q_y180 = Quaternion(axis=[0, 1, 0], angle=np.pi)
    q = q * q_y180
    return q.elements  # (w, x, y, z)


def evenly_spaced_val_mask(n: int, val_fraction: float = 0.10) -> np.ndarray:
    """
    Deterministic 'every k-th' selection with centered offset.
    For 10% -> k≈10, select indices i where (i - offset) % k == 0.
    """
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(round(1.0 / max(1e-9, val_fraction))))
    offset = k // 2
    m = np.zeros(n, dtype=bool)
    m[np.arange(offset, n, k)] = True
    if not m.any():
        m[n // 2] = True
    return m


# ---------------------------- Core pipeline ----------------------------


def create_TASE_dataset(
    scene: str,
    data_root: Optional[Path],
    old_data_subdir: str,
    new_data_subdir: Optional[str],
    metadata_template: Path,
    generate_features: bool,
    feature_hacks: bool,
    feature_encoder_ckpt: Optional[Path],
    generate_depth: bool,
    depth_model_name: str,
    val_fraction: float,
    prefer_down: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    use_pca: bool = False,
    pca_path: Optional[Path] = None,
    raw_dino: bool = False,
) -> None:
    root = _require_data_root(data_root)

    if new_data_subdir is None:
        new_data_subdir = (
            feature_encoder_ckpt.parent.parent.name
            if generate_features
            else f"TASE_{old_data_subdir}"
        )

    # Load template (resolve relative templates under new_data_subdir first, then root)
    mt = metadata_template
    if not mt.is_absolute():
        mt = root / old_data_subdir / mt
    if not mt.exists():
        mt = root / mt  # fallback
    if not mt.exists():
        raise FileNotFoundError(f"Metadata template not found: {metadata_template}")

    scene_meta = json.loads(Path(mt).read_text())

    # Prepare models
    if generate_features:
        from TASE.image_gen.dimensionality_reduction.embedding_model import (
            EmbeddingModel,
        )

        if raw_dino:
            encoder = EmbeddingModel.from_backbone("dinov3", device=device, dtype=dtype)
        else:
            encoder = EmbeddingModel.load_ckpt(feature_encoder_ckpt, pca_path=pca_path if use_pca else None)
            encoder = encoder.to(device, dtype).eval()
    else:
        encoder = None

    if generate_depth:
        depth_model = (
            torch.hub.load(
                "yvanyin/metric3d", depth_model_name, pretrain=True, map_location=device
            )
            .to(device, dtype=dtype)
            .eval()
        )
        depth_model_name_full = f"yvanyin/metric3d:{depth_model_name}"
    else:
        depth_model = None
        depth_model_name_full = None

    # Scene description
    extra = (
        " and depth, normals, confidences and background masks"
        if generate_depth
        else ""
    )
    stat_path = None
    if feature_encoder_ckpt is not None and not raw_dino:
        stat_path = feature_encoder_ckpt.parent.parent / "stats" if not use_pca else pca_path.parent / "stats"
        for stat_file in os.listdir(stat_path):
            if stat_file.startswith(feature_encoder_ckpt.stem) and stat_file.endswith(
                ".pt"
            ):
                stat_path = stat_path / stat_file
                break
        # copy the stat file to the new dataset dir
        if generate_features and stat_path.is_file():
            (root / new_data_subdir / "stats").mkdir(parents=True, exist_ok=True)
            dst_stat_path = root / new_data_subdir / "stats" / stat_path.name
            if not dst_stat_path.exists():
                dst_stat_path.write_bytes(stat_path.read_bytes())
                print(f"Copied stats file to {dst_stat_path}")
            stat_path = dst_stat_path
        else:
            stat_path = None
    feature_encoder_desc = "raw_dinov3 (identity mode)" if raw_dino else (str(feature_encoder_ckpt) if generate_features else None)
    scene_meta.update(
        {
            "name": scene,
            "description": (
                f"TASE dataset generated from the mipNeRF360 scene '{scene}'. "
                f"Contains truncation-aware semantic feature maps{extra}."
            ),
            "feature_encoder": feature_encoder_desc,
            "feature_stat_path": str(stat_path) if stat_path else None,
            "depth_model": depth_model_name_full,
        }
    )

    # Output dirs (under DATA_ROOT/new_data_subdir)
    sub = scene_meta["subdirs"]
    img_dir = root / new_data_subdir / sub["image_dir"]
    datamap_dir = root / new_data_subdir / sub["datamap_dir"]
    meta_out_dir = root / new_data_subdir / sub["metadata_dir"]
    pc_out_dir = root / new_data_subdir / sub["pointcloud_dir"]
    img_dir.mkdir(parents=True, exist_ok=True)
    datamap_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)
    pc_out_dir.mkdir(parents=True, exist_ok=True)

    # Input scene dir
    scene_in = root / old_data_subdir / scene

    # Traverse samples
    buffered_samples = []
    for meta in traverse_scene(scene_in, load_images=True, prefer_down=prefer_down):
        sample_name = scene + "_" + meta["image_path"].stem
        data_maps = {}

        img_bgr = meta.pop("image_bgr")
        if generate_features:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = (
                torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )  # (1,3,H,W)
            with torch.no_grad():
                feats = encoder.encode(
                    img_rgb.to(next(encoder.parameters()).device),
                    use_pca=use_pca,
                )  # (1,H',W',C)
            feats = feats.squeeze(0).cpu().numpy()  # (H',W',C)

            if feature_hacks:
                # crop out the outer x% of the feature map to reduce artifacts
                h, w = feats.shape[:2]
                ps = int(img_bgr.shape[0] / feats.shape[0])
                crop_frac = 0.03
                ch, cw = int(h * crop_frac), int(w * crop_frac)
                feats = feats[ch : h - ch, cw : w - cw]
                # crop the original image the same way for encoding consistency
                img_bgr = img_bgr[ch * ps : (h - ch) * ps, cw * ps : (w - cw) * ps]
                # shift the principal point in K accordingly
                K = meta["K"]
                K[0, 2] -= cw * ps
                K[1, 2] -= ch * ps
                # resize the image to reduce computation and memory
                # is it smart to change the effective ps in this way?
                if prefer_down <= 2:
                    fac = 0.5
                    h, w = img_bgr.shape[:2]
                    img_bgr = cv2.resize(
                        img_bgr,
                        (int(w * fac), int(h * fac)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    K *= fac

                meta["K"] = K
                meta["H"] = img_bgr.shape[0]
                meta["W"] = img_bgr.shape[1]

            data_maps["reduced_embeddings"] = feats

        if generate_depth:
            # Placeholder for depth pipeline if you add it later
            print("Depth generation not implemented yet.")
            pass

        if data_maps:
            np.savez_compressed(datamap_dir / f"{sample_name}.npz", **data_maps)

        cv2.imwrite(str(img_dir / f"{sample_name}.jpg"), img_bgr)

        # Camera conversion
        cam = meta["c2w"].numpy()
        K = meta["K"].numpy()
        H, W = meta["H"], meta["W"]

        if H != img_bgr.shape[0] or W != img_bgr.shape[1]:
            raise ValueError(
                f"Image size mismatch: meta {H}x{W} vs actual {img_bgr.shape[0]}x{img_bgr.shape[1]}"
            )

        intrinsics = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "height": int(H),
            "width": int(W),
        }

        rotation, translation = pose_to_rotation_translation(cam)
        rotation = convert_cam_mipNerf360(rotation)

        sample_meta = {
            "name": sample_name,
            "description": "mipNeRF360 image.",
            "translation": translation.tolist(),
            "rotation_wxyz": rotation.tolist(),
            "pose_type": "mipNeRF360",
            "intrinsics": intrinsics,
            "generator": "real (mipNeRF360)",
            "timestamp": 0,
            "generation": 0,
        }
        buffered_samples.append((sample_name, sample_meta))

    # Split train/val
    N = len(buffered_samples)
    val_mask = evenly_spaced_val_mask(N, val_fraction=val_fraction)
    samples_train, samples_val = {}, {}
    for i, (name, s_meta) in enumerate(buffered_samples):
        (samples_val if val_mask[i] else samples_train)[name] = s_meta

    scene_meta["samples_train"] = samples_train
    scene_meta["samples_val"] = samples_val
    scene_meta["splits"] = {
        "train": list(samples_train.keys()),
        "val": list(samples_val.keys()),
        "strategy": "evenly_spaced_stride",
        "val_fraction": float(val_fraction),
    }

    # Point cloud
    pc = load_mipnerf360_pointcloud(scene_in)
    if pc is not None:
        np.savez_compressed(pc_out_dir / f"{scene}.npz", **pc)
        print(
            f"Saved pointcloud with {pc['xyz'].shape[0]} points to {pc_out_dir / f'{scene}.npz'}"
        )
    else:
        print("No pointcloud found.")

    # copy over the stats if they exist

    # Save metadata
    scene_meta_path = meta_out_dir / f"{scene}.json"
    scene_meta_path.write_text(json.dumps(scene_meta, indent=4))
    print(f"Saved TASE dataset for scene '{scene}' -> {scene_meta_path}")


# ---------------------------- CLI ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build TASE dataset from Mip-NeRF 360 scene")
    p.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        required=True,
        help="One or more scene names under <DATA_ROOT>/<old-subdir>/",
    )
    p.add_argument(
        "--old-subdir",
        type=str,
        default="mipNeRF360",
        help="Source dataset subdir under DATA_ROOT (e.g. mipNeRF360)",
    )
    p.add_argument(
        "--new-subdir",
        type=str,
        default="TASE_360",
        help="Target TASE subdir under DATA_ROOT",
    )
    p.add_argument(
        "--metadata-template",
        type=Path,
        default=Path("scene_template.json"),
        help="Path to scene_template.json (absolute or relative to old-subdir or DATA_ROOT)",
    )
    p.add_argument(
        "--features", action="store_true", help="Generate semantic feature maps"
    )
    p.add_argument(
        "--feature-ckpt",
        type=Path,
        nargs="*",
        default=[],
        help="One or more encoder checkpoints (used if --features).",
    )
    p.add_argument(
        "--depth", action="store_true", help="Generate depth maps (placeholder)"
    )
    p.add_argument(
        "--depth-model-name",
        type=str,
        default="metric3d_vit_large",
        help="Metric3D model name (torch.hub)",
    )
    p.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction")
    p.add_argument(
        "--prefer-down",
        type=int,
        default=4,
        help="Prefer images_<down> if present (use -1 to disable)",
    )
    p.add_argument(
        "--feature-hacks",
        action="store_true",
        help="Enable this option to apply experimental hacks to improve feature splatting quality",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override DATA_ROOT (absolute or relative to $DATA_ROOT)",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--use-pca",
        action="store_true",
        help="Whether to use PCA to reduce feature dimensionality.",
    )
    p.add_argument(
        "--pca-path",
        type=Path,
        default=None,
        help="Path to PCA components (if --use-pca). Should be a .pkl file containing a dict with 'components' and 'mean'.",
    )
    p.add_argument(
        "--raw-dino",
        action="store_true",
        help="Use raw DINOv3 features without adapter (identity mode). No --feature-ckpt needed.",
    )
    return p.parse_args()


def colmap_c2w_from_image(img):
    """Convert COLMAP (qvec, tvec) to camera-to-world 3x4 matrix."""
    rot_q_xyzw = img.frame.rig_from_world.rotation.quat
    Rcw = Quaternion(
        rot_q_xyzw[3], rot_q_xyzw[0], rot_q_xyzw[1], rot_q_xyzw[2]
    ).rotation_matrix  # world->cam
    tcw = img.frame.rig_from_world.translation
    Rwc = Rcw.T
    twc = -Rwc @ tcw
    c2w = np.concatenate([Rwc, twc[:, None]], axis=1)  # (3,4)
    return c2w


def to_llff_axes(c2w):
    """
    LLFF uses OpenGL coords: x-right, y-up, z-back.
    COLMAP camera coords are x-right, y-down, z-forward for the *camera*;
    after c2w, many pipelines flip the z to 'back' and y to 'up'.
    A common conversion is: flip y and z axes.
    """
    A = np.diag([1, -1, -1, 1])[:3, :]  # (3,4): flips y and z
    c2w_llff = (
        A @ np.vstack([c2w, [0, 0, 0, 1]])[:4, :]
    )  # make it 4x4, apply, back to 3x4
    return c2w_llff[:3, :4]


def reorient_camera_axes(c2w):
    """
    Reorient camera axes (y-up, z-back) without rotating world coordinates.
    World frame stays as in COLMAP; camera basis is post-rotated.
    """
    q1 = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)  # rotate camera axes: z-90
    q2 = Quaternion(axis=[0, 1, 0], angle=np.pi)  # then y-180
    q = q2 * q1  # apply q1 then q2
    R_cam = q.rotation_matrix
    c2w_new = np.array(c2w, dtype=c2w.dtype, copy=True)
    c2w_new[:3, :3] = c2w_new[:3, :3] @ R_cam  # right-multiply: change camera axes only
    # translation unchanged
    return c2w_new[:3, :4]


def per_image_bounds(img, points3D, q_low=5.0, q_high=95.0):
    """
    Estimate near/far from distances of points observed in this image.
    Uses robust percentiles to avoid outliers.
    """
    # Gather 3D points seen by this image
    dists = []
    cam_center = (colmap_c2w_from_image(img))[:, 3]
    for p2d in img.points2D:
        p3d_id = p2d.point3D_id
        if p3d_id < 0:  # unmatched
            continue
        if p3d_id not in points3D:
            continue
        p = points3D[p3d_id].xyz
        dists.append(np.linalg.norm(p - cam_center))
    if len(dists) < 10:
        # Fallback if too few points: generic near/far
        return 0.1, 5.0
    dists = np.asarray(dists)
    near = np.percentile(dists, q_low)
    far = np.percentile(dists, q_high)
    # keep a small gap
    if far <= near:
        far = near * 1.5
    return float(near), float(far)


def write_poses_bounds_from_recon(sparse, out_path):
    import pycolmap as pcm

    if not isinstance(sparse, pcm.Reconstruction):
        rec = pcm.Reconstruction(sparse)  # loads cameras, images, points3D
    else:
        rec = sparse
    # Map for easier access
    cams = rec.cameras
    imgs = rec.images  # dict: image_id -> Image
    pts3 = rec.points3D

    # Deterministic order
    images_sorted = sorted(imgs.values(), key=lambda im: im.name)

    rows = []
    for im in images_sorted:
        cam = cams[im.camera_id]
        H, W = cam.height, cam.width
        fx = cam.params[0]
        fy = (
            cam.params[1]
            if cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV")
            and len(cam.params) >= 2
            else fx
        )
        f = float(0.5 * (fx + fy))

        # Pose
        c2w = colmap_c2w_from_image(im)  # (3,4)
        # c2w = to_llff_axes(c2w)  # align to LLFF axes (y-up, z-back)
        c2w = reorient_camera_axes(c2w)  # reorient camera axes only

        # Bounds
        near, far = per_image_bounds(im, pts3, q_low=5.0, q_high=95.0)

        # Assemble 3x5: [c2w | [H,W,f]^T]
        hwf_col = np.array([[H], [W], [f]], dtype=np.float64)
        pose_3x5 = np.concatenate([c2w, hwf_col], axis=1)  # (3,5)

        row = np.concatenate(
            [pose_3x5.reshape(-1), np.array([near, far])], axis=0
        )  # (17,)
        rows.append(row)

    arr = np.stack(rows, axis=0)  # (N, 17)
    np.save(out_path, arr)
    print(f"Saved {arr.shape} to {out_path}")


def run_colmap(scene: Path):
    # pip install pycolmap opencv-python numpy
    import os, pycolmap as pcm

    # --- Paths ---
    root = scene
    imgs = os.path.join(root, "images")
    db = os.path.join(root, "database.db")
    sparse = os.path.join(root, "sparse")
    if os.path.exists(db):
        os.remove(db)

    # --- 2) SIFT features ---
    pcm.extract_features(db, imgs)
    # --- 3) Pair selection & matching ---
    pcm.match_exhaustive(db)  # or: match_sequential()/match_spatial()
    # (Recommended) geometric verification / filtering of matches
    # pcm.verify_matches(db)
    # --- 4) Incremental SfM (sparse reconstruction) ---
    maps = pcm.incremental_mapping(
        database_path=db, image_path=imgs, output_path=sparse
    )
    # write out
    maps[0].write(os.path.join(sparse, "0"))
    # write the poses and bounds in mipNeRF360 format
    write_poses_bounds_from_recon(maps[0], os.path.join(root, "poses_bounds.npy"))


def main():
    args = parse_args()

    if args.features and not args.raw_dino and not args.feature_ckpt:
        raise ValueError("--features requires --feature-ckpt (or --raw-dino)")
    
    if args.use_pca and args.pca_path is None:
        raise ValueError("--use-pca requires --pca-path")

    prefer_down = None if args.prefer_down < 0 else args.prefer_down
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32

    for scene in args.scenes:
        # check for colmap model, run if not present
        scene_dir = _require_data_root(args.data_root) / args.old_subdir / scene
        if not _find_sparse_dir(scene_dir):
            print(f"COLMAP model not found for scene '{scene}', running COLMAP...")
            run_colmap(scene_dir)
        if True or not (scene_dir / "poses_bounds.npy").exists():
            print(f"poses_bounds.npy not found for scene '{scene}', generating...")
            write_poses_bounds_from_recon(
                _find_sparse_dir(scene_dir), scene_dir / "poses_bounds.npy"
            )

        if args.raw_dino:
            print(f"Processing scene '{scene}' with raw DINOv3 features (no adapter)...")
            create_TASE_dataset(
                scene=scene,
                data_root=args.data_root,
                old_data_subdir=args.old_subdir,
                new_data_subdir=f"{args.new_subdir}_raw_dino",
                metadata_template=args.metadata_template,
                generate_features=True,
                feature_hacks=args.feature_hacks,
                feature_encoder_ckpt=None,
                generate_depth=bool(args.depth),
                depth_model_name=args.depth_model_name,
                val_fraction=float(args.val_frac),
                prefer_down=prefer_down,
                use_pca=False,
                pca_path=None,
                raw_dino=True,
                device=device,
                dtype=dtype,
            )
        elif args.features:
            for ckpt in args.feature_ckpt:
                print(f"Processing scene '{scene}' with features from {ckpt}...")
                feat_model_name = "PCA" if args.use_pca else ckpt.parent.parent.name 
                create_TASE_dataset(
                    scene=scene,
                    data_root=args.data_root,
                    old_data_subdir=args.old_subdir,
                    new_data_subdir=f"{args.new_subdir}_{feat_model_name}",
                    metadata_template=args.metadata_template,
                    generate_features=True,
                    feature_hacks=args.feature_hacks,
                    feature_encoder_ckpt=ckpt,
                    generate_depth=bool(args.depth),
                    depth_model_name=args.depth_model_name,
                    val_fraction=float(args.val_frac),
                    prefer_down=prefer_down,
                    use_pca=args.use_pca,
                    pca_path=args.pca_path,
                    raw_dino=args.raw_dino,
                    device=device,
                    dtype=dtype,
                )
        else:
            create_TASE_dataset(
                scene=scene,
                data_root=args.data_root,
                old_data_subdir=args.old_subdir,
                new_data_subdir=args.new_subdir,
                metadata_template=args.metadata_template,
                feature_hacks=bool(args.feature_hacks),
                generate_features=bool(args.features),
                feature_encoder_ckpt=args.feature_ckpt,
                generate_depth=bool(args.depth),
                depth_model_name=args.depth_model_name,
                val_fraction=float(args.val_frac),
                use_pca=args.use_pca,
                pca_path=args.pca_path,
                raw_dino=args.raw_dino,
                prefer_down=prefer_down,
                device=device,
                dtype=dtype,
            )


def convert_transforms_json(old_path: Path, TASE_json: Path, out_path: Path):
    """Convert the transforms.json form a nerf2nerf dataset to a regular COLMAP format."""
    import json
    from pathlib import Path
    from TASE.splatting.util.data_util import rotation_translation_to_pose
    import numpy as np

    with open(old_path, "r") as f:
        old_data = json.load(f)

    with open(TASE_json, "r") as f:
        TASE_data = json.load(f)

    frames = old_data["frames"]

    TASE_name = TASE_data["name"]
    for frame in frames:
        filename = frame["file_path"]
        filename = Path(filename).stem
        key = f"{TASE_name}_{filename}"
        if key in TASE_data["samples_train"]:
            TASE_frame = TASE_data["samples_train"][key]
        elif key in TASE_data["samples_val"]:
            TASE_frame = TASE_data["samples_val"][key]
        else:
            print(f"Frame {key} not found in TASE json!")
            continue
        rot = TASE_frame["rotation_wxyz"]
        trans = TASE_frame["translation"]
        c2w = rotation_translation_to_pose(rot, trans)  # (4,4)
        w2c = np.linalg.inv(c2w)
        frame["transform_matrix"] = w2c.tolist()

    # write out the new json
    with open(out_path, "w") as f:
        json.dump(old_data, f, indent=4)


if __name__ == "__main__":
    main()
    '''
    convert_transforms_json(
        Path(
            "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Repos/GaussianEditor/dataset/bear/old_transforms.json"
        ),
        Path("DATA/TASE_360_AE_EQV/metadata/bear.json"),
        Path(
            "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Repos/GaussianEditor/dataset/bear/transforms.json"
        ),
    )
    '''

"""

cd TASE/splatting/data
python mipNerf_tase_converter.py \
  --scenes bear face bicycle counter garden \
  --old-subdir /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Data/mipNeRF360 \
  --new-subdir TASE_360 \
  --metadata-template scene_template.json \
  --features \
  --feature-ckpt \
    /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Results/MRL/dinov3/Ablations/AE_EQV/ckpt/epoch_1.pth \
  --val-frac 0.10 \
  --prefer-down 4 \
  --device cuda \
  --use-pca \
  --pca-path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/fat2hi/Workspace/Repos/TASE/OUTPUT/PCA/dinov3/torch_ipca.pkl 
"""
