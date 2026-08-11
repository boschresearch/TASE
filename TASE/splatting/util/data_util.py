# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Literal, Tuple, List, Union, Any, Optional
from collections import defaultdict
import torch
from TASE.splatting.trainer.full_images_datamanager import FullImageDatamanager
import os
from pyquaternion import Quaternion
import math
from torch.nn import functional as F


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    path = Path(os.path.expandvars(path)).expanduser()
    if not path.is_absolute():
        path = "$TASE_ROOT/" + str(path)
    path = Path(os.path.expandvars(path)).expanduser()
    return path


def rotation_translation_to_pose(r_quat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose


def mat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to (w, x, y, z) via the trace method. This is used because the orthogonality check in pyquaternion is too strict."""
    # Preallocate
    q = np.empty(4, dtype=float)
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2  # S = 4*w
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    else:
        # Find the largest diagonal element
        idx = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if idx == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*x
            q[1] = 0.25 * S
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif idx == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*y
            q[2] = 0.25 * S
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*z
            q[3] = 0.25 * S
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S

    return q / np.linalg.norm(q)  # normalize for safety


def pose_to_rotation_translation(
    pose: np.ndarray, ortho_tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4×4 pose matrix to quaternion rotation (w, x, y, z) and translation.
    If the top‐left 3×3 isn’t quite orthogonal, project it via SVD.
    """
    assert pose.shape == (4, 4), "Pose must be a 4×4 matrix."
    R = pose[:3, :3]
    # Check orthogonality: R Rᵀ ≈ I
    if not np.allclose(R @ R.T, np.eye(3), atol=ortho_tol):
        # Project onto SO(3) via SVD: R' = U Vᵀ
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        # Ensure a proper rotation (det=+1)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

    # Now convert to quaternion
    q = mat2quat(R)
    # q = Quaternion._from_matrix(R).elements
    t = pose[:3, 3]
    return q, t


def resize_intrinsics(
    K: torch.Tensor, scale: Union[float, int, Tuple[float, float], torch.Tensor]
) -> torch.Tensor:
    """
    Scale one or more 3×3 camera intrinsics for image resizing.

    Args:
        K:      Tensor of shape (..., 3, 3), e.g. (3,3) or (B,3,3).
        scale:  - single float/int for uniform scaling (sx = sy),
                - tuple (sx, sy) for the same scale on all Ks,
                - or Tensor of shape (..., 2) giving per-K scales [sx, sy].

    Returns:
        K_resized: Tensor of same shape as K, with
                   K[...,0,:] *= sx and K[...,1,:] *= sy.
    """
    # clone and ensure float dtype
    K_scaled = K.clone().to(dtype=torch.get_default_dtype())

    # Determine sx, sy tensor of shape (...,2)
    if isinstance(scale, (float, int)) or (torch.is_tensor(scale) and scale.ndim == 0):
        # uniform scalar
        s = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.item())
        batch_shape = K.shape[:-2]
        scales = torch.tensor([s, s], dtype=K_scaled.dtype, device=K_scaled.device)
        scales = scales.view(*([1] * len(batch_shape)), 2).expand(*batch_shape, 2)

    elif isinstance(scale, tuple):
        sx, sy = scale
        batch_shape = K.shape[:-2]
        scales = torch.tensor([sx, sy], dtype=K_scaled.dtype, device=K_scaled.device)
        scales = scales.view(*([1] * len(batch_shape)), 2).expand(*batch_shape, 2)

    else:
        # assume tensor of shape (...,2)
        scales = scale.to(dtype=K_scaled.dtype, device=K_scaled.device)
        if scales.ndim == 1 and scales.shape[0] == 2:
            # single [sx, sy]
            batch_shape = K.shape[:-2]
            scales = scales.view(*([1] * len(batch_shape)), 2).expand(*batch_shape, 2)

    # apply scaling: first row by sx, second row by sy
    K_scaled[..., 0, :] *= scales[..., 0].unsqueeze(-1)
    K_scaled[..., 1, :] *= scales[..., 1].unsqueeze(-1)

    return K_scaled


def load_data_map(
    filepath: Path,
    metadata: Dict = None,
    compression_mode: str = "png",
) -> np.ndarray:
    """
    Load a single map (depth, confidence, normals, or mask) from disk,
    supporting 'npz', 'npy', and 'png' modes, and return as a HxW float32 array.
    """
    mode = compression_mode
    suffix = filepath.suffix.lower()

    # 1) NPZ: pick the first array inside
    if mode == "npz" or suffix == ".npz":
        arr_dict = np.load(filepath)
        arr = next(iter(arr_dict.values()))
    # 2) NPY: raw array
    elif mode == "npy" or suffix == ".npy":
        arr = np.load(filepath)
    # 3) PNG: read uint image and dequantize
    else:  # png
        # lazily load metadata if not already (only works for nuscnes)
        if metadata is None:
            meta_path = Path(filepath.parent.parent.parent) / "depth_metadata.json"
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        raw = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not load image {filepath}")
        # depth maps were saved as uint16
        if raw.dtype == np.uint16:
            arr = raw.astype(np.float32) / 65535.0  # normalize to [0, 1]
        elif raw.dtype == np.uint8:
            arr = raw.astype(np.float32) / 255.0
        # denormalize if metadata is provided
        if "samples" in metadata:
            metadata = metadata["samples"][filepath.stem]
        min_val = metadata.get(f"{filepath.parent.name}_min", 0.0)
        max_val = metadata.get(f"{filepath.parent.name}_max", 1.0)
        arr = arr * (max_val - min_val) + min_val

    # finally, ensure correct shape (H,W) or (H,W,3)
    # resizing is done downstream if needed
    return arr.astype(np.float32), metadata


def convert_to_4x4(camera_batch: torch.Tensor):
    """
    Converts a batch of camera [R | t] matrices (B, 3, 4) into full 4x4 transformation matrices.

    Args:
        camera_batch (torch.Tensor): Tensor of shape (B, 3, 4) where each element is [R|t].

    Returns:
        torch.Tensor: Tensor of shape (B, 4, 4) with the bottom row set to [0, 0, 0, 1].
    """
    B = camera_batch.shape[0]
    # Create a batch of identity matrices of shape (B, 4, 4)
    transforms = torch.eye(4, device=camera_batch.device).unsqueeze(0).repeat(B, 1, 1)
    # Replace the top 3 rows with the camera matrices
    transforms[:, :3, :] = camera_batch
    return transforms


def get_samples_dict(
    data_mngr,
    device: Literal["cuda", "cpu"],
    info: dict = None,
    max_resolution: int = 128,
    downsample_factor: int = None,
) -> dict:
    """
    Collect a list of per-camera sample dicts with singular keys.
    Each dict contains the available fields for that sample:
      - cam2world: (4,4) np.float32
      - K: (3,3) np.float32 (scaled if downsampled)
      - dim: (2,) np.int32 [H, W]
      - idx: int
      - pose_type: str
      - generation: int
      - filename: str
      - img: (H,W,3|1) np.float32
      - depth, depth_conf: (H,W,1) np.float32 (if present)
      - normal, normal_conf: (H,W,3|1) np.float32 (if present)
      - bg_mask: (H,W,1) np.bool_ (if present)
      - feature: (H,W,3|C) np.float32 (if present; here we keep first 3 channels as before)
    """

    def _to_ch_first(x: torch.Tensor) -> torch.Tensor:
        """Normalize to (C, H, W)."""
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1)  # HWC -> CHW
        if x.ndim == 2:
            x = x.unsqueeze(0)  # HW -> 1HW
        return x

    cameras: Dict[str, Dict[str, Any]] = {}
    num_train = len(data_mngr.train_dataset)

    for i in range(num_train):
        cams, data = data_mngr.next_train(step=1)

        # --- inputs present? ---
        has_depth = (
            ("depth" in data)
            and ("depth_confidence" in data)
            and ("normals" in data)
            and ("normal_confidence" in data)
            and ("bg_mask" in data)
        )
        has_feats = "reduced_embeddings" in data

        # --- image ---
        img = _to_ch_first(data["image"])  # (C,H,W)

        # --- optional maps ---
        if has_depth:
            depth = _to_ch_first(data["depth"])
            dconf = _to_ch_first(data["depth_confidence"])
            norm = _to_ch_first(data["normals"])
            nconf = _to_ch_first(data["normal_confidence"])
            bgm = _to_ch_first(data["bg_mask"])

        if has_feats:
            feats = _to_ch_first(
                data["reduced_embeddings"][..., :3]
            )  # keep first 3 channels as before

        # --- compute downsample scale ---
        if downsample_factor is None:
            larger_res = max(img.shape[1], img.shape[2])
            scale_factor = (
                math.ceil(larger_res / max_resolution)
                if larger_res > max_resolution
                else 1
            )
        else:
            scale_factor = downsample_factor

        # --- resize (keep image bilinear, masks nearest) ---
        if scale_factor > 1:
            new_h = img.shape[1] // scale_factor
            new_w = img.shape[2] // scale_factor

            img = F.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            if has_depth:
                depth = F.interpolate(
                    depth.unsqueeze(0), size=(new_h, new_w), mode="nearest"
                ).squeeze(0)
                dconf = F.interpolate(
                    dconf.unsqueeze(0), size=(new_h, new_w), mode="nearest"
                ).squeeze(0)
                norm = F.interpolate(
                    norm.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                nconf = F.interpolate(
                    nconf.unsqueeze(0), size=(new_h, new_w), mode="nearest"
                ).squeeze(0)
                bgm = (
                    F.interpolate(
                        bgm.unsqueeze(0).float(), size=(new_h, new_w), mode="nearest"
                    )
                    .squeeze(0)
                    .bool()
                )

            if has_feats:
                feats = F.interpolate(
                    feats.unsqueeze(0), size=(new_h, new_w), mode="nearest"
                ).squeeze(0)
        else:
            # Ensure features match image size if not already
            if has_feats and (feats.shape[1:] != img.shape[1:]):
                feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=(img.shape[1], img.shape[2]),
                    mode="nearest",
                ).squeeze(0)

        # --- intrinsics (scaled if needed) ---
        K = torch.tensor(
            [[cams.fx, 0, cams.cx], [0, cams.fy, cams.cy], [0, 0, 1]],
            device=device,
            dtype=torch.float32,
        )
        if scale_factor > 1:
            s = float(scale_factor)
            K[0, 0] /= s
            K[1, 1] /= s
            K[0, 2] /= s
            K[1, 2] /= s

        # --- extrinsics ---
        cam2w = cams.camera_to_worlds
        cam2w = convert_to_4x4(cam2w)  # keep behavior consistent with the original

        # --- pose type / generation / filename / idx ---
        # Pose type: try per-sample; otherwise "unknown"
        if getattr(cams, "metadata", None):
            idx = int(data["image_idx"])
            cams_metadata = cams.metadata[idx]
            pose_type = cams_metadata.get("pose_type", "unknown")
            gen = int(cams_metadata.get("generation", 0))
            filename = str(
                cams_metadata.get("filename", f"unknown_{i}")
            )  # expected to be present
        else:
            pose_type = "unknown"
            gen = 0
            filename = f"unknown_{i}"
            idx = i

        H, W = img.shape[1], img.shape[2]

        # --- pack a per-camera dict (singular keys) ---
        cam_dict: Dict[str, Any] = {
            "cam2world": (
                cam2w[0].detach().cpu().numpy().astype("float32")
                if cam2w.ndim == 3
                else cam2w.detach().cpu().numpy().astype("float32")
            ),
            "K": K.detach().cpu().numpy().astype("float32"),
            "dim": torch.tensor([H, W]).cpu().numpy().astype("int32"),
            "idx": idx,
            "pose_type": pose_type,
            "generation": gen,
            "filename": filename,
            "img": img.permute(1, 2, 0).detach().cpu().numpy(),  # (H,W,C)
        }

        if has_depth:
            cam_dict["depth"] = depth.permute(1, 2, 0).detach().cpu().numpy()
            cam_dict["depth_conf"] = dconf.permute(1, 2, 0).detach().cpu().numpy()
            cam_dict["normal"] = norm.permute(1, 2, 0).detach().cpu().numpy()
            cam_dict["normal_conf"] = nconf.permute(1, 2, 0).detach().cpu().numpy()
            cam_dict["bg_mask"] = (
                bgm.permute(1, 2, 0).detach().cpu().numpy().astype(bool)
            )

        if has_feats:
            cam_dict["feature"] = feats.permute(1, 2, 0).detach().cpu().numpy()

        cameras[filename] = cam_dict

    if info is None:
        info = {}
    info["cameras"] = cameras
    return info


def store_samples(
    data_mngr: FullImageDatamanager,
    device: Literal["cuda", "cpu"],
    info: dict = None,
    path: str = None,
    downsample_factor: int = 8,
    compress: bool = False,
):
    """
    Extracts all training samples and stores them as one compressed .npz,
    with per-pixel maps in channel-last format (H, W, C or H, W, 1).
    NOTE: I might deprecate this and load stuff for the visualizer directly from the dataset or the datamanager.
    """
    if "cameras" not in info:
        info = get_samples_dict(
            data_mngr=data_mngr,
            device=device,
            info=info,
            scale_factor=downsample_factor,
        )
    arrays = info["cameras"]

    # --- prepare output ---
    os.makedirs(path, exist_ok=True)

    # save single compressed archive
    fname = os.path.join(path, "training_samples.npz")
    if compress:
        np.savez_compressed(fname, **arrays)
    else:
        np.savez(fname, **arrays)

    # ——— point-cloud outputs ———
    if "pointclouds" in info:
        pointcloud_dict = {}
        for key in info["pointclouds"]:
            obj = info[key]
            if isinstance(obj, torch.Tensor):
                # convert to numpy
                pointcloud_dict[key] = obj.cpu().numpy()
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, torch.Tensor):
                        item = item.cpu().numpy()
                    if isinstance(item, np.ndarray):
                        pointcloud_dict[f"{key}_{i}"] = item
        # save point clouds as npz
        pc_path = os.path.join(path, "pointclouds.npz")
        if compress:
            np.savez_compressed(pc_path, **pointcloud_dict)
        else:
            np.savez(pc_path, **pointcloud_dict)

    return info, path


def get_all_cam_params(
    data_mngr: FullImageDatamanager, device: Literal["cuda", "cpu"]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # extracts all the camera matrices from the DataManager
    camtoworlds = []
    Ks = []
    dims = []
    num_train_cams = len(data_mngr.train_dataset)
    for i in range(num_train_cams):
        cams, data = data_mngr.next_train(step=1)
        camtoworld = cams.camera_to_worlds
        camtoworlds.append(camtoworld)
        K = torch.tensor([[cams.fx, 0, cams.cx], [0, cams.fy, cams.cy], [0, 0, 1]]).to(
            device
        )
        Ks.append(K)
        dim = torch.tensor([cams.height, cams.width])
        dims.append(dim)
    camtoworlds = torch.cat(camtoworlds, dim=0)
    camtoworlds = convert_to_4x4(camtoworlds)
    Ks = torch.stack(Ks, dim=0)  # [n, 3, 3]
    dims = torch.stack(dims, dim=0)
    return camtoworlds, Ks, dims


def load_samples(path: Union[str, Path], device: torch.device = None) -> dict:
    """
    Loads training samples from compressed .npz archives.
    Returns a dict of torch tensors on `device` (or CPU if None).

    Keys: cam2worlds, Ks, dims, imgs, idx (+ depths, depth_confs, normals, normal_confs, bg_masks if present)
        Also loads init_points, rand_points, depthmap_points if files exist.
    """
    arrays = {}
    npz_file = os.path.join(path, "training_samples.npz")
    if os.path.exists(npz_file):
        arrays["cameras"] = {}
        try:
            with np.load(npz_file) as data:
                for key in data.files:
                    arr = data[key]
                    tensor = torch.from_numpy(arr)
                    if device is not None:
                        tensor = tensor.to(device)
                    arrays["cameras"][key] = tensor
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            return {}

    # Load point clouds if present
    npz_file = os.path.join(path, "pointclouds.npz")
    if os.path.exists(npz_file):
        arrays["pointclouds"] = {}
        try:
            with np.load(npz_file) as data:
                for key in data.files:
                    arr = data[key]
                    tensor = torch.from_numpy(arr)
                    if device is not None:
                        tensor = tensor.to(device)
                    arrays["pointclouds"][key] = tensor
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")

    return arrays


def load_TASE_samples(
    scene: Union[str, Path],
    dataset_path: Union[str, Path] = None,
    device: str = "cpu",
    dyn_bounding_boxes: bool = False,
    scene_transform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """
    Loads training samples from TASE's .npz archives.
    Returns a dict of torch tensors on specified device.
    This is meant for visualization purposes, not for training. Use a data manager for training.
    """

    from TASE.splatting.trainer.TASE_dataparser import TASE_DataParserConfig, TASE
    from TASE.splatting.trainer.full_images_datamanager import (
        FullImageDatamanager,
        FullImageDatamanagerConfig,
    )
    from TASE.splatting.trainer.TASE_dataset import TASE_dataset

    print(
        f"Loading TASE samples from scene '{scene}' in dataset path '{dataset_path}' for visualization"
    )

    # Check if full path has been provided
    scene = Path(scene)
    if scene.is_file():
        dataset_path = scene.parent.parent
        scene = scene.stem
    elif dataset_path is not None:
        dataset_path = Path(dataset_path)
        scene = str(scene)

    # Load the TASE dataset configuration
    parser_config = TASE_DataParserConfig(
        data_dir=dataset_path, metadata=scene, scene_transform=scene_transform
    )
    manager_config = FullImageDatamanagerConfig(dataparser=parser_config)
    manager = FullImageDatamanager[TASE_dataset](config=manager_config)
    parser = manager.dataparser

    info = get_samples_dict(data_mngr=manager, device=device)

    # load the pointclouds
    points, rgb, labels = parser.load_pointcloud()
    if points is not None:
        if "pointclouds" not in info:
            info["pointclouds"] = {}
        if rgb is not None:
            rgb = torch.from_numpy(rgb).float() / 255.0
            info["pointclouds"]["rgb_points"] = torch.cat(
                [torch.from_numpy(points).float(), rgb], dim=-1
            )
        else:
            info["pointclouds"]["points"] = torch.from_numpy(points).float()

    # load the dynamic bounding boxes
    if dyn_bounding_boxes:
        bbxs = parser.load_dynamic_bbxs()
        info["bounding_boxes"] = bbxs

    return info


def intrinsics_to_tensors(intrinsics: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts the intrinsics dictionary to tensors."""
    K = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ]
    )
    K = torch.tensor(K, dtype=torch.float32)
    dims = np.array([intrinsics["height"], intrinsics["width"]])
    dims = torch.tensor(dims, dtype=torch.int32)
    return K, dims


def voxelize_pointcloud(
    points: torch.Tensor,
    labels: torch.Tensor = None,
    voxel_grid_resolution: Tuple[int, int, int] = (200, 200, 30),
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Partition a point cloud into voxels.

    Parameters:
        points: (N, ≥3) tensor of (x,y,z,…) coordinates.
        voxel_grid_resolution: number of voxels along (x, y, z).

    Returns:
        voxel_to_point_indices: dict mapping voxel-index triples → list of point indices.
    """
    coords = points[:, :3]
    if points.shape[1] > 3 and labels is None:
        labels = points[:, 3]  # use 4th column as labels if available

    grid_min = coords.min(0)[0]
    grid_max = coords.max(0)[0]

    # voxel size per axis
    res = torch.tensor(voxel_grid_resolution, device=points.device, dtype=coords.dtype)
    voxel_size = (grid_max - grid_min) / (res - 1)

    # compute integer voxel indices
    shifted = coords - grid_min.unsqueeze(0)
    idx = (shifted / voxel_size).floor().long()  # (N,3)

    # group points by voxel
    vox2pts = defaultdict(list)
    for i, vidx in enumerate(idx.tolist()):
        vox2pts[tuple(vidx)].append(i)

    return dict(vox2pts)


def get_voxel_grid_tensor(
    points: torch.Tensor,
    labels: torch.Tensor = None,
    voxel_grid_resolution: Tuple[int, int, int] = (200, 200, 30),
) -> torch.Tensor:
    """
    Create a voxel grid tensor from a point cloud.
    Parameters:
        points: (N, ≥3) tensor of (x,y,z,…) coordinates.
        labels: (N,) tensor of labels for each point, optional.
        voxel_grid_resolution: number of voxels along (x, y, z).
    Returns:
        voxel_grid: (D, H, W, C) tensor where C=1+1 if labels are provided, else C = 1.
    """
    voxel_to_point_indices, voxel_to_labels = voxelize_pointcloud(
        points, labels, voxel_grid_resolution
    )
    # Create an empty voxel grid tensor
    voxel_grid_shape = (
        voxel_grid_resolution[2],
        voxel_grid_resolution[1],
        voxel_grid_resolution[0],
    )
    num_channels = 1 + (1 if voxel_to_labels is not None else 0)
    voxel_grid = torch.zeros(
        voxel_grid_shape + (num_channels,), dtype=torch.float32, device=points.device
    )
    # Fill the voxel grid with point indices and labels
    for voxel_idx, point_indices in voxel_to_point_indices.items():
        z, y, x = voxel_idx
        if voxel_to_labels is not None:
            # If labels are provided, fill both point indices and labels
            voxel_grid[z, y, x, 0] = len(point_indices)
            # Get the majority vote for the label in this voxel
            label_counts = torch.tensor(
                voxel_to_labels[voxel_idx], dtype=torch.float32, device=points.device
            )
            if label_counts.numel() > 0:
                # Count occurrences of each label
                unique_labels, counts = label_counts.unique(return_counts=True)
                # Get the most common label
                most_common_label = unique_labels[counts.argmax()]
                voxel_grid[z, y, x, 1] = most_common_label
            else:
                # If no labels, set to -1
                voxel_grid[z, y, x, 1] = -1
        else:
            # If no labels, just fill the point indices count
            voxel_grid[z, y, x, 0] = len(point_indices)
    return voxel_grid


def sample_from_voxels(
    voxel_to_point_indices: Dict[Tuple[int, int, int], List[int]],
    n: int,
    occupancy_threshold: int = 5,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Evenly sample up to `n` point-indices from occupied voxels.

    Parameters:
        voxel_to_point_indices: output of `voxelize_pointcloud`.
        n: total number of indices to sample.
        occupancy_threshold: min number of points for a voxel to be considered.
        verbose: whether to print stats.
        device: device for the returned tensor.

    Returns:
        sampled_indices: 1D tensor of length ≲ n (may be >n if voxels × per-voxel > n)
    """
    # filter occupied voxels
    occupied = [
        k for k, v in voxel_to_point_indices.items() if len(v) >= occupancy_threshold
    ]

    if verbose:
        print(f"Voxels total: {len(voxel_to_point_indices)}, occupied: {len(occupied)}")

    if not occupied:
        if verbose:
            print("No voxel meets the occupancy threshold.")
        return torch.empty(0, dtype=torch.long, device=device)

    samples_per_voxel = max(1, n // len(occupied))
    if verbose:
        print(
            f"Sampling {samples_per_voxel} pts per occupied voxel → target ~{samples_per_voxel*len(occupied)} pts"
        )

    all_samples: List[int] = []
    for key in occupied:
        pts = voxel_to_point_indices[key]
        if len(pts) >= samples_per_voxel:
            chosen = np.random.choice(pts, size=samples_per_voxel, replace=False)
        else:
            chosen = np.random.choice(pts, size=samples_per_voxel, replace=True)
        all_samples.extend(chosen.tolist())

    sampled_indices = torch.tensor(all_samples, dtype=torch.long, device=device)
    if verbose:
        print(f"Returned {sampled_indices.numel()} indices.")
    return sampled_indices


def voxelize_and_sample_pointcloud(
    points: torch.Tensor,
    n: int,
    voxel_grid_resolution: Tuple[int, int, int] = (200, 200, 30),
    occupancy_threshold: int = 5,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Voxelize a point cloud and sample `n` points from occupied voxels.

    Parameters:
        points: (N, ≥3) tensor of (x,y,z,…) coordinates.
        n: total number of indices to sample.
        voxel_grid_resolution: number of voxels along (x, y, z).
        occupancy_threshold: min number of points for a voxel to be considered.
        verbose: whether to print stats.
        device: device for the returned tensor.

    Returns:
        sampled_indices: 1D tensor of length ≲ n (may be >n if voxels × per-voxel > n)
    """
    vox2pts = voxelize_pointcloud(points, voxel_grid_resolution)
    return sample_from_voxels(vox2pts, n, occupancy_threshold, verbose, device)


def generate_depthmap_pointcloud(
    dataparser_outputs,
    downscale_factor: int = 8,
    conf_threshold: float = 0.9,
    point_for_cam_pos: bool = False,
) -> List[np.ndarray]:
    """
    Generate a list of point clouds (one per image) from per-pixel depth maps.
    Requires:
      - dataparser_outputs.image_filenames: List[Path]
      - dataparser_outputs.cameras.height[i], width[i]
      - dataparser_outputs.dataparser_scale: float or array-like scalar
      - dataparser_outputs.metadata["depth_filenames"][i], etc.
      - dataparser_outputs.mask_filenames[i]
      - dataparser_outputs.cameras.camera_to_worlds[i]: 3×4 numpy or torch

    Parameters:
        dataparser_outputs: Object (or dataclass) with the following attributes:
            - image_filenames: List[Path] to RGB images
            - cameras: object with attributes
                • height: sequence of ints
                • width: sequence of ints
                • fx, fy, cx, cy: sequences or tensors of intrinsics per image
                • camera_to_worlds: sequence of 3×4 poses (numpy or torch)
            - dataparser_scale: scalar to multiply raw depth by
            - metadata: dict with keys:
                “depth_filenames”, “depth_confidence_filenames”,
                “bg_mask_filenames”, “normal_filenames”, “normal_confidence_filenames”
            - mask_filenames: List[Path] to dynamic-object masks

        downscale_factor (int): Factor by which to downsample height/width.
        conf_threshold (float): Minimum confidence to accept a depth pixel.
        point_for_cam_pos (bool): If True, also add a point at the camera origin.

    Returns:
        List[np.ndarray]: Each element is an array of shape (M, 9), where each row is
                          [x, y, z, r, g, b, nx, ny, nz]. Coordinates are in world frame.
    """
    outputs_list: List[np.ndarray] = []

    num_images = len(dataparser_outputs.image_filenames)
    # load the metadtata for depth scaling
    meta = json.load(open(dataparser_outputs.metadata["scene_metadata_filename"], "r"))

    for i in range(num_images):
        img_path = dataparser_outputs.image_filenames[i]
        height = int(dataparser_outputs.cameras.height[i])
        width = int(dataparser_outputs.cameras.width[i])
        scale = dataparser_outputs.dataparser_scale

        # 1. Load depth & related maps via helper (handles npz/npy/png)
        if "datamap_filenames" in dataparser_outputs.metadata:
            datamap = np.load(
                dataparser_outputs.metadata["datamap_filenames"][i], allow_pickle=True
            )
            depth_map = datamap["depth"]
            conf_map = datamap["depth_confidence"]
            bg_mask_map = datamap["bg_mask"]
            normal_map = datamap["normals"]
            dyn_mask = datamap["dyn_mask"]
        else:
            depth_map, depth_meta = load_data_map(
                Path(dataparser_outputs.metadata["depth_filenames"][i]), meta
            )
            conf_map, _ = load_data_map(
                Path(dataparser_outputs.metadata["depth_confidence_filenames"][i]),
                depth_meta,
            )
            bg_mask_map, _ = load_data_map(
                Path(dataparser_outputs.metadata["bg_mask_filenames"][i]), depth_meta
            )
            normal_map, _ = load_data_map(
                Path(dataparser_outputs.metadata["normal_filenames"][i]), depth_meta
            )
            dyn_mask = cv2.imread(
                str(dataparser_outputs.mask_filenames[i]), cv2.IMREAD_GRAYSCALE
            )

        # 2. Load RGB image and normalize to [0,1]
        bgr_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 3. Downsample spatially
        th = height // downscale_factor
        tw = width // downscale_factor
        depth_map = cv2.resize(depth_map, (tw, th), interpolation=cv2.INTER_NEAREST)
        conf_map = cv2.resize(conf_map, (tw, th), interpolation=cv2.INTER_NEAREST)
        bg_mask_map = cv2.resize(
            bg_mask_map, (tw, th), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        rgb_img = cv2.resize(rgb_img, (tw, th), interpolation=cv2.INTER_LINEAR)
        normal_map = cv2.resize(normal_map, (tw, th), interpolation=cv2.INTER_LINEAR)
        dyn_mask = cv2.resize(
            dyn_mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 4. Build intrinsic matrix at downsampled resolution
        fx = float(dataparser_outputs.cameras.fx[i]) / downscale_factor
        fy = float(dataparser_outputs.cameras.fy[i]) / downscale_factor
        cx = float(dataparser_outputs.cameras.cx[i]) / downscale_factor
        cy = float(dataparser_outputs.cameras.cy[i]) / downscale_factor
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # 5. Create pixel grid
        ys, xs = np.meshgrid(np.arange(th), np.arange(tw), indexing="ij")
        ones = np.ones_like(xs)
        pixel_hom = np.stack((xs, ys, ones), axis=0).reshape(3, -1)  # (3, th*tw)

        # 6. Backproject to camera space
        depth_flat = depth_map.flatten()
        cam_pts = (np.linalg.inv(K) @ pixel_hom) * depth_flat  # (3, th*tw)

        valid_mask = (
            (depth_flat > 0)
            & (conf_map.flatten() > conf_threshold)
            & bg_mask_map.flatten()
            & dyn_mask.flatten()
        )
        cam_pts = cam_pts[:, valid_mask]  # (3, M)
        colors = rgb_img.reshape(-1, 3)[valid_mask]  # (M,3)
        normals = normal_map.reshape(-1, 3)[valid_mask]  # (M,3)

        # 7. Optionally add camera origin as a point
        if point_for_cam_pos:
            cam_pts = np.concatenate([cam_pts, np.zeros((3, 1))], axis=1)
            colors = np.vstack([colors, [0.0, 0.0, 0.0]])
            normals = np.vstack([normals, [0.0, 0.0, 1.0]])

        # 8. Transform to world coordinates
        #    dataparser_outputs.cameras.camera_to_worlds[i] is a 3×4 pose: [R | t]
        pose_3x4 = dataparser_outputs.cameras.camera_to_worlds[i]
        if isinstance(pose_3x4, torch.Tensor):
            pose_np = pose_3x4.cpu().numpy()
        else:
            pose_np = np.array(pose_3x4, dtype=np.float32)
        # Append homogeneous row
        ones_row = np.ones((1, cam_pts.shape[1]), dtype=cam_pts.dtype)
        cam_pts_hom = np.concatenate([cam_pts, ones_row], axis=0)  # (4, M)
        world_pts_hom = pose_np @ cam_pts_hom  # (3, M)
        world_pts = world_pts_hom.T  # (M,3)

        # 9. Rotate normals into world frame (R @ normals.T).T
        R = pose_np[:3, :3]
        norms_world = (R @ normals.T).T  # (M,3)

        # 10. Stack (x,y,z, r,g,b, nx,ny,nz)
        combined = np.hstack([world_pts, colors, norms_world])  # (M,9)
        outputs_list.append(combined)

    return outputs_list


def exponential_decay_weights(
    n: int,
    max_val: float,
    min_val: float,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Create a length-n vector that starts at max_val and decays
    exponentially to min_val at index n-1.

    Args:
        n       (int): number of weights, must be >= 2
        max_val (float): weight at i=0
        min_val (float): weight at i=n-1
        device:   torch device or string
        dtype:    torch dtype (defaults to torch.get_default_dtype())

    Returns:
        Tensor of shape (n,) with
            w[i] = max_val * exp(-λ * i),
        where λ = ln(max_val/min_val) / (n-1).
    """
    assert n >= 2, "n must be >= 2"
    dtype = dtype or torch.get_default_dtype()
    # compute decay rate λ so that w[n-1] = min_val
    lam = math.log(max_val / min_val) / (n - 1)
    # indices 0 … n-1
    idx = torch.arange(n, device=device, dtype=dtype)
    # exponential decay
    w = max_val * torch.exp(-lam * idx)
    return w


def gaussian_decay_weights(
    n, max_val=1.0, min_val=0.05, sigma_ratio=0.1, device="cpu", dtype=None
):
    # sigma_ratio is relative to length: smaller → sharper drop
    dtype = dtype or torch.get_default_dtype()
    sigma = sigma_ratio * (n - 1)
    idx = torch.arange(n, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (idx / sigma) ** 2)
    g = (g - g.min()) / (g.max() - g.min())  # normalize to [0,1]
    return min_val + (max_val - min_val) * g


def load_cameras_json(
    path: Union[str, Path],
    pose_type: str = "external",
    generation: int = 0,
) -> Dict[str, Dict]:
    """Load a cameras.json file (3DGS / COLMAP convention) into the TASE viewer's
    per-camera dict format.

    The cameras.json schema expected per entry:
        id        – integer camera id
        img_name  – source image filename stem
        width, height – image dimensions in pixels
        position  – [x, y, z] camera center in world space
        rotation  – 3x3 world-to-camera rotation matrix (row-major)
        fx, fy    – focal lengths in pixels

    Args:
        path:       Path to the cameras.json file.
        pose_type:  Label stored in each camera's ``pose_type`` field.
        generation: Generation index used for colour-coding in the viewer.

    Returns:
        A ``cameras`` dict mapping string keys to per-camera dicts compatible
        with :func:`get_samples_dict` output (keys: cam2world, K, dim, idx,
        pose_type, generation, filename).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cameras.json not found: {path}")

    with open(path, "r") as f:
        entries = json.load(f)

    cameras: Dict[str, Dict] = {}
    for entry in entries:
        cam_id = int(entry["id"])
        img_name = str(entry.get("img_name", str(cam_id)))
        width = int(entry["width"])
        height = int(entry["height"])
        fx = float(entry["fx"])
        fy = float(entry["fy"])
        position = np.array(entry["position"], dtype=np.float32)  # camera center
        R_w2c = np.array(entry["rotation"], dtype=np.float32)     # world-to-camera rotation

        # Build cam-to-world: C2W rotation = R_w2c^T, translation = camera center
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = position

        K = np.array(
            [[fx, 0.0, width / 2.0],
             [0.0, fy, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        dim = np.array([height, width], dtype=np.int32)

        key = f"{cam_id:06d}_{img_name}"
        cameras[key] = {
            "cam2world": c2w,
            "K": K,
            "dim": dim,
            "idx": cam_id,
            "pose_type": pose_type,
            "generation": generation,
            "filename": img_name,
        }

    return cameras


if __name__ == "__main__":
    # Example usage
    print(gaussian_decay_weights(128))
