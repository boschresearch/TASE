# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from nerfstudio
#   (https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nuscenes_dataparser.py)
# Copyright 2022 the Regents of the University of California, Nerfstudio Team
# and contributors. All rights reserved., licensed under the Apache-2.0
# license, cf. 3rd-party-licenses.txt file in the root directory of this
# source tree.

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type, List
from collections import defaultdict
import json

import numpy as np
import pyquaternion
import torch
import cv2
from tqdm import tqdm
import argparse
import shutil
import datetime

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from TASE.splatting.util.data_util import *

"""
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer
)
from pytorch3d.structures import Pointclouds

"""


@dataclass
class TASE_DataParserConfig(DataParserConfig):
    # Target class for the dataparser config
    _target: Type = field(default_factory=lambda: TASE)
    # Root path of the dataset
    data_dir: Path = Path("DATA/TASE_Gen")
    # Name of the scene
    metadata: str = "scene_0000.json"
    # Name of the new metadata file to be created, if any
    new_metadata: Optional[str] = None
    # List of the active genrations of samples, None means all
    active_sample_gens: List[str] = None
    # Rotation (wxyz quaternion) and translation (3,) to apply to the scene
    scene_transform: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class TASE(DataParser):
    """
    TASE Data Parser.

    This can be used for 3D Scene Editing - supports both reconstruction with integrated TASE semantics and generation of new samples with TASE semantics. The dataparser can be used to load the dataset, generate new samples, and save the metadata.

    """

    config: TASE_DataParserConfig
    cam_rot_trans: List[Tuple[np.ndarray, np.ndarray]] = field(init=False)
    cam_poses: torch.Tensor = field(init=False)
    cam_Ks: torch.Tensor = field(init=False)
    cam_dims: torch.Tensor = field(init=False)

    def __init__(self, config: TASE_DataParserConfig):
        self.config = config
        self.metadata = None
        if config.metadata is not None and config.data_dir is not None:
            meta_name = (
                config.metadata
                if ".json" in str(config.metadata)
                else str(config.metadata) + ".json"
            )
            meta_path = Path(config.data_dir) / "metadata" / meta_name
            if meta_path.is_file():
                self.metadata = json.load(open(meta_path))
        if not self.metadata:
            print(f"WARNING: Metadata file {config.metadata} does not exist.")
            self.metadata = {}
        self.datadir = config.data_dir
        self.active_gens = config.active_sample_gens
        self.gen_cnt = 0
        self.version_cnt = 0
        if config.new_metadata is not None:
            self.new_metadata = config.new_metadata
        else:
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.new_metadata = f"{self.metadata.get("name", "unknown")}_{datetime_str}"

    @staticmethod
    def compute_projection_matrix(
        *,
        znear: float,
        zfar: float,
        width: int,
        height: int,
        fx: torch.Tensor = None,
        fy: torch.Tensor = None,
        fov_x: torch.Tensor = None,
        fov_y: torch.Tensor = None,
        cx: torch.Tensor = None,
        cy: torch.Tensor = None,
        degrees: bool = True,
        device: str = "cpu",
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Build a batch of 4×4 projection matrices P (shape (B,4,4)) that map
        camera‐space points into NDC ([-1,1]^3), using either focal lengths
        or fields‐of‐view (or both).
        NOTE: Depricated

        Args:
            znear, zfar: near/far clip planes (scalars).
            width, height: image size in pixels.
            fx, fy: optional float tensor of shape (B,) giving focal lengths in px.
            fov_x, fov_y: optional float tensor of shape (B,) giving HFOV/VFOV.
            cx, cy: optional float tensor of shape (B,) giving principal points in px.
                    Defaults to center of image if None.
            degrees: interpret fov_x/fov_y as degrees if True.
            device: “cpu” or “cuda”.
            eps: tolerance for checking consistency if both fx and fov_x are given.

        Returns:
            P of shape (B, 4, 4).
        """

        # Determine batch size B
        # We require at least one of (fx,fov_x) and one of (fy,fov_y) to be provided
        # and batch‐shapable.
        def _ensure_tensor(x, name):
            if x is None:
                return None
            t = torch.as_tensor(x, device=device)
            return t

        fx = _ensure_tensor(fx, "fx")
        fy = _ensure_tensor(fy, "fy")
        fov_x = _ensure_tensor(fov_x, "fov_x")
        fov_y = _ensure_tensor(fov_y, "fov_y")
        cx = (
            _ensure_tensor(cx, "cx")
            if cx is not None
            else torch.full((1,), width / 2, device=device)
        )
        cy = (
            _ensure_tensor(cy, "cy")
            if cy is not None
            else torch.full((1,), height / 2, device=device)
        )

        # Infer batch size
        candidates = [t for t in (fx, fy, fov_x, fov_y, cx, cy) if t is not None]
        if not candidates:
            raise ValueError("Must provide at least one of fx/fov_x, fy/fov_y, etc.")
        B = candidates[0].shape[0]
        for t in candidates:
            if t.shape[0] != B:
                raise ValueError("All inputs must have same batch length")

        # If fov in degrees, convert to radians
        if degrees:
            if fov_x is not None:
                fov_x = fov_x * (math.pi / 180)
            if fov_y is not None:
                fov_y = fov_y * (math.pi / 180)

        # Derive fx,fy from fov if needed
        # fx = znear / tan(fov_x/2)
        if fx is None and fov_x is not None:
            fx = znear / torch.tan(fov_x / 2)
        if fy is None and fov_y is not None:
            fy = znear / torch.tan(fov_y / 2)

        # If both provided, sanity‐check
        if fx is not None and fov_x is not None:
            fx_from_fov = znear / torch.tan(fov_x / 2)
            diff = (fx - fx_from_fov).abs()
            if torch.any(diff > eps):
                print(
                    f"fx ({fx.tolist()}) and fov_x→fx ({fx_from_fov.tolist()}) differ by >{eps}"
                )
        if fy is not None and fov_y is not None:
            fy_from_fov = znear / torch.tan(fov_y / 2)
            diff = (fy - fy_from_fov).abs()
            if torch.any(diff > eps):
                print(
                    f"fy ({fy.tolist()}) and fov_y→fy ({fy_from_fov.tolist()}) differ by >{eps}"
                )

        # Expand cx, cy to batch if needed
        if cx.shape[0] != B:
            cx = cx.expand(B)
        if cy.shape[0] != B:
            cy = cy.expand(B)

        # Now build P
        P = torch.zeros((B, 4, 4), device=device, dtype=torch.float32)

        # Scale x and y to NDC
        P[:, 0, 0] = 2.0 * fx / width
        P[:, 1, 1] = 2.0 * fy / height

        # Shift principal point into NDC
        P[:, 0, 2] = (width - 2.0 * cx) / width
        P[:, 1, 2] = (2.0 * cy - height) / height

        # Depth mapping (camera‐space z → [0,1] at znear/zfar)
        P[:, 2, 2] = zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        # Homogeneous row for right‐handed OpenGL convention
        P[:, 3, 2] = 1.0

        return P

    @staticmethod
    def render_semantic_map(
        points: np.ndarray,  # (N_pts, 3) world-space coords
        cls: np.ndarray,  # (N_pts,) integer labels
        cameras,  # Has .fx, .fy, .cx, .cy, .height, .width, .camera_to_worlds (B×4×4) -> Cameras from _generate_dataparser_outputs
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Render a batch of semantic maps from one point cloud under B cameras.
        Returns a tensor of shape (B, H, W, 1) with the per-pixel class index.
        NOTE: This doesn't work yet and I am probably going to change to rendering the gaussians instead.
        """
        device = torch.device(device)

        # 1) Build camera intrinsics tensor K and compute world→camera extrinsics
        B = cameras.camera_to_worlds.shape[0]
        R_c2w = cameras.camera_to_worlds[:, :3, :3].to(device)  # (B,3,3)
        t_c2w = cameras.camera_to_worlds[:, :3, 3].to(device)  # (B,3)

        # invert to get R_w2c, T_w2c for FoVPerspectiveCameras
        R_w2c = R_c2w.transpose(1, 2)  # (B,3,3)
        t_w2c = -(R_w2c @ t_c2w.unsqueeze(-1)).squeeze(-1)  # (B,3)

        widths = cameras.width.to(device).squeeze(-1)  # (B,)
        heights = cameras.height.to(device).squeeze(-1)  # (B,)
        # Check heights
        unique_h = torch.unique(heights)
        if unique_h.numel() > 1:
            print(
                f"Found multiple camera heights: {unique_h.tolist()}. "
                f"Proceeding by using height={unique_h[0].item()} for all."
            )
        # Check widths
        unique_w = torch.unique(widths)
        if unique_w.numel() > 1:
            print(
                f"Found multiple camera widths: {unique_w.tolist()}. "
                f"Proceeding by using width={unique_w[0].item()} for all."
            )

        # Then pick the “global” size from the first element:
        height = float(heights[0].item())
        width = float(widths[0].item())

        # assume cameras.fx, fy, cx, cy are all torch.Tensor of shape (B,1)
        fx = cameras.fx.to(device).squeeze(-1)  # (B,)
        fy = cameras.fy.to(device).squeeze(-1)  # (B,)
        cx = cameras.cx.to(device).squeeze(-1)  # (B,)
        cy = cameras.cy.to(device).squeeze(-1)  # (B,)

        # your near/far planes
        znear = 0.1
        zfar = 1000.0

        # Number of cameras
        B = fx.shape[0]

        Ks = TASE.compute_projection_matrix(
            znear=znear,
            zfar=zfar,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            device=device,
        )

        cams = PerspectiveCameras(device=device, R=R_w2c, T=t_w2c, K=Ks)

        # 2) Convert point cloud + classes into a batched Pointclouds
        pts = torch.from_numpy(points).to(device).float()  # (N,3)
        cls_t = torch.from_numpy(cls).to(device).long()  # (N,)

        # expand to (B, N, 3) and (B, N, 1)
        pts_batch = pts.unsqueeze(0).expand(B, -1, -1)  # (B,N,3)
        cls_feat = cls_t.unsqueeze(-1).float()  # (N,1)
        feats_batch = cls_feat.unsqueeze(0).expand(B, -1, -1)  # (B,N,1)

        pc_batch = Pointclouds(points=pts_batch, features=feats_batch)

        # 3) Rasterize + composite
        rast_settings = PointsRasterizationSettings(
            image_size=(int(height), int(width)),  # (H,W)
            radius=0.1,
            points_per_pixel=1,  # only one point per pixel (opacity = 1.0)
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cams, raster_settings=rast_settings),
            compositor=AlphaCompositor(background_color=(0.0,)),
        )
        # point_cloud is a Pointclouds(...) object
        images = renderer(pc_batch)  # → (N, H, W, C)

        return images

    @staticmethod
    def _load_pointcloud(
        path: Union[str, Path],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = Path(path)
        if not path.is_file():
            print(
                f"Point cloud file {path} does not exist. Skipping loading point cloud."
            )
            return None
        pc_data = np.load(path, allow_pickle=True)

        if "xyz" in pc_data:
            points = pc_data["xyz"]
            rgb = pc_data["rgb"] if "rgb" in pc_data else None
            cls = pc_data["class"] if "class" in pc_data else None
            return points, rgb, cls

        if "arr_0" in pc_data:
            pc_data = pc_data["arr_0"]
        elif "points" in pc_data:
            pc_data = pc_data["points"]
        else:
            print("No point cloud data found in the file.")
            print(f"Files: {pc_data.files}")
            return None
        cls, rgb = None, None
        points = pc_data[:, :3]
        if pc_data.shape[1] == 4:
            cls = pc_data[:, 3]
        if pc_data.shape[1] == 6:
            rgb = pc_data[:, 3:6]
            rgb = (rgb * 255).astype(np.uint8)
        if pc_data.shape[1] >= 7:
            rgb = pc_data[:, 3:6]
            rgb = (rgb * 255).astype(np.uint8)
            cls = pc_data[:, 6]
        if cls:
            cls = cls.astype(np.int32)  # Ensure class labels are integers
        return points, rgb, cls

    def load_pointcloud(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Loads the semantic point cloud data from the dataset directory."""
        path = (
            self.datadir
            / self.metadata["subdirs"]["pointcloud_dir"]
            / (self.metadata["name"] + ".npz")
        )
        if not path.is_file():
            print(
                f"Semantic point cloud file {path} does not exist. Skipping loading point cloud."
            )
            return None
        points, rgb, cls = self._load_pointcloud(path)
        if self.config.scene_transform is not None:
            rot, tran = self.config.scene_transform
            if rot is not None and tran is not None:
                rot = pyquaternion.Quaternion(rot)
                points = (rot.rotation_matrix @ points.T).T - tran
        return points, rgb, cls

    def load_dynamic_bbxs(self) -> Optional[List[torch.Tensor]]:
        """Loads the dynamic bounding boxes from the dataset directory."""
        path = (
            self.datadir
            / self.metadata["subdirs"]["bbxs_dir"]
            / (self.metadata["name"] + ".npz")
        )
        if not path.is_file():
            print(
                f"Dynamic bounding boxes file {path} does not exist. Skipping loading dynamic bounding boxes."
            )
            return None
        bbx_data = np.load(path, allow_pickle=True)
        ret = {}
        for key in bbx_data.files:
            ret[key] = bbx_data[key]
        return ret

    def get_poses(
        self, split="train"
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Loads the camera poses from the dataset directory."""

        if split == "val":
            samples = self.metadata.get("samples_val", None)
        elif split == "test":
            samples = self.metadata.get("samples_test", None)
        elif split == "train":
            samples = self.metadata.get("samples_train", None)

        if not samples:
            samples = self.metadata.get("samples", {})
            print(f"No split-specific samples found for split '{split}'. Using 'samples' key with {len(samples)} samples.")

        poses = []
        rot_trans = []
        for key, sample in samples.items():
            if "rotation_wxyz" in sample and "translation" in sample:
                rot = np.array(sample["rotation_wxyz"])
                tran = np.array(sample["translation"])
                # apply the scene transform if given
                if self.config.scene_transform is not None:
                    scene_rot, scene_tran = self.config.scene_transform
                    scene_rot = pyquaternion.Quaternion(scene_rot)
                    rot = pyquaternion.Quaternion(rot)
                    new_rot = scene_rot * rot
                    new_tran = scene_rot.rotate(tran) - scene_tran
                    rot, tran = new_rot.elements, new_tran
                rot_trans.append((rot, tran))
                pose = rotation_translation_to_pose(rot, tran)
                poses.append(pose)
        self.cam_rot_trans = rot_trans
        self.cam_poses = (
            np.stack(poses) if poses else np.eye(4)[None, ...]
        )  # Ensure at least one pose
        return self.cam_poses, self.cam_rot_trans

    def _generate_dataparser_outputs(
        self, split="train", ignore_missing=True, generate_datamaps=True
    ) -> DataparserOutputs:
        """Generates the dataparser outputs for the given split."""
        image_filenames = []
        datamap_filenames = []

        intrinsics = {
            "fx": [],
            "fy": [],
            "cx": [],
            "cy": [],
            "width": [],
            "height": [],
        }
        poses = []
        rot_trans = []
        cam_Ks = []
        cam_dims = []

        pose_types = []
        gens = []
        idxs = []

        self.depth_available = True
        samples = None
        if split == "val":
            samples = self.metadata.get("samples_val", None)
        elif split == "test":
            samples = self.metadata.get("samples_test", None)
        elif split == "train":
            samples = self.metadata.get("samples_train", None)

        if not samples:
            samples = self.metadata.get("samples", {})

        image_dir = Path(self.datadir) / self.metadata["subdirs"]["image_dir"]
        datamap_dir = Path(self.datadir) / self.metadata["subdirs"]["datamap_dir"]

        metadata = {}
        new_idx = -1

        for idx, (key, sample) in enumerate(samples.items()):
            # skip inactive gens
            if (
                self.active_gens is not None
                and sample.get("generation", 0) not in self.active_gens
            ):
                continue

            # --- filenames ---
            img_filename = (image_dir / sample["name"]).with_suffix(".jpg")
            if not img_filename.is_file():
                img_filename = img_filename.with_suffix(".png")
            if not img_filename.is_file() and not ignore_missing:
                print(f"Image file {img_filename} does not exist. Using dummy image.")
                img_filename = image_dir / "dummy_image.jpg"
            image_filenames.append(img_filename)
            new_idx += 1

            datamap_filename = (datamap_dir / sample["name"]).with_suffix(".npz")
            if not datamap_filename.is_file():
                if not ignore_missing:
                    print(f"Datamap file {datamap_filename} does not exist")
                    if generate_datamaps:
                        print(
                            "Automatic sample generation is not implemented yet. Using dummy datamap."
                        )
                        datamap_filename = datamap_dir / "dummy_datamap.npz"
                    else:
                        datamap_filename = None
            datamap_filenames.append(datamap_filename)

            # --- intrinsics ---
            intrinsics["cx"].append(sample["intrinsics"]["cx"])
            intrinsics["cy"].append(sample["intrinsics"]["cy"])
            intrinsics["fx"].append(sample["intrinsics"]["fx"])
            intrinsics["fy"].append(sample["intrinsics"]["fy"])
            intrinsics["width"].append(sample["intrinsics"]["width"])
            intrinsics["height"].append(sample["intrinsics"]["height"])

            # --- pose ---
            rot_tran = (
                np.array(sample["rotation_wxyz"]),
                np.array(sample["translation"]),
            )

            # Apply scene transform if given
            if self.config.scene_transform is not None:
                scene_rot, scene_tran = self.config.scene_transform
                if scene_rot is not None and scene_tran is not None:
                    scene_rot = pyquaternion.Quaternion(scene_rot)
                    rot, tran = rot_tran
                    rot = pyquaternion.Quaternion(rot)
                    new_rot = scene_rot * rot
                    new_tran = scene_rot.rotate(tran) - scene_tran
                    rot_tran = (new_rot.elements, new_tran)

            rot_trans.append(rot_tran)
            poses.append(rotation_translation_to_pose(rot_tran[0], rot_tran[1]))

            # --- tensors ---
            K, dims = intrinsics_to_tensors(sample["intrinsics"])
            cam_Ks.append(K)
            cam_dims.append(dims)

            # --- metadata ---
            meta = {
                "pose_type": sample.get("pose_type", "unknown"),
                "generation": sample.get("generation", 0),
                "filename": img_filename.name,
                "datamap_filename": datamap_filename.name if datamap_filename else None,
            }
            metadata[new_idx] = meta

        if not ignore_missing and len(poses) == 0:
            print("Info: Empty dataset. No images found.")
            print("Adding in a dummy camera to prevent errors.")
            img_filename = image_dir / "dummy_image.jpg"
            datamap_filenames.append(datamap_dir / "dummy_datamap.npz")
            image_filenames.append(img_filename)
            intrinsics["fx"].append(1000)
            intrinsics["fy"].append(1000)
            intrinsics["cx"].append(320)
            intrinsics["cy"].append(240)
            intrinsics["width"].append(640)
            intrinsics["height"].append(480)
            poses.append(np.eye(4))
            cam_Ks.append(
                torch.tensor(
                    [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=torch.float32
                )
            )
            cam_dims.append(torch.tensor([640, 480]))
            pose_types.append("dummy")
            gens.append(0)
            idxs.append(0)

        if split == "train":
            self.cam_poses = torch.from_numpy(np.stack(poses)).float()
            self.cam_rot_trans = rot_trans
            self.cam_Ks = torch.stack(cam_Ks)
            self.cam_dims = torch.stack(cam_dims)

        cameras = Cameras(
            fx=torch.tensor(intrinsics["fx"]).float(),
            fy=torch.tensor(intrinsics["fy"]).float(),
            cx=torch.tensor(intrinsics["cx"]).float(),
            cy=torch.tensor(intrinsics["cy"]).float(),
            width=torch.tensor(intrinsics["width"]).int(),
            height=torch.tensor(intrinsics["height"]).int(),
            camera_to_worlds=torch.from_numpy(np.stack(poses)).float()[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            metadata=metadata,
        )

        translations = self.cam_poses[:, :3, 3]
        scene_min = translations.min(dim=0).values
        scene_max = translations.max(dim=0).values
        self.scene_box = torch.stack([scene_min, scene_max])

        dataparser_transform = torch.eye(4)

        meta_filename = (
            self.metadata["name"]
            if ".json" in self.metadata["name"]
            else self.metadata["name"] + ".json"
        )
        metadata.update(
            {
                "scene_metadata_filename": Path(self.datadir)
                / self.metadata["subdirs"]["metadata_dir"]
                / meta_filename,
                "data_dir": Path(self.datadir),
            }
        )

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=self.scene_box,
            dataparser_scale=1.0,
            dataparser_transform=dataparser_transform,
            metadata=metadata,
        )

    def add_sample(
        self,
        sample: dict,
        idx: int,
        datadir: Path,
        split: str = "train",
    ) -> str:
        """
        Add a single sample:
        - Generate a unique name using current self.gen_cnt and idx
        - Save image and datamap (if provided)
        - Update self.metadata["samples"]
        Returns the final unique sample name.
        """
        image_dir = datadir / self.metadata["subdirs"]["image_dir"]
        datamap_dir = datadir / self.metadata["subdirs"]["datamap_dir"]

        # Base name for this sample
        scene = Path(self.metadata["name"]).stem
        name = f"{scene}_gen_{self.gen_cnt:02d}_sample_{idx:04d}"
        if getattr(self, "version_cnt", 0) > 0:
            name += f"_v_{self.version_cnt:04d}"

        # Ensure name uniqueness on disk
        def _exists(n: str) -> bool:
            return (image_dir / n).with_suffix(".jpg").is_file() or (
                image_dir / n
            ).with_suffix(".png").is_file()

        while _exists(name) and self.version_cnt < 10_000:
            print(f"Sample {name} already exists. Appending version index.")
            self.version_cnt = getattr(self, "version_cnt", 0) + 1
            name = f"{scene}_gen_{self.gen_cnt:02d}_sample_{idx:04d}_v_{self.version_cnt:04d}"

        # Save image (if given). Prefer .jpg as before.
        if "image" in sample and sample["image"] is not None:
            image_filename = (image_dir / name).with_suffix(".jpg")
            cv2.imwrite(str(image_filename), sample["image"])

        # Save datamap (if given)
        if "datamap" in sample and sample["datamap"] is not None:
            datamap_filename = (datamap_dir / name).with_suffix(".npz")
            np.savez(datamap_filename, **sample["datamap"])
        else:
            print(
                f"Warning: No datamap provided for sample {name}. Skipping datamap saving."
            )

        # Update metadata entry
        sample_meta = {
            "name": name,
            "description": sample.get("description", ""),
            "intrinsics": sample["intrinsics"],
            "rotation_wxyz": sample["rotation_wxyz"].tolist(),
            "translation": sample["translation"].tolist(),
            "generation": self.gen_cnt,
            "artifact_removal": sample.get("artifact_removal", False),
            "background_color": sample.get("background_color", [0, 0, 0]),
            # Optional passthroughs if present (won't hurt, helps downstream)
            **({} if "pose_type" not in sample else {"pose_type": sample["pose_type"]}),
            **({} if "image_idx" not in sample else {"image_idx": sample["image_idx"]}),
            **({} if "filename" not in sample else {"filename": sample["filename"]}),
        }
        # Add to appropriate split
        if split == "train":
            if "samples_train" not in self.metadata:
                self.metadata["samples_train"] = {}
            self.metadata["samples_train"][name] = sample_meta
        elif split == "val":
            if "samples_val" not in self.metadata:
                self.metadata["samples_val"] = {}
            self.metadata["samples_val"][name] = sample_meta
        elif split == "test":
            if "samples_test" not in self.metadata:
                self.metadata["samples_test"] = {}
            self.metadata["samples_test"][name] = sample_meta
        else:
            if "samples" not in self.metadata:
                self.metadata["samples"] = {}
            self.metadata["samples"][name] = sample_meta
        return name

    def save_metadata(self, datadir: Path) -> Path:
        """
        Persist self.metadata to disk.
        Returns the path of the written metadata file.
        """
        # Keep same behavior of renaming metadata["name"] to new_metadata (without .json)
        self.metadata["name"] = (
            self.new_metadata
            if ".json" not in self.new_metadata
            else self.new_metadata[:-5]
        )
        metaname = (
            self.new_metadata
            if ".json" in self.new_metadata
            else self.new_metadata + ".json"
        )

        metadata_dir = datadir / self.metadata["subdirs"]["metadata_dir"]
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / metaname

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Metadata saved to {metadata_path}")
        return metadata_path

    def add_samples(self, samples: list[dict], new_data_dir: Path | None = None) -> str:
        """
        Integrate a list of samples into the dataset:
        - Bumps generation counter once
        - Ensures directories exist
        - Adds each sample via add_sample
        - Saves metadata at the end
        Returns self.new_metadata (same as original).
        """
        # New generation
        self.gen_cnt += 1

        datadir = Path(new_data_dir) if new_data_dir is not None else self.datadir
        print(f"Using new data directory: {datadir}")

        # Ensure directories exist once
        image_dir = datadir / self.metadata["subdirs"]["image_dir"]
        datamap_dir = datadir / self.metadata["subdirs"]["datamap_dir"]
        image_dir.mkdir(parents=True, exist_ok=True)
        datamap_dir.mkdir(parents=True, exist_ok=True)

        # Add each sample
        for i, sample in enumerate(samples):
            self.add_sample(self, sample=sample, idx=i, datadir=datadir)

        # Persist metadata
        self.save_metadata(self, datadir=datadir)

        print(f"Added {len(samples)} samples to the dataset.")
        return self.new_metadata

    def clear_generated(self):
        """
        Clear all the generated samples from the metadata and reset counters
        """
        sample_dict_keys = ["samples", "samples_train", "samples_val", "samples_test"]
        for key in sample_dict_keys:
            if key in self.metadata:
                samples = self.metadata[key]
                new_samples = {}
                for name, sample in samples.items():
                    if "generation" in sample and sample["generation"] == 0:
                        new_samples[name] = sample
                self.metadata[key] = new_samples
        # Reset counters
        self.version_cnt = 0
        self.gen_cnt = 0
