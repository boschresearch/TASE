# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from gsplat V 1.5.3
#   (https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py)
# Copyright 2025 Nerfstudio Team, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never
from pathlib import PosixPath

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import contextlib

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from TASE.splatting.util.utils import (
    CameraOptModule,
    knn,
    compute_knn,
    rgb_to_sh,
    set_random_seed,
    sh_to_rgb,
)
from TASE.splatting.util.traj import (
    generate_interpolated_path,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from TASE.splatting.trainer.custom_strategy import DefaultStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.utils import save_ply

### Additional imports
from TASE.splatting.trainer.TASE_dataparser import TASE_DataParserConfig, TASE
from TASE.splatting.trainer.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from TASE.splatting.trainer.TASE_dataset import TASE_dataset
from nerfstudio.data.datasets.base_dataset import (
    InputDataset,
)  # I might just fork this so that I no longer import any code from nerfstudio. It does have some further dependencies though.
from TASE.splatting.viewer.viewer import Viewer, ColorMapper
from TASE.splatting.common.background_model import MLP_BackgroundModel
from TASE.splatting.trainer.trainer_cfg import Config
from TASE.splatting.common.rendering import Scene
from TASE.splatting.util.data_util import *
from TASE.image_gen.dimensionality_reduction.embedder_stats import (
    InDistMarginLoss,
    build_loss_from_stats,
    sample_from_stats,
)

# memory profiling
import torchvision
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function


def indices_to_one_hot(indices: torch.Tensor, dtype: torch.dtype = torch.float32):
    """
    Convert a tensor of class indices to a compact one-hot encoding over only
    the classes present, and return a mapping from original class indices to
    new one-hot positions.

    Args:
        indices (Tensor): LongTensor of shape (N, 1) or (N,), with values in [0, C-1].
        dtype (torch.dtype): output dtype (default: float32).

    Returns:
        one_hot (Tensor): One-hot encoded tensor of shape (N, M), where M is the
                          number of unique classes in `indices`, in dtype.
        mapping (dict):   A dict mapping original class indices -> new positions [0..M-1].
    """
    # flatten to 1D
    idx = indices.view(-1).long()

    # find unique classes and sort
    unique_classes = torch.unique(idx)
    unique_classes, _ = torch.sort(unique_classes)

    # build mapping from old index to new position
    mapping = {int(c): i for i, c in enumerate(unique_classes)}

    # remap idx to new indices
    new_idx = idx.clone()
    for old, new in mapping.items():
        new_idx[idx == old] = new

    # create one-hot encoding
    num_classes = len(unique_classes)
    if num_classes > 255:
        print(
            f"Warning: High number of classes ({num_classes}), could incur large memory footprint."
        )
    one_hot = F.one_hot(new_idx, num_classes=len(unique_classes)).to(dtype)

    return one_hot, mapping


def create_splats_with_optimizers(
    parser: TASE,
    init_type: Literal["sfm", "random", "lidar", "depthmap", "sem_pc"],
    learning_rates: Optional[Dict[str, float]] = None,
    init_num_pts: int = 100_000,
    init_extent: float = 1.5,
    init_opacity: float = 0.1,
    init_scale: float = 0.5,
    max_init_size: Optional[float] = None,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    colormap: Union[str, dict] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create splats and optimizers."""
    info = {}
    info["pointclouds"] = {}

    print(
        f"Creating splats with init_type={init_type}, init_num_pts={init_num_pts}, init_extent={init_extent}, init_opacity={init_opacity}, init_scale={init_scale}, sh_degree={sh_degree}"
    )

    # define the scene space for initialization
    scene_box = parser.scene_box
    # extend the scene box in z direction since the cameras usually lie within a plane
    scene_box[0, 2] -= 5.0
    scene_box[1, 2] += 20.0
    diff = (scene_box[1] - scene_box[0]) / 2.0
    center = scene_box[0] + diff
    # extend the total scale accoding to the config
    aabb_min, aabb_max = center - diff * init_extent, center + diff * init_extent

    if init_type == "sfm":
        # I might remove this later on
        points = torch.from_numpy(parser.points)
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        info["pointclouds"]["sfm_points"] = points
    elif init_type == "random":
        # Sample points uniformly within the scene_box
        points = torch.rand((init_num_pts, 3)) * (aabb_max - aabb_min) + aabb_min
        rgbs = torch.rand((init_num_pts, 3))
        info["pointclouds"]["rand_points"] = points
    elif init_type == "lidar":
        # Get the lidar points from the dataparser (resampled in voxels)
        lidar_points = parser.generate_lidar_pc().float()
        lidar_points = lidar_points[
            ..., :3
        ]  # remove the semantic labels values if they are present
        # Sample remaining points uniformly within the scene_box
        num_rand = init_num_pts - lidar_points.shape[0]
        rand_points = torch.rand((num_rand, 3)) * (aabb_max - aabb_min) + aabb_min
        points = torch.cat([lidar_points, rand_points], dim=0)
        rgbs = torch.rand((points.shape[0], 3)).float()
        # for display in the viewer
        info["pointclouds"]["init_points"] = lidar_points
        info["pointclouds"]["rand_points"] = rand_points
    elif init_type == "depthmap":
        # Get the pointclouds from the depthmaps
        dataparser_out = parser._generate_dataparser_outputs()
        points_list = generate_depthmap_pointcloud(dataparser_out, downscale_factor=16)
        points = np.concatenate(points_list, axis=0)
        points = torch.from_numpy(points).float().to(device)
        # Downsample the pointclouds to the desired number of points
        if init_num_pts is not None and points.shape[0] > init_num_pts:
            indices = voxelize_and_sample_pointcloud(
                points, n=init_num_pts, occupancy_threshold=3
            )
            points = points[indices]
        rgbs = points[:, 3:6]
        points = points[:, :3]
        # display in the viewer
        info["pointclouds"]["init_points"] = points
    elif init_type == "sem_pc":
        # Load the pointcloud
        points, _, class_idxs = parser.load_pointcloud()
        points = torch.from_numpy(points).float()
        class_idxs = torch.from_numpy(class_idxs).long()
        if init_num_pts is not None and points.shape[0] > init_num_pts:
            indices = voxelize_and_sample_pointcloud(
                points, n=init_num_pts, occupancy_threshold=3
            )
            points = points[indices]
            class_idxs = class_idxs[indices]
        points, colormap_dict = SemanticLabels.colormap_sem_pc(
            points, class_idxs, colormap=colormap
        )  # colorize the points for visualization
        info["pointclouds"]["sem_points"] = points
        info["colormap_info"] = colormap_dict
        # Random RGB colors
        rgbs = torch.rand((points.shape[0], 3)).float()
        points = points[:, :3]
    elif init_type == "pointcloud":
        # generic npz pointcloud with xyz and rgb values
        points, rgbs, _ = parser.load_pointcloud()
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgbs).float() / 255.0
        info["pointclouds"]["init_points"] = points
    else:
        raise ValueError(
            "Please specify a correct init_type: lidar, depthmap or random"
        )

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    min_num_neighbours = 5
    dist2_knn, ind_knn = compute_knn(points, min_num_neighbours * 3)
    dist2_avg = (dist2_knn[:, 1:4]).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    if max_init_size is not None:
        # Remove any points that are too big now (floaters)
        too_big = dist_avg > max_init_size
        if too_big.any():
            print(f"Info: {too_big.sum().item()} points are too big, removing them.")
            points = points[~too_big]
            rgbs = rgbs[~too_big]
            scales = scales[~too_big]
            class_idxs = class_idxs[~too_big] if init_type == "sem_pc" else None
    scales = torch.clamp(scales, max=0.05)  # [N, 3]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), learning_rates["means"] * scene_scale),
        ("scales", torch.nn.Parameter(scales), learning_rates["scales"]),
        ("quats", torch.nn.Parameter(quats), learning_rates["quats"]),
        ("opacities", torch.nn.Parameter(opacities), learning_rates["opacities"]),
    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), learning_rates["sh0"]))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), learning_rates["shN"]))

    # boolean mask for selection
    selection = torch.zeros((N, 1), dtype=torch.float32)
    params.append(("selection", torch.nn.Parameter(selection), 0.0))

    if feature_dim is not None:
        # features will be used to render a feature map
        feature_init_dist = parser.metadata.get("feature_stat_path", None)
        if feature_init_dist is not None:
            # Load the feature distribution from a file
            features = sample_from_stats(feature_init_dist, B=N, device=device)
            if features.shape[-1] > feature_dim:
                print("Truncating loaded features to match feature_dim.")
                features = features[:, :feature_dim]
            if features.shape[-1] < feature_dim:
                print(
                    f"[ERROR] Feature dimension of the loaded distribution ({features.shape[-1]}) does not match the specified feature_dim ({feature_dim}). Initializing randomly."
                )
                features = torch.normal(mean=0, std=0.3, size=(N, feature_dim))
        else:
            # Initialize features randomly, roughly following the expected distribution of the features
            print(
                "Warning: Initializing features randomly, consider providing a feature distribution file for better results."
            )
            features = torch.normal(mean=0, std=0.3, size=(N, feature_dim))
        params.append(
            ("features", torch.nn.Parameter(features), learning_rates["features"])
        )

    if init_type == "sem_pc":
        clx_one_hot, mapping = indices_to_one_hot(class_idxs)
        info["one_hot_mapping"] = mapping
        params.append(
            ("classes", torch.nn.Parameter(clx_one_hot), learning_rates["classes"])
        )

    # Convert to the desired dtype and display the memory footprint
    for i, (name, value, lr) in enumerate(params):
        # cast
        new_value = value.to(dtype)
        # compute bytes
        n_bytes = new_value.element_size() * new_value.nelement()
        # display in GB
        print(f"{name:20s}: {n_bytes / (1024**3):6.3f} GB")
        # store back
        params[i] = (name, new_value, lr)

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers, info


class Runner:
    """Engine for training and testing."""

    viewer = None

    def load_dataset(self, active_sample_gens=None):
        cfg = self.cfg
        # Load data: Training data should contain initial points and colors.
        if cfg.dataset_type is None or cfg.dataset_type == "colmap":
            print(
                "This runner only supports TASE datasets. Please convert to this format first."
            )
            return None
        elif cfg.dataset_type == "TASE":
            print("Warning: TASE dataset format is currently under development!")
            if ".json" not in cfg.scene:
                cfg.scene += ".json"
            data_dir = PosixPath(cfg.data_dir) 
            if not data_dir.exists():
                data_dir = PosixPath(os.environ.get("TASE_ROOT", "")) / cfg.data_dir
            parser_cfg = TASE_DataParserConfig(
                data_dir=data_dir,
                metadata=cfg.scene,
                active_sample_gens=active_sample_gens,
            )
            config = FullImageDatamanagerConfig(dataparser=parser_cfg)
            self.data_mngr = FullImageDatamanager[TASE_dataset](config=config)
            self.parser = self.data_mngr.dataparser
        elif cfg.dataset_type == "nuscenes":
            if cfg.mask_dir is not None:
                mask_dir = PosixPath(cfg.mask_dir)
            else:
                mask_dir = None
            parser_cfg = NuScenesDataParserConfig(
                data_dir=PosixPath(cfg.data_dir),
                data=PosixPath(
                    cfg.scene
                ),  # the specific nuScenes scene to load, e.g., "scene-0071"
                local_dir=PosixPath(
                    cfg.local_data_dir
                ),  # the local directory where the data is stored
                mask_dir=mask_dir,  # masks of dynamic objects generated with /nerfstudio/scripts/datasets/process_nuscenes_masks.py
                cameras=(
                    "FRONT",
                    "FRONT_LEFT",
                    "FRONT_RIGHT",
                    "BACK_LEFT",
                    "BACK_RIGHT",
                    "BACK",
                ),  # intrinsics for the BACK camera significantly differ from the other cameras, depth maps show some higher disparity
                # cameras = ('FRONT', ),
                version="v1.0-mini",
            )
            # self.parser = NuScenes(parser_cfg)
            if self.cfg.depth_loss:
                self.data_mngr = FullImageDatamanager[DepthMaskDataset](
                    config=FullImageDatamanagerConfig(dataparser=parser_cfg)
                )
            else:
                self.data_mngr = FullImageDatamanager[InputDataset](
                    config=FullImageDatamanagerConfig(dataparser=parser_cfg)
                )
            self.parser = self.data_mngr.dataparser

        # Load the camera poses
        self.train_cam_poses, self.train_cam_ks, self.train_cam_dims = (
            self.parser.cam_poses,
            self.parser.cam_Ks,
            self.parser.cam_dims,
        )
        print(f"Number of training views: {len(self.train_cam_poses)}")

        self.active_sample_gens = active_sample_gens

    def add_cams_to_viewer(self):
        """Add the camera poses to the viewer."""
        info = {}
        samples_dict = get_samples_dict(
            data_mngr=self.data_mngr, device=self.device, info=info
        )
        # Random color for the cameras
        color = np.random.rand(3).astype(np.float32)  # Random color for the cameras
        if not self.cfg.disable_viewer and self.viewer is not None:
            # Add the camera poses to the viewer
            self.viewer.add_samples(samples_dict["cameras"], color=color)
            print(f"Added {len(self.train_cam_poses)} cameras to the viewer.")
        else:
            print("Viewer is disabled, skipping adding cameras.")

    def __init__(
        self,
        cfg: Config,
        scene: Scene = None,
        parser: TASE = None,
        viewer: Viewer = None,
        load_dataset: bool = True,
        active_sample_gens: Optional[List[int]] = None,
        eval_mode: bool = False,
    ) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = f"cuda"
        self.eval_mode = eval_mode

        # not sure if this makes sense
        self.scene_scale = cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if parser is not None:
            self.parser = parser
            parser_cfg = parser.config
            config = FullImageDatamanagerConfig(dataparser=parser_cfg)
            self.data_mngr = FullImageDatamanager[TASE_dataset](config=config)
            self.parser = self.data_mngr.dataparser

        # torch.cuda.memory._record_memory_history(max_entries=1000000, stacks='all')

        # Make sure the last step is always used for saving and evaluation
        if (cfg.max_steps) not in cfg.eval_steps:
            cfg.eval_steps.append(cfg.max_steps)
        if (cfg.max_steps) not in cfg.save_steps:
            cfg.save_steps.append(cfg.max_steps)

        if not self.eval_mode:
            # Where to dump results.
            os.makedirs(cfg.result_dir, exist_ok=True)
            # Tensorboard writer
            self.writer = SummaryWriter(log_dir=self.cfg.tb_dir)
        else:
            self.writer = None
            self.cfg.disable_viewer = True  # disable viewer in eval mode, since it is not needed and can cause issues with multiple runs on the same machine
        self.last_ckpt_path = ""
        # Setup output directories.
        cfg.set_paths()
            
        # Load the dataset
        if load_dataset:
            self.load_dataset(active_sample_gens=active_sample_gens)
        else:
            print(
                "Skipping dataset loading, please load the specific sample generations manually!"
            )
            self.active_sample_gens = None

        if scene is None:
            if self.eval_mode:
                raise ValueError("Error: Scene must be provided in eval mode.")
            print("Creating a new scene...")
            self.scene = Scene(
                cfg=cfg,
                device=self.device,
            )

            # Initialize the gaussians
            self.scene.splats, self.optimizers, info = create_splats_with_optimizers(
                self.parser,
                init_type=cfg.init_type,
                learning_rates=cfg.learning_rates,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                max_init_size=cfg.max_init_size,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                feature_dim=cfg.num_semantic_features,
                sparse_grad=cfg.sparse_grad,
                visible_adam=cfg.visible_adam,
                batch_size=cfg.batch_size,
                device=self.device,
                dtype=(
                    torch.bfloat16 if cfg.splat_dtype == "bfloat16" else torch.float32
                )
            )

            # Background Model
            if cfg.background_model == "mlp":
                self.scene.background_model = MLP_BackgroundModel().to(self.device)
                self.scene.background_optimizer = torch.optim.Adam(
                    self.scene.background_model.parameters(),
                    lr=cfg.learning_rates.get("background", 1e-5),
                )
                if cfg.feature_loss:
                    self.scene.feature_bg_model = MLP_BackgroundModel(
                        output_dim=cfg.num_semantic_features
                    ).to(self.device)
                    self.scene.feature_bg_optimizer = torch.optim.Adam(
                        self.scene.feature_bg_model.parameters(),
                        lr=cfg.learning_rates.get("feature_bg", 1e-5),
                    )
        else:
            print("Using the provided scene...")
            info = {}
            self.scene = scene
            optimizer_class = None
            if cfg.sparse_grad:
                optimizer_class = torch.optim.SparseAdam
            elif cfg.visible_adam:
                optimizer_class = SelectiveAdam
            else:
                optimizer_class = torch.optim.Adam
            BS = cfg.batch_size
            self.optimizers = {}
            for name, param in self.scene.splats.items():
                if not param.requires_grad:
                    continue
                self.optimizers[name] = optimizer_class(
                    [
                        {
                            "params": param,
                            "lr": cfg.learning_rates.get(name, 0.0) * math.sqrt(BS),
                            "name": name,
                        }
                    ],
                    eps=1e-15 / math.sqrt(BS),
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                )

            # Make sure the scene has the same config
            self.scene.cfg = cfg
            self.scene.sh_degree = cfg.sh_degree
            # If there are background models, add them to the optimizers
            if (
                hasattr(self.scene, "background_model")
                and self.scene.background_model is not None
            ):
                self.scene.background_optimizer = torch.optim.Adam(
                    self.scene.background_model.parameters(),
                    lr=cfg.learning_rates.get("background", 1e-5),
                )
            if (
                hasattr(self.scene, "feature_bg_model")
                and self.scene.feature_bg_model is not None
            ):
                self.scene.feature_bg_model = MLP_BackgroundModel(
                    output_dim=cfg.num_semantic_features
                ).to(self.device)
                self.scene.feature_bg_optimizer = torch.optim.Adam(
                    self.scene.feature_bg_model.parameters(),
                    lr=cfg.learning_rates.get("feature_bg", 1e-5),
                )

        print("Model initialized. Number of GS:", len(self.scene.splats["means"]))

        # Load the loss module for the feature distance regularization
        self.feature_dist_loss_module = None
        if cfg.feature_dist_regularization > 0.0:
            path = Path(self.parser.metadata.get("feature_stat_path", None))
            path = resolve_path(path)
            if not path or not path.exists():
                print(
                    f"ERROR: Could not find the feature distance loss module at {path}. Setting weight to 0.0."
                )
                cfg.feature_dist_regularization = 0.0
            else:
                loss_module = build_loss_from_stats(stats_path=path, device=self.device)
                self.feature_dist_loss_module = loss_module
                print(f"Loaded feature distribution loss module from {path}.")

        # Densification Strategy
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # Pose optimization
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.parser.cam_poses)).to(
                self.device
            )
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.parser.cam_poses)).to(
                self.device
            )
            self.pose_perturb.random_init(cfg.pose_noise)

        # Per image embedding for whitebalance and brightness correction
        if cfg.color_correct:
            # create the color correction matrices
            num_images = len(self.parser.cam_poses)
            # M is defined as an offset from the identity. This way m can be regularized to be close to 0.
            self.color_correct_m = (
                torch.zeros((num_images, 3, 3)).float().to(device=self.device)
            )
            self.color_correct_m.requires_grad = True
            self.color_correct_b = (
                torch.zeros((num_images, 3)).float().to(device=self.device)
            )
            self.color_correct_b.requires_grad = True
            self.color_correct_optimizer = torch.optim.Adam(
                [self.color_correct_m, self.color_correct_b],
                lr=cfg.color_correct_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.color_correct_reg,
            )
        else:
            self.color_correct_m = None
            self.color_correct_b = None

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # For depth loss decay
        self.cur_depth_lambda = None

        # Skipping the prep of visualizations if the viewer is passed or disabled
        if viewer is None and not cfg.disable_viewer:
            # Generate a pc from the depthmaps for the viewer
            if cfg.generate_depthmap_pc:
                pointclouds = generate_depthmap_pointcloud()
                for i, pc in enumerate(pointclouds):
                    info["pointclouds"][f"depthmap_points_{i}"] = pc

            # Try to load the semantic pointcloud for visualization
            if hasattr(self.parser, "load_sem_pc") and not cfg.init_type == "sem_pc":
                sem_points = self.parser.load_sem_pc()
                if sem_points is not None:
                    points, labels = sem_points
                    points = torch.from_numpy(points).float()
                    labels = torch.from_numpy(labels).long()
                    idxs = voxelize_and_sample_pointcloud(points, n=100_000)
                    points = points[idxs]
                    labels = labels[idxs]
                    sem_points, colormap_info = SemanticLabels.colormap_sem_pc(
                        points, labels
                    )
                    info["pointclouds"]["sem_points"] = sem_points
                    info["colormap_info"] = colormap_info

            # store as numpy files for visualization
            if cfg.store_train_samples:
                try:
                    samples_dict, _ = store_samples(
                        data_mngr=self.data_mngr,
                        device=self.device,
                        info=info,
                        path=os.path.join(self.cfg.result_dir, "training_samples"),
                    )
                except Exception as e:
                    print(f"Error while storing training samples: {e}")
                    samples_dict = None
            else:
                samples_dict = get_samples_dict(
                    data_mngr=self.data_mngr, device=self.device, info=info
                )

            # Colormapper for the viewer functions
            self.scene.depth_colormapper = ColorMapper(device=self.device)

        # Viewer
        if viewer is not None:
            self.viewer = viewer
            self.server = viewer.server
            self.cfg.disable_viewer = False
            self.viewer.set_mode("training")
            print("Using the provided viewer...")
        elif not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = Viewer(
                server=self.server,
                scene=self.scene,
                cfg=cfg,
                mode="training",
                initial_pose=self.parser.cam_poses[0],
                colormapper=self.scene.depth_colormapper,
                samples=samples_dict,
                color_correct=(
                    (self.color_correct_m, self.color_correct_b)
                    if cfg.color_correct
                    else None
                ),
                pose_opt=self.pose_adjust if cfg.pose_opt else None,
                pose_perturb=self.pose_perturb if cfg.pose_noise > 0.0 else None,
            )
        else:
            self.viewer = None
            self.server = None
            print("Viewer is disabled.")

    def train(
        self,
        paused=False,
        steps: Optional[int] = None,
        densify=True,
        generational_weights: Optional[dict] = None,
    ) -> None:
        """
        Optimize the Gaussians.
        """

        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        if steps is not None:
            max_steps = steps
        else:
            max_steps = cfg.max_steps
        print(f"Training for {max_steps} steps...")
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0],
                    gamma=0.01 ** (1.0 / (max_steps - cfg.pose_opt_start_iter)),
                )
            )

        # Training loop.
        global_tic = time.time()

        print(f"Start fitting {cfg.scene}...")

        if not cfg.disable_viewer and paused:
            self.viewer.state.status = "paused"

        pbar = tqdm.tqdm(range(init_step, max_steps), disable=None)

        # set all the losses to none first
        l1loss = None
        ssimloss = None
        depthloss = None
        normal_loss = None
        alpha_loss = None
        local_alpha_loss = None
        feat_loss = None
        feat_dist_loss = None
        feat_smoothness_loss = None
        geometry_loss = None
        reg_loss = None
        info = None  # placeholder for the info dict returned by the rasterizer

        # reset the state of the densification strategy
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                    if self.viewer.cancel:
                        break
                while not self.viewer.lock.acquire(timeout=1.0):
                    time.sleep(0.01)
                    if self.viewer.cancel:
                        break
                tic = time.time()

            data = self.data_mngr.next_train(step=1)

            # Unpack data
            cams, data = data
            camtoworlds = cams.camera_to_worlds.to(device)  # [1, 3, 4]
            camtoworlds = convert_to_4x4(camtoworlds)
            camtoworlds_initial = camtoworlds.clone()
            Ks = []
            for cam in cams:
                K = torch.tensor(
                    [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]]
                ).to(device)
                Ks.append(K)
            Ks = torch.stack(Ks, dim=0)  # [n, 3, 3]
            pixels = data["image"].to(device)  # [H, W, 3]
            pixels = pixels.unsqueeze(0)  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = torch.tensor(data["image_idx"]).unsqueeze(0).to(device)

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            # Get masks if available and required
            dyn_masks = (
                data["dyn_mask"].unsqueeze(0).to(device)
                if cfg.use_dyn_mask and "dyn_mask" in data
                else torch.ones_like(pixels[..., 0:1], device=device)
            )  # [1, H, W, 1]
            loss_masks = (
                data["loss_mask"].unsqueeze(0).to(device)
                if cfg.use_loss_mask and "loss_mask" in data
                else torch.ones_like(pixels[..., 0:1], device=device)
            )
            bg_masks = (
                data["bg_masks"].unsqueeze(0).to(device)
                if cfg.use_bg_mask and "bg_masks" in data
                else torch.ones_like(pixels[..., 0:1], device=device)
            )

            # Artifact removal
            if cams.metadata[data["image_idx"]].get("artifact_removal", 0):
                background_color = torch.tensor(
                    cams.metadata[data["image_idx"]].get(
                        "artifact_background_color", [0.0, 0.0, 0.0]
                    )
                ).to(device)
                inpainting_mask = torch.tensor(
                    cams.metadata[data["image_idx"]].get("inpainting_mask")
                ).to(device)
                artifact_removal = True
            else:
                background_color = torch.tensor([0.0, 0.0, 0.0]).to(device)
                inpainting_mask = torch.ones_like(pixels[..., 0:1], device=device)
                artifact_removal = False

            # if there are generational weights, retrieve the gen to which the current image belongs to
            if generational_weights is not None:
                gen_id = cams.metadata[data["image_idx"]].get("generation", 0)
                loss_weight = generational_weights.get(gen_id, 1.0)
            else:
                loss_weight = 1.0

            if cfg.depth_loss:
                if "depth" not in data:
                    print(
                        "Warning: Depth maps not found in the dataset, disabeling the depth supervision."
                    )
                    cfg.depth_loss = False
                else:
                    depths_gt = data["depth"].unsqueeze(0).to(device)  # [1, H, W, 1]
                    depths_conf = (
                        data["depth_confidence"].unsqueeze(0).to(device)
                    )  # [1, H, W, 1]

            if cfg.normal_loss:
                if "normals" not in data:
                    print(
                        "Warning: Normal maps not found in the dataset, disabeling the normal supervision."
                    )
                    cfg.normal_loss = False
                else:
                    normals_gt = data["normals"].unsqueeze(0).to(device)  # [1, H, W, 3]
                    normals_conf = (
                        data["normal_confidence"].unsqueeze(0).to(device)
                    )  # [1, H, W, 1]

            if cfg.num_semantic_features > 0:
                sem_features = data["reduced_embeddings"].unsqueeze(0).to(device)
                sem_features = sem_features[
                    ..., : cfg.num_semantic_features
                ]  # [1, H, W, num_semantic_features]

            height, width = pixels.shape[1:3]

            if cfg.pose_opt and step > cfg.pose_opt_start_iter:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            loss = 0.0

            """
            NOTE: Features and RGB colors could be rasterized jointly, but since the feature maps are usually at a lower resolution than the RGB images, we rasterize them separately.
            If the feature maps were at the same resolution as the RGB images, we could rasterize them together.
            """

            if cfg.feature_loss:
                # Rasterize at the resolution of the feature maps
                _, feat_H, feat_W, N_dims = sem_features.shape
                scale_x = feat_W / width
                scale_y = feat_H / height
                Ks_feat = resize_intrinsics(
                    K=Ks,
                    scale=(scale_x, scale_y),
                )
                # Resize the dynamic & loss masks to the feature map resolution
                if dyn_masks is not None:
                    dyn_masks_feat = F.interpolate(
                        dyn_masks.permute(0, 3, 1, 2),
                        size=(feat_H, feat_W),
                        mode="area",
                    ).permute(0, 2, 3, 1)
                else:
                    dyn_masks_feat = torch.ones(
                        (1, feat_H, feat_W, 1), device=device, dtype=pixels.dtype
                    )
                if loss_masks is not None:
                    loss_masks_feat = F.interpolate(
                        loss_masks.permute(0, 3, 1, 2),
                        size=(feat_H, feat_W),
                        mode="area",
                    ).permute(0, 2, 3, 1)
                else:
                    loss_masks_feat = torch.ones_like(dyn_masks_feat)
                if inpainting_mask is not None:
                    inpainting_mask_feat = F.interpolate(
                        inpainting_mask.permute(0, 3, 1, 2),
                        size=(feat_H, feat_W),
                        mode="area",
                    ).permute(0, 2, 3, 1)

                feat_renders, alphas, normals, info = self.scene.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks_feat,
                    width=feat_W,
                    height=feat_H,
                    sh_degree=None,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB",
                    color_mode="Features",
                    masks=dyn_masks_feat,
                )
                feat_dims = feat_renders.shape[-1]
                weights = gaussian_decay_weights(
                    feat_renders.shape[-1],
                    1.0,
                    0.05,
                    sigma_ratio=0.1,
                    device=feat_renders.device,
                    dtype=feat_renders.dtype,
                )
                sem_features = sem_features[..., :feat_dims]

                feat_renders_w = feat_renders * weights[None, None, None, :]
                sem_features_w = sem_features * weights[None, None, None, :]

                # Per-pixel feature loss map
                feat_loss_map = 1 - F.cosine_similarity(
                    feat_renders_w, sem_features_w, dim=-1
                )
                feat_loss_map += 0.1 * (feat_renders_w - sem_features_w).abs().mean(
                    dim=-1
                )
                mask_feat = (
                    dyn_masks_feat[..., 0]
                    * loss_masks_feat[..., 0]
                    * inpainting_mask_feat[..., 0]
                ).float()
                denom_feat = mask_feat.sum().clamp_min(1.0)
                feat_loss = (feat_loss_map * mask_feat).sum() / denom_feat
                loss = loss + feat_loss * cfg.feature_lambda
            else:
                feat_loss = None
                feat_renders = None

            # Feature distribution regularization
            if (
                cfg.feature_loss
                and cfg.feature_dist_regularization > 0.0
                and self.feature_dist_loss_module is not None
                and step % cfg.feature_dist_loss_interval == 0
            ):
                # calculate the feature distance loss
                feat_dist_loss, _ = self.feature_dist_loss_module(
                    self.scene.splats["features"],
                )
                feat_dist_loss = feat_dist_loss
                loss = loss + feat_dist_loss * cfg.feature_dist_regularization

            # Feature smoothness loss (3D) -> Move this to a function
            if (
                cfg.feature_loss
                and cfg.feature_local_smoothness > 0.0
                and step % cfg.feature_smoothness_loss_interval == 0
            ):
                means = self.scene.splats["means"]  # (N,3), float
                feats = self.scene.splats["features"]  # (N,C), float
                assert means.ndim == 2 and means.shape[1] == 3
                N = means.shape[0]

                if N >= 2:
                    # ---- Hyperparams (with safe fallbacks) ----
                    K = min(
                        getattr(cfg, "feature_knn_K", 5), N
                    )  # total neighbors incl. self
                    dmax = getattr(
                        cfg, "feature_knn_max_dist", float("inf")
                    )  # absolute 3D distance
                    max_age = getattr(
                        cfg, "feature_knn_recompute_max_steps", 0
                    )  # 0 => only when N changes

                    # ---- Recompute policy: on count change OR after max_age steps ----
                    need_recompute = (
                        (not hasattr(self, "knn_inds"))
                        or (self.knn_inds.shape[0] != N)
                        or (
                            max_age > 0
                            and (
                                not hasattr(self, "knn_last_step")
                                or (step - self.knn_last_step) >= max_age
                            )
                        )
                    )

                    if need_recompute:
                        with torch.no_grad():
                            xyz = means.to(dtype=torch.float32)  # device-preserving
                            d2, idx = compute_knn(xyz, K=K)  # (1,N,K)
                            self.knn_inds = idx.long()  # (N,K); [:,0] == self
                            self.knn_d2 = d2  # (N,K) squared distances
                            self.knn_last_step = int(step)

                    # ---- Cosine smoothness with distance mask (exclude self and far neighbors) ----
                    nbr_idx = self.knn_inds[:, 1:]  # (N, K-1)
                    nbr_d2 = self.knn_d2[:, 1:]  # (N, K-1)
                    if nbr_idx.numel() > 0:
                        # distance gate
                        if torch.isfinite(torch.tensor(dmax, device=means.device)):
                            valid = nbr_d2 <= (dmax * dmax)
                        else:
                            valid = torch.ones_like(nbr_d2, dtype=torch.bool)

                        # gather features
                        ref = F.normalize(feats, dim=-1, eps=1e-6).unsqueeze(
                            1
                        )  # (N,1,C)
                        nbr = F.normalize(feats[nbr_idx], dim=-1, eps=1e-6)  # (N,K-1,C)

                        # cosine similarity and loss (1 - cos), averaged over valid pairs only
                        cos = (ref * nbr).sum(-1)  # (N,K-1)
                        per_pair = (1.0 - cos).clamp_min(
                            0.0
                        ) * valid.float()  # mask out invalid
                        denom = valid.sum().clamp(min=1)  # avoid div-by-zero
                        feat_smoothness_loss = per_pair.sum() / denom

                        loss = (
                            loss + feat_smoothness_loss * cfg.feature_local_smoothness
                        )

            # geometry change penalty
            if cfg.geometry_change_penalty and cfg.geometry_change_penalty > 0.0:
                if "original_means" not in self.scene.splats:
                    print(
                        "Warning: original_means not found in splats, cannot apply geometry change penalty."
                    )
                    cfg.geometry_change_penalty = 0.0
                else:
                    geometry_change_penalty = cfg.geometry_change_penalty
                    p = int(getattr(cfg, "geometry_change_exp", 4))
                    p = max(2, p)  # ensure even >=2
                    diff_means = (
                        self.scene.splats["means"] - self.scene.splats["original_means"]
                    )
                    diff_scales = (
                        self.scene.splats["scales"]
                        - self.scene.splats["original_scales"]
                    )

                    # promote to float32 for stability
                    dm = diff_means.to(torch.float32)
                    ds = diff_scales.to(torch.float32)

                    # squared norms
                    means_sq = dm.pow(2).sum(-1)
                    scales_sq = ds.pow(2).sum(-1)

                    # clamp to prevent overflow pre-exponent
                    means_sq = means_sq.clamp_max(1e6)
                    scales_sq = scales_sq.clamp_max(1e6)

                    loss_means = means_sq.pow(p / 2).mean()
                    loss_scales = scales_sq.pow(p / 2).mean()

                    geom_raw = loss_means + loss_scales

                    geometry_loss = geom_raw * geometry_change_penalty
                    if not torch.isfinite(geometry_loss):
                        geometry_loss = torch.zeros((), device=geom_raw.device)

                    loss = loss + geometry_loss

            # Image space rasterization for the other losses
            if cfg.rgb_loss and step >= cfg.feature_only_iters:
                renders, alphas, normals, info = self.scene.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=dyn_masks,
                    background_color=background_color if artifact_removal else None,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.rgb_loss:
                    # transform the gt images if per-image embeddings are used
                    if (
                        self.cfg.color_correct
                        and step > self.cfg.color_correct_start_iter
                    ):
                        img_m = self.color_correct_m[image_ids] + torch.eye(3).to(
                            device
                        )
                        img_b = self.color_correct_b[image_ids]
                        colors = (colors - img_b) @ img_m.transpose(1, 2)
                        colors = torch.clamp(colors, 0, 1)

                    # Masked L1 (channel aware)
                    diff = (colors - pixels).abs() * loss_masks
                    denom_px = loss_masks.sum().clamp_min(1.0) * 3.0
                    l1loss = diff.sum() / denom_px

                    # Masked SSIM (multiply images by mask)
                    colors_ssim = colors * loss_masks
                    pixels_ssim = pixels * loss_masks
                    ssimloss = 1.0 - fused_ssim(
                        colors_ssim.permute(0, 3, 1, 2),
                        pixels_ssim.permute(0, 3, 1, 2),
                        padding="valid",
                    )

                    loss = loss + l1loss * cfg.l1_lambda + ssimloss * cfg.ssim_lambda

                if cfg.depth_loss:
                    # alpha loss (masked)
                    alpha_diff = (alphas - bg_masks).abs() * loss_masks
                    alpha_loss = alpha_diff.sum() / loss_masks.sum().clamp_min(1.0)
                    loss = (
                        loss * (1.0 - cfg.alpha_lambda) + alpha_loss * cfg.alpha_lambda
                    )
                    # depth loss
                    depthloss = torch.norm(depths_gt - depths, dim=-1, p=1) / depths_gt[
                        ..., 0
                    ].clamp_min(1e-6)
                    depthloss = torch.clamp(depthloss - cfg.depth_tollerance, min=0.0)
                    depth_mask = (
                        depths_conf[..., 0]
                        * bg_masks[..., 0]
                        * dyn_masks[..., 0]
                        * loss_masks[..., 0]
                    )
                    depthloss = (
                        depthloss * depth_mask
                    ).sum() / depth_mask.sum().clamp_min(1.0)
                    loss = loss + depthloss * cfg.depth_lambda

                if (
                    cfg.normal_loss
                    and step > cfg.normal_loss_start_iter
                    and normals is not None
                ):
                    normal_loss_map = (
                        1 - F.cosine_similarity(normals, normals_gt, dim=3)
                    ) / 2.0
                    mask_norm = (
                        normals_conf[..., 0]
                        * bg_masks[..., 0]
                        * dyn_masks[..., 0]
                        * loss_masks[..., 0]
                    )
                    normal_loss = (
                        normal_loss_map * mask_norm
                    ).sum() / mask_norm.sum().clamp_min(1.0)
                    loss = loss + normal_loss * cfg.normal_lambda

                # new background artifact removal logic
                if (
                    cfg.local_alpha_reg_interval > 0
                    and step % cfg.local_alpha_reg_interval == 0
                ):
                    bg_masks = torch.ones_like(pixels[..., 0:1], device=device)
                    # render just the foreground
                    fg_renders, fg_alphas, _, _ = self.scene.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                        masks=dyn_masks,
                        background_color=[1.0, 1.0, 1.0],
                        background_mode="fg",
                    )
                    # render just the background
                    bg_renders, bg_alphas, _, _ = self.scene.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                        masks=dyn_masks,
                        background_color=None,
                        background_mode="bg",
                    )
                    # Get the L1 between the background render and the foreground render
                    bg_diff = (
                        (bg_renders - fg_renders).abs().mean(dim=-1, keepdim=True)
                    )  # [1,H,W,1]
                    # Make the target alpha mask be the difference (max normalized)
                    target_alpha = bg_diff / (bg_diff.max() + 1e-8)
                    # amplify the target alpha
                    target_alpha = target_alpha.pow(cfg.local_alpha_reg_exponent)
                    local_alpha_loss = (fg_alphas - target_alpha).clamp(min=0.0).mean()
                    loss = loss + local_alpha_loss * cfg.local_alpha_reg_lambda

                # splat parameter decay (in-place to preserve optimizer parameter objects)
                if cfg.opacity_decay > 0.0:
                    opacity_decay = cfg.opacity_decay * loss_masks.mean() * loss_weight
                    # only apply if the mean is above 0.5
                    if self.scene.splats["opacities"].mean() > cfg.target_opacity / 2.0:
                        with torch.no_grad():
                            self.scene.splats["opacities"].sub_(opacity_decay)

                if cfg.color_desaturate > 0.0 or cfg.color_reg > 0.0:
                    target = torch.as_tensor(
                        cfg.target_color,
                        device=self.scene.splats["sh0"].device,
                        dtype=self.scene.splats["sh0"].dtype,
                    )
                    # Match shape: broadcast last dim = 3; adapt if your rgb is [*, 3]
                    rgb = sh_to_rgb(self.scene.splats["sh0"])
                    target = target.view(*([1] * (rgb.ndim - 1)), 3)

                    dir_vec = target - rgb  # vector toward target
                    eps = 1e-8
                    norm = torch.linalg.norm(
                        dir_vec, ord=2, dim=-1, keepdim=True
                    ).clamp_min(eps)

                    # Nonlinear scaling on the norm (preserve direction):
                    # magnitude := norm ** alpha  (alpha ≥ 1 amplifies far ones; 0<alpha<1 compresses)
                    magnitude = norm.pow(cfg.color_desaturate_exponent)
                    dir_unit = dir_vec / norm

                    # Optional: cap step length to max_step
                    max_step = getattr(cfg, "color_desaturate_max_step", 1.0)
                    desat_step = dir_unit * torch.minimum(
                        magnitude,
                        torch.as_tensor(max_step, dtype=norm.dtype, device=norm.device),
                    )

                    desaturate_dir = desat_step

                if cfg.color_desaturate > 0.0:
                    color_desaturate = (
                        cfg.color_desaturate * loss_masks.mean() * loss_weight
                    )
                    with torch.no_grad():
                        self.scene.splats["sh0"].add_(color_desaturate * desaturate_dir)

                # regularizations
                reg_loss = 0.0
                if cfg.opacity_reg > 0.0:
                    opacity_reg = loss_masks.mean() * cfg.opacity_reg
                    # Set the target opacity to 0.5 to avoid complete disappearance of splats
                    opacities = self.scene.splats["opacities"]
                    opacity_loss = torch.abs(
                        opacities
                        - torch.tensor(
                            [cfg.target_opacity], device=opacities.device
                        ).unsqueeze(0)
                    ).mean()
                    reg_loss = reg_loss + opacity_reg * opacity_loss
                if cfg.color_reg > 0.0:
                    color_reg = loss_masks.mean() * cfg.color_reg
                    reg_loss = reg_loss + color_reg * torch.abs(desaturate_dir).mean()
                if cfg.scale_reg > 0.0:
                    scale_reg = loss_masks.mean() * cfg.scale_reg
                    reg_loss = (
                        reg_loss
                        + scale_reg
                        * torch.abs(torch.exp(self.scene.splats["scales"])).mean() ** 2
                    )
                if cfg.z_scale_reg > 0.0:
                    z_scale_reg = loss_masks.mean() * cfg.z_scale_reg
                    reg_loss = (
                        reg_loss
                        + z_scale_reg
                        * torch.abs(
                            torch.exp(self.scene.splats["scales"][..., 2])
                        ).mean()
                    )
                loss = loss + reg_loss

            if densify and info is not None:
                self.cfg.strategy.step_pre_backward(
                    params=self.scene.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            if not torch.isfinite(loss):
                print("Loss is NaN or Inf!")
                for sub_loss in [
                    l1loss,
                    ssimloss,
                    depthloss,
                    normal_loss,
                    feat_loss,
                    feat_dist_loss,
                    feat_smoothness_loss,
                    geometry_loss,
                    reg_loss,
                ]:
                    if sub_loss is not None:
                        print(f"Sub-loss value: {sub_loss.item()}")
                print("Sanitizing splat parameters and skipping step...")
                for k, v in self.scene.splats.items():
                    if torch.is_floating_point(v):
                        self.scene.splats[k] = torch.nan_to_num(
                            v, nan=0.0, posinf=0.0, neginf=0.0
                        )
                self.viewer.lock.release()
                continue

            loss = loss * loss_weight
            loss.backward()

            gaussian_cnt = self.scene.splats["means"].shape[0]

            # Update the pbar
            desc = (
                f"loss={loss.item():.3f}| "
                f"sh degree={sh_degree_to_use}| gaussians={gaussian_cnt:_} | "
            )
            if l1loss is not None:
                desc += f"l1 ={l1loss.item()*cfg.l1_lambda:.3f} | "
            if ssimloss is not None:
                desc += f"ssim ={ssimloss.item()*cfg.ssim_lambda:.3f} | "
            if feat_loss is not None:
                desc += f"feat ={feat_loss.item()*cfg.feature_lambda:.3f} | "
            if feat_dist_loss is not None:
                desc += f"feat dist ={feat_dist_loss.item()*cfg.feature_dist_regularization:.3f} | "
            if feat_smoothness_loss is not None:
                desc += f"feat smooth ={feat_smoothness_loss.item()*cfg.feature_local_smoothness:.3f} | "
            if geometry_loss is not None:
                desc += f"geometry ={geometry_loss.item()*cfg.geometry_change_penalty:.3f} | "
            if reg_loss is not None and reg_loss.item() > 0.0:
                desc += f"reg ={reg_loss.item():.3f} | "
            if depthloss is not None:
                desc += f"depth ={depthloss.item()*cfg.depth_lambda:.3f} | "
            if alpha_loss is not None:
                desc += f"alpha ={alpha_loss.item()*cfg.alpha_lambda:.3f} | "
            if local_alpha_loss is not None:
                desc += f"local alpha ={local_alpha_loss.item()*cfg.local_alpha_reg_lambda:.3f} | "
            if normal_loss is not None:
                desc += f"normal ={normal_loss.item()*cfg.normal_lambda:.3f} | "
            if cfg.pose_opt:
                # monitor the pose deviation from the original pose
                pose_err = F.l1_loss(camtoworlds_initial, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                if l1loss:
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                if ssimloss:
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if feat_loss:
                    self.writer.add_scalar("train/feat_loss", feat_loss.item(), step)
                if feat_dist_loss:
                    self.writer.add_scalar(
                        "train/feat_dist_loss", feat_dist_loss.item(), step
                    )
                if feat_smoothness_loss:
                    self.writer.add_scalar(
                        "train/feat_smoothness_loss",
                        feat_smoothness_loss.item(),
                        step,
                    )
                if geometry_loss:
                    self.writer.add_scalar(
                        "train/geometry_loss", geometry_loss.item(), step
                    )
                if depthloss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                    self.writer.add_scalar("train/alpha_loss", alpha_loss.item(), step)
                if normal_loss:
                    self.writer.add_scalar(
                        "train/normal_loss", normal_loss.item(), step
                    )
                if reg_loss:
                    self.writer.add_scalar("train/reg_loss", reg_loss.item(), step)
                if cfg.pose_opt:
                    self.writer.add_scalar("train/pose_err", pose_err.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.scene.splats["means"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:

                if self.active_sample_gens is None:
                    name = f"step{step:04d}"
                else:
                    name = f"gen{''.join(map(str, self.active_sample_gens))}_step{step:04d}"

                mem = torch.cuda.max_memory_allocated() / 1024**3
                info = {
                    "step": step,
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.scene.splats["means"]),
                }
                if cfg.pose_opt:
                    info["pose_adjust"] = self.pose_adjust.state_dict()
                self.last_ckpt_path = f"{self.cfg.ckpt_dir}/{name}.pt"
                self.scene.save_ckpt(self.last_ckpt_path, info=info)
                print(f"Saved checkpoint at step {step} to path {self.last_ckpt_path}.")
            if cfg.save_ply and (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ):
                save_ply(self.scene.splats, f"{self.cfg.ply_dir}/{name}.ply")

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.scene.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step > cfg.pose_opt_start_iter:
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            # background model
            if cfg.optimize_background:
                if self.scene.background_model is not None:
                    self.scene.background_optimizer.step()
                    self.scene.background_optimizer.zero_grad(set_to_none=True)
                if self.scene.feature_bg_model is not None:
                    self.scene.feature_bg_optimizer.step()
                    self.scene.feature_bg_optimizer.zero_grad(set_to_none=True)
            # color correction
            if self.cfg.color_correct:
                self.color_correct_optimizer.step()
                self.color_correct_optimizer.zero_grad(set_to_none=True)
            # lr schedulers
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer (densification)
            if densify:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.scene.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                        prune_only=getattr(cfg, "prune_only", False),
                    )
                else:
                    assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
            if step in [i - 1 for i in cfg.save_steps]:
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            # Update the viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step, gaussian_cnt)

            if self.viewer is not None and self.viewer.cancel:
                print("Training cancelled by the viewer.")
                break

            if step > max_steps:
                break

        return self.last_ckpt_path

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluation that mirrors train(): computes feature, RGB, depth, normal, alpha, and feature-dist losses when possible."""

        if self.data_mngr.eval_dataset is None or len(self.data_mngr.eval_dataset) == 0:
            print("No evaluation dataset, skipping eval...")
            return

        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        ellipse_time = 0.0
        metrics = defaultdict(list)
        len_eval = len(self.data_mngr.eval_dataset)

        for i in range(len_eval):
            # ---- Fetch batch (same structure as train) ----
            data = self.data_mngr.next_eval(step=1)
            cams, data = data

            camtoworlds = convert_to_4x4(cams.camera_to_worlds.to(device))  # [1,4,4]
            Ks = []
            for cam in cams:
                K = torch.tensor(
                    [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]],
                    device=device,
                    dtype=torch.float32,
                )
                Ks.append(K)
            Ks = torch.stack(Ks, dim=0)  # [n,3,3]

            pixels = data["image"].to(device)  # [H,W,3] in same scale as train
            pixels = pixels.unsqueeze(0)  # [1,H,W,3]
            image_ids = torch.tensor(data.get("image_idx", 0)).unsqueeze(0).to(device)

            # Masks (dyn/bg). If missing, default to ones like train does.
            dyn_masks = (
                data["dyn_mask"].unsqueeze(0).to(device)
                if "dyn_mask" in data
                else torch.ones_like(pixels[..., 0:1], device=device)
            )  # [1,H,W,1]
            bg_masks = (
                data["bg_masks"].unsqueeze(0).to(device)
                if "bg_masks" in data
                else torch.ones_like(pixels[..., 0:1], device=device)
            )  # [1,H,W,1]
            if cfg.use_loss_mask and "loss_mask" in data:
                loss_masks = data["loss_mask"].unsqueeze(0).to(device)
            else:
                loss_masks = torch.ones_like(pixels[..., 0:1], device=device)

            # Depth & normals (optional)
            have_depth_normals = (
                ("depth" in data and "normals" in data)
                if cfg.depth_loss or cfg.normal_loss
                else False
            )
            if have_depth_normals:
                depths_gt = data["depth"].unsqueeze(0).to(device)  # [1,H,W,1]
                depths_conf = (
                    data["depth_confidence"].unsqueeze(0).to(device)
                )  # [1,H,W,1]
                normals_gt = data["normals"].unsqueeze(0).to(device)  # [1,H,W,3]
                normals_conf = (
                    data["normal_confidence"].unsqueeze(0).to(device)
                )  # [1,H,W,1]

            # Semantic features (optional)
            have_sem_feats = (
                cfg.feature_loss
                and cfg.num_semantic_features > 0
                and "reduced_embeddings" in data
            )
            if have_sem_feats:
                sem_features = data["reduced_embeddings"].unsqueeze(0).to(device)
                sem_features = sem_features[
                    ..., : cfg.num_semantic_features
                ]  # [1,Hf,Wf,C]

            height, width = pixels.shape[1:3]

            # ---- Forward passes ----
            tic = time.time()

            total_loss = torch.tensor(0.0, device=device)
            l1loss = ssimloss = depthloss = alpha_loss = normal_loss = feat_loss = (
                feat_dist_loss
            ) = None

            # --- Feature-space rasterization & loss (same as train) ---
            if have_sem_feats:
                _, feat_H, feat_W, _ = sem_features.shape
                scale_x = feat_W / width
                scale_y = feat_H / height
                Ks_feat = resize_intrinsics(K=Ks, scale=(scale_x, scale_y))
                dyn_masks_feat = F.interpolate(
                    dyn_masks.permute(0, 3, 1, 2), size=(feat_H, feat_W), mode="nearest"
                ).permute(0, 2, 3, 1)
                loss_masks_feat = F.interpolate(
                    loss_masks.permute(0, 3, 1, 2),
                    size=(feat_H, feat_W),
                    mode="nearest",
                ).permute(0, 2, 3, 1)
                feat_renders, _, _, _ = self.scene.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks_feat,
                    width=feat_W,
                    height=feat_H,
                    sh_degree=None,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB",
                    color_mode="Features",
                    masks=dyn_masks_feat,
                )
                weights = gaussian_decay_weights(
                    feat_renders.shape[-1],
                    1.0,
                    0.05,
                    sigma_ratio=0.1,
                    device=feat_renders.device,
                    dtype=feat_renders.dtype,
                )
                feat_renders_w = feat_renders * weights[None, None, None, :]
                sem_features_w = sem_features * weights[None, None, None, :]
                feat_loss_map = 1 - F.cosine_similarity(
                    feat_renders_w, sem_features_w, dim=-1
                )
                feat_loss_map += (feat_renders_w - sem_features_w).abs().mean(dim=-1)
                mask_feat = (dyn_masks_feat[..., 0] * loss_masks_feat[..., 0]).float()
                feat_loss = (
                    feat_loss_map * mask_feat
                ).sum() / mask_feat.sum().clamp_min(1.0)
                total_loss = total_loss + cfg.feature_lambda * feat_loss

            # --- Feature distance regularization ---
            if (
                cfg.feature_dist_regularization > 0.0
                and getattr(self, "feature_dist_loss_module", None) is not None
            ):
                feat_dist_loss, _ = self.feature_dist_loss_module(
                    self.scene.splats["features"]
                )
                total_loss = (
                    total_loss + cfg.feature_dist_regularization * feat_dist_loss
                )

            # --- Image-space rasterization & losses (same gating as train) ---
            if step > cfg.feature_only_iters:
                # Render RGB (+ depth if enabled) for loss computation
                renders, alphas, normals, _ = self.scene.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=min(
                        cfg.sh_degree,
                        (
                            step // cfg.sh_degree_interval
                            if cfg.sh_degree_interval
                            else cfg.sh_degree
                        ),
                    ),
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=dyn_masks,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                # Optional color correction (same condition as train)
                if cfg.rgb_loss:
                    colors_corr = colors
                    if cfg.color_correct and step > cfg.color_correct_start_iter:
                        img_m = self.color_correct_m[image_ids] + torch.eye(
                            3, device=device
                        )
                        img_b = self.color_correct_b[image_ids]
                        colors_corr = (colors_corr - img_b) @ img_m.transpose(1, 2)
                        colors_corr = torch.clamp(colors_corr, 0, 1)
                # Masked L1 (channel aware)
                diff = (colors_corr - pixels).abs() * loss_masks
                denom_px = loss_masks.sum().clamp_min(1.0) * 3.0
                l1loss = diff.sum() / denom_px

                # Masked SSIM (multiply images by mask)
                colors_ssim = colors_corr * loss_masks
                pixels_ssim = pixels * loss_masks
                ssimloss = 1.0 - fused_ssim(
                    colors_ssim.permute(0, 3, 1, 2),
                    pixels_ssim.permute(0, 3, 1, 2),
                    padding="valid",
                )

                total_loss = (
                    total_loss + cfg.l1_lambda * l1loss + cfg.ssim_lambda * ssimloss
                )

                # Depth + alpha loss
                if cfg.alpha_loss and bg_masks is not None:
                    alpha_diff = (alphas - bg_masks).abs() * loss_masks
                    alpha_loss = alpha_diff.sum() / loss_masks.sum().clamp_min(1.0)
                    total_loss = (
                        total_loss * (1.0 - cfg.alpha_lambda)
                        + cfg.alpha_lambda * alpha_loss
                    )
                if cfg.depth_loss and depths is not None and have_depth_normals:
                    depthloss = torch.norm(depths_gt - depths, dim=-1, p=1) / (
                        depths_gt[..., 0].clamp_min(1e-6)
                    )
                    depthloss = torch.clamp(depthloss - cfg.depth_tollerance, min=0.0)
                    depth_mask = (
                        depths_conf[..., 0]
                        * bg_masks[..., 0]
                        * dyn_masks[..., 0]
                        * loss_masks[..., 0]
                    )
                    depthloss = (
                        depthloss * depth_mask
                    ).sum() / depth_mask.sum().clamp_min(1.0)
                    total_loss = total_loss + cfg.depth_lambda * depthloss

                # Normal loss
                if (
                    cfg.normal_loss
                    and step > cfg.normal_loss_start_iter
                    and normals is not None
                    and have_depth_normals
                ):
                    normal_loss_map = (
                        1.0 - F.cosine_similarity(normals, normals_gt, dim=3)
                    ) * 0.5
                    normal_mask = (
                        normals_conf[..., 0]
                        * bg_masks[..., 0]
                        * dyn_masks[..., 0]
                        * loss_masks[..., 0]
                    )
                    normal_loss = (
                        normal_loss_map * normal_mask
                    ).sum() / normal_mask.sum().clamp_min(1.0)
                    total_loss = total_loss + cfg.normal_lambda * normal_loss

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # ---- Save renders & compute image metrics (PSNR/SSIM/LPIPS) when we have colors) ----
            # For visualization/metrics we render RGB regardless of gating if not already done.
            if step <= cfg.feature_only_iters or (cfg.depth_loss and depths is None):
                colors_vis, _, _, _ = self.scene.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=None,
                )
            else:
                colors_vis = colors

            colors_vis = torch.clamp(colors_vis, 0.0, 1.0)
            canvas = (
                torch.cat([pixels, colors_vis], dim=2).squeeze(0).detach().cpu().numpy()
            )
            canvas = (canvas * 255.0).astype(np.uint8)
            # os.makedirs(self.cfg.render_dir, exist_ok=True)
            imageio.imwrite(
                f"{self.cfg.render_dir}/{stage}_step{step}_{i:04d}.png", canvas
            )

            # Metrics expect CHW in [0,1]
            px = pixels.permute(0, 3, 1, 2)
            cx = colors_vis.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(cx, px))
            metrics["ssim"].append(self.ssim(cx, px))
            metrics["lpips"].append(self.lpips(cx, px))

            # Per-loss logging (append only if computed)
            if l1loss is not None:
                metrics["l1loss"].append(l1loss)
            if ssimloss is not None:
                metrics["ssimloss"].append(ssimloss)
            if feat_loss is not None:
                metrics["feat_loss"].append(feat_loss)
            if feat_dist_loss is not None:
                metrics["feat_dist_loss"].append(feat_dist_loss)
            if depthloss is not None:
                metrics["depthloss"].append(depthloss)
            if alpha_loss is not None:
                metrics["alpha_loss"].append(alpha_loss)
            if normal_loss is not None:
                metrics["normal_loss"].append(normal_loss)
            metrics["loss"].append(total_loss)

        # ---- Aggregate + write stats ----
        if len_eval > 0:
            ellipse_time /= len_eval
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.scene.splats["means"]),
                }
            )

            print(
                f"PSNR: {stats.get('psnr', 0):.3f}, SSIM: {stats.get('ssim', 0):.4f}, LPIPS: {stats.get('lpips', 0):.3f}, Feature Cosine: {stats.get('feat_loss', 0):.3f}, "
                f"Time: {stats['ellipse_time']:.3f}s/image  GS: {stats['num_GS']}"
            )

            # Save json
            with open(f"{self.cfg.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)

            # TensorBoard
            if not self.eval_mode:
                for k, v in stats.items():
                    self.writer.add_scalar(f"{stage}/{k}", v, step)
                self.writer.flush()

        return stats

    def train_sds(
        self,
        sample_gen,  # SampleGenerator
        *,
        steps: int = 5000,
        batch_size: int = 2,
        prompt: str = "",
        camera_sample_strategy: Literal[
            "random", "road", "perturb", "original"
        ] = "original",
        rotation_std: float = 0.2,
        translation_std: float = 0.2,
        feature_ctrl_strength: float = 0.7,
        feature_ctrl_dims: int = 64,
        text_ctrl_strength: float = 3.5,
        weight_by_sigma: bool = False,
        densify: bool = True,
        log_every: int = 100,
        save_every: int = 1000,
        start_noise_frac: float = 0.9,
        end_noise_frac: float = 0.5,
    ) -> None:
        """
        Pure-SDS training loop for Gaussian splats using a diffusion prior via ControlNet+FLUX.

        - Only the SDS loss is applied (no RGB/depth/feature reconstruction terms).
        - Densification can be toggled with `densify`.
        - Uses SampleGenerator for ControlNet/FLUX plumbing and feature rendering.

        Assumes:
          - self.scene.splats and self.optimizers are initialized.
          - self.cfg.result_dir, tb writer, and (optionally) viewer exist.
        """
        device = torch.device(self.device)
        global_step0 = 0  # you can load/resume if desired
        pbar = tqdm.trange(
            global_step0, global_step0 + steps, desc="SDS Training", disable=None
        )

        # Short aliases
        H, W = sample_gen.H, sample_gen.W
        fov = sample_gen.fov

        # Keep a simple LR schedule on "means" like your main loop, optional
        schedulers = []
        if "means" in self.optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.01 ** (1.0 / steps)
                )
            )

        for step in pbar:
            if step > global_step0 + steps - 1:
                break

            if not self.cfg.disable_viewer:
                # allow pause in viewer if you want parity with main loop
                while self.viewer is not None and self.viewer.state.status == "paused":
                    time.sleep(0.01)
                lock = (
                    getattr(self, "viewer", None).lock
                    if self.viewer is not None
                    else contextlib.nullcontext()
                )
            else:
                lock = contextlib.nullcontext()

            with lock:
                tic = time.time()

                # 1) Linear noise schedule
                noise_frac = start_noise_frac + (end_noise_frac - start_noise_frac) * (
                    step / max(steps - 1, 1)
                )

                sds_loss, info = sample_gen.batch_sds_loss(
                    batch_size=batch_size,
                    prompts=prompt,
                    noise_frac=noise_frac,
                    camera_sample_strategy=camera_sample_strategy,
                    rotation_std=rotation_std,
                    translation_std=translation_std,
                    feature_ctrl_strength=feature_ctrl_strength,
                    feature_ctrl_dims=feature_ctrl_dims,
                    text_ctrl_strength=text_ctrl_strength,
                    weight_by_sigma=weight_by_sigma,
                )

                if densify and isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_pre_backward(
                        params=self.scene.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )

                loss_val = float(sds_loss.item())
                pbar.set_postfix(
                    {
                        "sds": f"{loss_val:.4f}",
                        "GS": f"{len(self.scene.splats['means']):_}",
                        "noise": f"{noise_frac:.3f}",
                    }
                )

                # 3) Backprop into splats only
                self.scene.splats.zero_grad(set_to_none=True)
                sds_loss.backward()

                # 4) Optimizer steps
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # 5) Optional densification pass (keep conservative during pure SDS)
                if densify and isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.scene.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=self.cfg.packed,
                    )

                # 6) Lightweight logging
                if step % log_every == 0:
                    mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
                    self.writer.add_scalar("sds/loss", loss_val, step)
                    self.writer.add_scalar(
                        "sds/num_GS", len(self.scene.splats["means"]), step
                    )
                    self.writer.add_scalar("sds/mem_gb", mem_gb, step)
                    self.writer.flush()

                # 7) Periodic checkpoint
                if save_every > 0 and step % save_every == 0:
                    name = f"sds_step{step:06d}"
                    stats = {
                        "loss": loss_val,
                        "num_GS": len(self.scene.splats["means"]),
                        "elapsed_s": time.time() - tic,
                    }
                    os.makedirs(self.cfg.stats_dir, exist_ok=True)
                    os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
                    with open(f"{self.cfg.stats_dir}/{name}.json", "w") as f:
                        json.dump(stats, f)
                    self.last_ckpt_path = f"{self.cfg.ckpt_dir}/{name}.pt"
                    torch.save(
                        {"step": step, "splats": self.scene.splats.state_dict()},
                        self.last_ckpt_path,
                    )
                    print(
                        f"SDS Checkpoint saved to {self.last_ckpt_path} at step {step}."
                    )

                # 9) Scheduler step(s)
                for sch in schedulers:
                    sch.step()

                # 10) Update viewer throughput
                if not self.cfg.disable_viewer and self.viewer is not None:
                    # number here is symbolic (we don't count rays in SDS block)
                    self.viewer.update(
                        step,
                        num_train_rays_per_step=0,
                        gaussian_count=len(self.scene.splats["means"]),
                    )

                if self.viewer is not None and self.viewer.cancel:
                    print("SDS Training cancelled by the viewer.")
                    break

                return self.last_ckpt_path

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if hasattr(self.parser, "camera_poses_front"):
            # Use the front camera poses for trajectory rendering
            camtoworlds_all = self.parser.camera_poses_front
        else:
            print("Did not find front camera poses, skipping trajectory rendering.")
            return

        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 10
            )  # [N, 3, 4]
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        camtoworlds_all = convert_to_4x4(camtoworlds_all)

        K = self.train_cam_ks[0].float().to(device)
        height, width = (
            self.train_cam_dims[0, 0].item(),
            self.train_cam_dims[0, 1].item(),
        )

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(
            f"{video_dir}/traj_{step}.mp4", fps=30, format="FFMPEG"
        )
        for i in tqdm.trange(
            len(camtoworlds_all), desc="Rendering trajectory", disable=None
        ):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _, _ = self.scene.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths_colored = self.scene.depth_colormapper.apply_colormap(
                depths, vmax=100.0
            )  # [1, H, W, 3]
            canvas_list = [colors, depths_colored]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")

        compress_dir = f"{cfg.result_dir}/compression/"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.scene.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.scene.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        runner.scene.load_from_ckpt(cfg.ckpt)
        step = cfg.ckpt.split("_")[-1].split(".")[0]
        step = int(step)
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
