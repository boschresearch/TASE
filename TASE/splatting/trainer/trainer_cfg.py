# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from gsplat V 1.5.3
#   (https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py)
# Copyright 2025 Nerfstudio Team, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Literal
from typing_extensions import assert_never
from pathlib import Path
from TASE.splatting.trainer.custom_strategy import DefaultStrategy
import json
import yaml
import os

LEGACY_STRATEGY_TAG = (
    "!!python/object:TASE.splatting.trainer.custom_strategy.DefaultStrategy"
)


@dataclass
class Config:
    # ---------- General ----------
    splat_dtype: Literal["float32", "bfloat16"] = "float32"
    disable_viewer: bool = False
    keep_viewer: bool = False
    tb_server: bool = False
    viewer_pose: Optional[str] = None
    store_train_samples: bool = False
    generate_depthmap_pc: bool = False
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"

    # Dataset & I/O
    data_dir: Optional[str] = None  # <-- keep only this one
    local_data_dir: Optional[str] = None
    data_factor: int = 4
    result_dir: Optional[str] = None

    ckpt_dir_name: str = "ckpts"
    ply_dir_name: str = "ply"
    tb_dir_name: str = "tb"
    stats_dir_name: str = "stats"
    render_dir_name: str = "renders"

    ckpt_dir: Optional[str] = None
    ply_dir: Optional[str] = None
    tb_dir: Optional[str] = None
    stats_dir: Optional[str] = None
    render_dir: Optional[str] = None

    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0

    # Training schedule
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Init
    init_type: Literal["sfm", "random", "depth", "lidar", "sem_pc"] = "sem_pc"
    semantic_colormap: Optional[str] = None  # safer; fill at runtime if needed
    init_num_pts: int = 100_000
    init_extent: float = 1.5
    init_size: float = -1.0
    sh_degree: int = 2
    num_semantic_features: int = 64
    sh_degree_interval: int = 2000
    init_opa: float = 0.1
    init_scale: float = 1.0
    max_init_size: Optional[float] = None

    # Losses
    ssim_lambda: float = 0.2
    l1_lambda: float = 1.0
    near_plane: float = 0.01
    far_plane: float = 1e10

    use_loss_mask: bool = False  # Only applied where loss masks are available
    use_bg_mask: bool = True  # Only applied where bg masks are available
    use_dyn_mask: bool = True  # Only applied where dynamic masks are available

    # Strategy (serialize by name)
    strategy_name: Literal["default"] = "default"
    strategy: Optional[Union["DefaultStrategy"]] = field(
        default=None, repr=False, compare=False
    )

    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = True

    opacity_reg: float = 0.0
    color_reg: float = 0.0
    scale_reg: float = 0.0
    z_scale_reg: float = 0.0

    opacity_decay: float = 0.0  # new
    color_desaturate: float = 0.0  # new

    target_opacity: float = 1.0  # new
    target_color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])  # new

    # Pose optimization
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_opt_start_iter: int = 0
    pose_noise: float = 0.0  # test only

    # Color correction
    color_correct: bool = False
    color_correct_lr: float = 1e-6
    color_correct_reg: float = 1e-7
    color_correct_start_iter: int = 15_000  # <-- now a field

    # Depth / feature / normal losses
    depth_loss: bool = False
    depth_lambda: float = 1.0
    depth_lambda_min: float = 5e-3
    depth_lambda_decay: float = 10_000
    depth_tollerance: float = 0.05

    rgb_loss: bool = True
    feature_loss: bool = True
    feature_lambda: float = 1.0
    feature_only_iters: int = 0

    feature_dist_regularization: float = 0
    feature_dist_loss_interval: int = 50

    feature_local_smoothness: float = 0
    feature_smoothness_loss_interval: int = 10
    feature_knn_recompute_max_steps: int = 100
    feature_knn_K: int = 5
    feature_knn_max_dist: float = 0.05

    normal_loss: bool = False
    normal_lambda: float = 0.1
    normal_loss_start_iter: int = 10_000

    alpha_loss: bool = False
    alpha_lambda: float = 1.0
    local_alpha_reg_interval: int = -1
    local_alpha_reg_lambda: float = 0.1
    local_alpha_reg_exponent: float = 2.0

    prune_only: bool = False

    # Logging
    tb_every: int = 100
    tb_save_image: bool = False
    lpips_net: Literal["vgg", "alex"] = "alex"

    # LRs
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "means": 1e-3,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 1e-1,
            "sh0": 5e-3,
            "shN": 2.5e-3 / 20,
            "features": 2.5e-3,
            "classes": 2.5e-3,
            "background": 1e-5,
            "feature_bg": 1e-5,
        }
    )
    geometry_change_penalty: float = None  # disabled by default
    geometry_change_exponent: float = 4.0
    color_desaturate_exponent: float = 2.0

    # Dataset type & extras
    dataset_type: Optional[Literal["colmap", "nuscenes", "TASE"]] = "TASE"
    scene: str = "unnamed"  # <-- now a field
    mask_dir: Optional[str] = None  # <-- now a field
    background_model: Optional[Literal["mlp"]] = None
    optimize_background: bool = False  # <-- now a field
    show_pcs: bool = True  # <-- now a field

    def set_paths(self):
        for attr in ["ckpt_dir", "ply_dir", "tb_dir", "stats_dir", "render_dir"]:
            dir_name = getattr(self, attr + "_name")
            if self.result_dir is None:
                raise ValueError("result_dir must be set before calling set_paths()")
            full_path = str(Path(self.result_dir) / dir_name)
            os.makedirs(full_path, exist_ok=True)
            setattr(self, attr, full_path)

    def __post_init__(self):
        # Fill semantic colormap default lazily (portable default)
        if self.semantic_colormap is None:
            self.semantic_colormap = "util/nuscenes_ctrl_cmap.json"

        # Construct strategy from name or from a dict if loaded from YAML/JSON
        if self.strategy is None or isinstance(self.strategy, dict):
            if self.strategy_name == "default":
                strategy = DefaultStrategy()
            else:
                print(
                    f"[WARNING] Unknown strategy_name: {self.strategy_name}, using default strategy."
                )
                strategy = DefaultStrategy()

            if isinstance(self.strategy, dict):
                # If loaded from dict, update fields
                for k, v in self.strategy.items():
                    if hasattr(strategy, k):
                        setattr(strategy, k, v)
                    else:
                        print(
                            f"[WARNING] Unknown field '{k}' for strategy '{self.strategy_name}'"
                        )

            self.strategy = strategy

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        s = self.strategy
        if isinstance(s, DefaultStrategy):
            s.refine_start_iter = int(s.refine_start_iter * factor)
            s.refine_stop_iter = int(s.refine_stop_iter * factor)
            s.reset_every = int(s.reset_every * factor)
            s.refine_every = int(s.refine_every * factor)
        else:
            assert_never(s)  # type: ignore

    # ---------- Serialization helpers ----------
    def to_dict(self) -> Dict:
        d = asdict(self)
        # drop non-serializable runtime objects
        d.pop("strategy", None)
        # keep strategy_name (cheap string) to reconstruct on load
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        # create without runtime-only objects, they’ll be reconstructed in __post_init__
        return cls(**d)

    def to_json(self, path: Union[str, Path]):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_yaml(self, path: Union[str, Path]):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        if not path.is_file():
            # just filename, look in default configs dir
            path = Path(__file__).parent.parent / "configs" / path
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            text = f.read()
        if LEGACY_STRATEGY_TAG in text:
            # strip unsafe tag; keep mapping
            text = text.replace(LEGACY_STRATEGY_TAG, "")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def copy_yaml(cls, src: Union[str, Path], dst: Union[str, Path]):
        src = Path(src)
        if not src.is_file():
            # just filename, look in default configs dir
            src = Path(__file__).parent.parent / "configs" / src
        if not src.is_file():
            print(f"[ERROR] Config file not found: {src}, could not copy.")
            return
        with open(src, "r") as f_src:
            text = f_src.read()
        with open(dst, "w") as f_dst:
            f_dst.write(text)
