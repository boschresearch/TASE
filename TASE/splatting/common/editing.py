# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import shutil
from pathlib import Path

# Forward references to your types
"""
from TASE.splatting.common.rendering import Scene
from TASE.splatting.trainer.TASE_dataparser import TASE
from TASE.splatting.trainer.trainer import Runner
from TASE.splatting.viewer.viewer import Viewer
"""
import traceback

from TASE.splatting.common.generative_helper import SampleGenerator
from TASE.splatting.common.rendering import Scene
from TASE.splatting.trainer.TASE_dataparser import TASE
from TASE.splatting.viewer.viewer import Viewer
from TASE.splatting.trainer.trainer import Runner, Config as RunnerConfig
from TASE.splatting.common.rendering import _load_ckpt_and_dataset
from TASE.splatting.util.data_util import resolve_path

import gc


def list_cuda_tensors(top=25):
    entries = []
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            bytes_ = obj.nelement() * obj.element_size()
            entries.append((bytes_, obj))
    entries.sort(key=lambda x: x[0], reverse=True)
    total = sum(b for b, _ in entries)
    for b, t in entries[:top]:
        print(
            f"{b/1024**2:7.2f} MB | shape={tuple(t.shape)} dtype={t.dtype} grad={t.requires_grad}"
        )
    print(f"Total (tracked this way): {total/1024**2:.2f} MB")


class SceneEditingModule:
    """
    Scene Editing (Global & Local) — UI-driven module.

    Goals:
      - Global edits: regenerate frames for a pose set (start: original poses).
      - Local edits: regenerate only selected regions (from segmentation selection),
        and restrict subsequent optimization to the selected Gaussians.
      - Separate buttons for sample generation and scene optimization.

    Data contracts:
      - `data["means"]      : (N,3) float32 point positions`
      - `data["features"]   : (N,C) float32 per-point features (unit norm)`
      - `data["selection"]  : (N,1) float32 in {-1,0,1}`  (from SegmentationModule3D)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        server,
        viewer: Viewer,
        scene: Scene,
        parser: TASE,
        runner_config: Path | str,
        sample_gen: Optional[SampleGenerator] = None,
        old_data_dir: Optional[str] = None,
        new_data_dir: Optional[str] = None,
        reconstruction_dir: Optional[str] = None,
        *,
        preset: Optional[str] = None,
        name: Optional[str] = None,
        flip_z: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.server = server
        self.viewer = viewer
        self.scene = scene
        self.parser = parser
        self.device = device
        self.dtype = dtype

        # Reuse existing sample generator if provided
        self.sample_gen = sample_gen or SampleGenerator(
            scene=scene, parser=parser, device=device, dtype=dtype
        )
        self.reconstruction_dir = reconstruction_dir
        self.name = name
        self.flip_z = flip_z    

        # An optional list of presets to apply on init
        if preset is None:
            default_preset = (
                Path(__file__).parent.parent / "configs" / "base_parameters.json"
            )
            self.preset = str(default_preset)
        else:
            self.preset = preset

        # UI handles
        self.ui: Dict[str, object] = {}  # UI controls by key, populated in attach()

        # Runtime state (UI + runtime config) -> Have to be set before trying to load a preset!
        self.auto_presets: List[str] = []
        self.state: Dict[str, object] = {}
        self.loaded_datadir = old_data_dir
        self.scene: Scene = None

        self.load_preset(self.preset)
        self.state.update(
            {
                "status_action": "",
                "status_phase": "",
                "status_detail": "",
                "log_truncate": 180,
                "last_log": "",
                "busy": False,
                "cancel": False,
            }
        )

        if runner_config is not None:
            self.state["runner_config"] = str(runner_config)

        # Working dir
        self.editing_datadir = (
            str(new_data_dir)
            + f"/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}"
        ) if self.name is None else str(new_data_dir) + f"/{self.name}"
        self.result_dir = self.editing_datadir
        self.state["ckpt_path"] = str(scene.ckpt_dir) if scene is not None else ""
        self.state["old_ckpt_path"] = str(scene.ckpt_dir) if scene is not None else ""
        self.gen_cnt = 0

        # Simple caches
        self._splat_cache: Dict[str, torch.Tensor] = (
            None  # original splat attributes for local editing
        )
        self._cache_edited_splats: Dict[str, torch.Tensor] = (
            None  # edited splat attributes for local editing
        )
        self._sampled_c2ws: List[torch.Tensor] = []  # sampled camera-to-world matrices
        self._sampled_cam_names: List[str] = []  # sampled camera names
        self._bounding_elipsoid: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None  # (center, axes) of bounding ellipsoid
        )
        self._viewer_bounding_elipsoid = (
            None  # Viser handle to the bounding ellipsoid in the viewer
        )

        # Synch the sample generator intrinsics
        self.sample_gen.update_intrinsics(
            height=self.state["height"],
            width=self.state["width"],
            fov=self.state["fov"],
        )

    # ------------------------------------------------------------------ #
    # UI wiring
    # ------------------------------------------------------------------ #
    def _bind(self, ui_key: str, state_key: str, cast=lambda x: x, post=None):
        """Bind a control to a state key with a simple on_update."""
        if ui_key not in self.ui:
            return
        ctrl = self.ui[ui_key]

        @ctrl.on_update
        def _(e):
            self.state[state_key] = cast(e.target.value)
            if post:
                post()
            # keep UI in sync if e.g. clamped/sanitized later
            self._update_ui_visibility()

    def _set_visible(self, keys: List[str], visible: bool):
        for k in keys:
            if k in self.ui and hasattr(self.ui[k], "visible"):
                self.ui[k].visible = bool(visible)

    def _sync_value(self, ui_key: str, value):
        if ui_key in self.ui and hasattr(self.ui[ui_key], "value"):
            self.ui[ui_key].value = value

    def _u_key(self, state_key: str) -> str:
        # UI controls bound to state live under "s:<state_key>"
        return f"s:{state_key}"

    def _infer_cast_from_state(self, state_key: str):
        # Use current state value type as the source of truth
        val = self.state.get(state_key, "")
        if isinstance(val, bool):
            return bool
        if isinstance(val, int):
            return int
        if isinstance(val, float):
            return float
        return str  # default

    def _post_update_intrinsics(self):
        if self.sample_gen:
            self.sample_gen.update_intrinsics(
                height=int(self.state["height"]),
                width=int(self.state["width"]),
                fov=float(self.state["fov"]),
            )

    def _auto_bind_all(self):
        """
        Auto-bind every UI control whose key starts with 's:' to the matching state key.
        Allows optional post-hooks per state key.
        """
        post_hooks = {
            # update selected splats if local edit params change
            "dilate_3d": self.modify_selected_splats,
            "perturb_3d": self.modify_selected_splats,
            "random_gaussian_init": self.modify_selected_splats,
            # switch between edited and OG splats
            "show_original": self.show_original_splats,
            # keep intrinsics in sync
            "height": self._post_update_intrinsics,
            "width": self._post_update_intrinsics,
            "fov": self._post_update_intrinsics,
        }

        for ui_key, ctrl in self.ui.items():
            if not ui_key.startswith("s:"):
                continue
            state_key = ui_key.split(":", 1)[1]
            cast = self._infer_cast_from_state(state_key)
            post = post_hooks.get(state_key)
            self._bind(ui_key, state_key, cast, post)

    # ---- Action guard helpers ----------------------------------------- #
    def _begin_action(self, label: str):
        if self.state.get("busy", False):
            self._log(
                f"Busy. '{label}' skipped. Stop or wait for current job to finish."
            )
            # Reflect that we're occupied
            self._set_status(
                action="Busy", phase="occupied", detail=f"({label} skipped)"
            )
            return False
        self.state["cancel"] = False
        self.viewer.cancel = False
        self.state["busy"] = True
        self._set_status(action=label)
        return True

    def _end_action(self, _label: str):
        self.state["busy"] = False
        self.state["cancel"] = False
        self.viewer.cancel = False
        self._set_status(action="Ready", phase="", detail="")

    def _run_action(self, label: str, fn):
        if not self._begin_action(label):
            return
        try:
            fn()
        except Exception as ex:
            tb_str = traceback.format_exc()
            self._log(f"{label} failed: {ex}\n{tb_str}")
            self._set_status(action=label, phase="error", detail=str(ex))
            self.viewer.cancel = False
        finally:
            self._end_action(label)

    def _request_stop(self):
        # clear the auto queue
        self.auto_presets = []
        if not self.state.get("busy", False):
            self._log("Not busy. Nothing to stop.")
            return
        if self.state.get("cancel", False):
            self._log("Stop already requested. Escalating.")
            self._set_status(action="Stopping", phase="escalating")
            raise RuntimeError("Force stop requested.")
            # NOTE: This does not work yet
        self.state["cancel"] = True
        self.viewer.cancel = True
        self._set_status(action="Stopping", phase="requested")
        self._log("Stop requested…")

    def _dump_params(self) -> None:
        self._log(f"Current state/params:\n{self.state}")
        params_name = "editing_params.json"
        params_path = Path(self.result_dir)
        for version in range(100):
            if (params_path / params_name).exists():
                params_name = f"editing_params_v{version}.json"
            else:
                break
        try:
            os.makedirs(params_path, exist_ok=True)
            with open((params_path / params_name), "w") as f:
                json.dump(self.state, f, indent=2)
            self._log(f"Dumped params to {params_path / params_name}")
        except Exception as ex:
            self._log(f"Failed to dump params to {params_path / params_name}: {ex}")

    # ---- Attach the UI ------------------------------------------ #
    def attach(self) -> None:
        server = self.server

        with server.gui.add_folder("Scene Editing", expand_by_default=False) as f:
            # Status
            self.ui["status_text"] = server.gui.add_markdown(
                content="Status: **Ready**"
            )
            self.ui["last_log_text"] = server.gui.add_markdown(content="")

            # -------- Workflow --------
            self.ui[self._u_key("mode")] = server.gui.add_dropdown(
                "Edit mode",
                options=["global", "local"],
                initial_value=self.state["mode"],
            )
            self.ui[self._u_key("control_mode")] = server.gui.add_dropdown(
                "Control mode",
                options=["manual", "auto"],
                initial_value=self.state["control_mode"],
            )
            self.ui[self._u_key("ckpt_path")] = server.gui.add_text(
                "Scene or preset", initial_value=str(self.state["ckpt_path"])
            )
            self.ui[self._u_key("show_original")] = server.gui.add_checkbox(
                "Show original", bool(self.state["show_original"])
            )
            self.ui[self._u_key("debug")] = server.gui.add_checkbox(
                "Debug mode", bool(self.state["debug"])
            )
            self.ui["stop_btn"] = server.gui.add_button("Stop", color=(150, 100, 100))
            self.ui["load_btn"] = server.gui.add_button("Load File")

            # -------- Local edit params --------
            with server.gui.add_folder(
                "Local edit params", expand_by_default=False
            ) as lf:
                self.ui[self._u_key("local_target")] = server.gui.add_dropdown(
                    "Local target",
                    options=["pos_only", "pos_or_zero"],
                    initial_value=self.state["local_target"],
                )
                self.ui[self._u_key("dilate_3d")] = server.gui.add_slider(
                    "Dilate Gaussians", 0.0, 5.0, 0.1, int(self.state["dilate_3d"])
                )
                self.ui[self._u_key("perturb_3d")] = server.gui.add_slider(
                    "Perturb Gaussians", 0.0, 0.5, 0.01, int(self.state["perturb_3d"])
                )
                self.ui[self._u_key("random_gaussian_init")] = server.gui.add_slider(
                    "Random Gaussian init %",
                    0.0,
                    1.0,
                    0.01,
                    float(self.state["random_gaussian_init"]),
                )
                self.ui[self._u_key("dilate_2d")] = server.gui.add_slider(
                    "Dilate mask (2D)", 0.0, 1.0, 0.01, float(self.state["dilate_2d"])
                )
                self.ui[self._u_key("mask_scheduling")] = server.gui.add_checkbox(
                    "Mask scheduling", bool(self.state["mask_scheduling"])
                )
                self.ui[self._u_key("feature_alpha_mask")] = server.gui.add_checkbox(
                    "Feature alpha mask", bool(self.state["feature_alpha_mask"])
                )

            # -------- Camera sampling --------
            with server.gui.add_folder(
                "Camera sampling", expand_by_default=False
            ) as cf:
                self.ui[self._u_key("num_samples")] = server.gui.add_slider(
                    "Num samples", 1, 128, 1, int(self.state["num_samples"])
                )
                self.ui[self._u_key("camera_sample_strategy")] = (
                    server.gui.add_dropdown(
                        "Strategy",
                        options=["original", "perturb", "object_centric"],
                        initial_value=self.state["camera_sample_strategy"],
                    )
                )
                self.ui[self._u_key("rotation_std")] = server.gui.add_slider(
                    "Rotation std", 0.0, 1.0, 0.01, float(self.state["rotation_std"])
                )
                self.ui[self._u_key("translation_std")] = server.gui.add_slider(
                    "Translation std",
                    0.0,
                    2.0,
                    0.01,
                    float(self.state["translation_std"]),
                )
                self.ui[self._u_key("min_cam_dist")] = server.gui.add_slider(
                    "Min cam dist",
                    0.1,
                    10.0,
                    0.1,
                    float(self.state["min_cam_dist"]),
                )
                self.ui[self._u_key("max_cam_dist")] = server.gui.add_slider(
                    "Max cam dist",
                    0.1,
                    10.0,
                    0.1,
                    float(self.state["max_cam_dist"]),
                )
                self.ui["sample_poses_btn"] = server.gui.add_button("Sample View Poses")

            # -------- Generation --------
            with server.gui.add_folder("Generation", expand_by_default=False) as gf:
                self.ui[self._u_key("prompt")] = server.gui.add_text(
                    "Prompt", initial_value=str(self.state["prompt"])
                )
                self.ui[self._u_key("seed")] = server.gui.add_slider(
                    "Seed", 0, 10_000_000, 1, int(self.state["seed"])
                )
                maxD = self.sample_gen.max_ctrl_channels if self.sample_gen else 64
                self.ui[self._u_key("ctrl_dims")] = server.gui.add_slider(
                    "Control dims (D)", 1, maxD, 1, int(self.state["ctrl_dims"])
                )
                self.ui[self._u_key("ctrl_strength")] = server.gui.add_slider(
                    "Control strength",
                    0.0,
                    1.0,
                    0.05,
                    float(self.state["ctrl_strength"]),
                )
                self.ui[self._u_key("text_strength")] = server.gui.add_slider(
                    "Text ctrl strength",
                    0.0,
                    10.0,
                    0.1,
                    float(self.state["text_strength"]),
                )
                self.ui[self._u_key("rgb_guidance_noise_frac")] = server.gui.add_slider(
                    "RGB guidance noise frac",
                    0.0,
                    1.0,
                    0.05,
                    float(self.state["rgb_guidance_noise_frac"]),
                )
                self.ui[self._u_key("height")] = server.gui.add_slider(
                    "Frame height", 256, 1024, 64, int(self.state["height"])
                )
                self.ui[self._u_key("width")] = server.gui.add_slider(
                    "Frame width", 256, 1024, 64, int(self.state["width"])
                )
                self.ui[self._u_key("fov")] = server.gui.add_slider(
                    "FOV (rad)", 0.5, 1.0, 0.05, float(self.state["fov"])
                )
                self.ui["generate_btn"] = server.gui.add_button("Generate Samples")

            # -------- Optimization --------
            with server.gui.add_folder("Optimization", expand_by_default=False) as of:
                self.ui[self._u_key("runner_config")] = server.gui.add_text(
                    "Runner config", initial_value=str(self.state["runner_config"])
                )
                self.ui[self._u_key("train_steps")] = server.gui.add_slider(
                    "Train steps", 100, 5000, 100, int(self.state["train_steps"])
                )
                self.ui[self._u_key("lr_scale_geometry")] = server.gui.add_slider(
                    "LR scale (geom)",
                    0.0,
                    10.0,
                    0.1,
                    float(self.state["lr_scale_geometry"]),
                )
                self.ui[self._u_key("geometry_change_penalty")] = server.gui.add_slider(
                    "Geometry change penalty",
                    0.0,
                    10.0,
                    0.1,
                    float(self.state["geometry_change_penalty"]),
                )
                self.ui[self._u_key("opacity_decay")] = server.gui.add_slider(
                    "Opacity decay",
                    0.0,
                    1.0,
                    0.05,
                    float(self.state["opacity_decay"]),
                )
                self.ui[self._u_key("color_desaturate")] = server.gui.add_slider(
                    "Color desaturate",
                    0.0,
                    20.0,
                    1.0,
                    float(self.state["color_desaturate"]),
                )
                self.ui[self._u_key("lr_scale_color")] = server.gui.add_slider(
                    "LR scale (color)",
                    0.1,
                    5.0,
                    0.1,
                    float(self.state["lr_scale_color"]),
                )
                self.ui[self._u_key("use_feature_loss")] = server.gui.add_checkbox(
                    "Feature loss", bool(self.state["use_feature_loss"])
                )
                self.ui[self._u_key("use_photometric")] = server.gui.add_checkbox(
                    "Photometric/SSIM", bool(self.state["use_photometric"])
                )
                self.ui[self._u_key("use_pseudo_depth")] = server.gui.add_checkbox(
                    "Pseudo-depth loss", bool(self.state["use_pseudo_depth"])
                )
                self.ui[self._u_key("opt_after_edit")] = server.gui.add_checkbox(
                    "Optimize after edit", bool(self.state["opt_after_edit"])
                )
                self.ui[self._u_key("densify_max_num_rel")] = server.gui.add_slider(
                    "Densify max. n Gaussians (rel)",
                    0.0,
                    1.0,
                    0.001,
                    float(self.state.get("densify_max_num_rel", 1.0)),
                )
                self.ui[self._u_key("densify")] = server.gui.add_checkbox(
                    "Densify during opt", bool(self.state["densify"])
                )
                self.ui["optimize_btn"] = server.gui.add_button("Optimize Scene")

            # -------- Auto-mode HPs --------
            with server.gui.add_folder(
                "Auto-mode hyperparameters", expand_by_default=True
            ) as af:
                self.ui[self._u_key("auto_edit_iters")] = server.gui.add_slider(
                    "Edit Iterations", 1, 50, 1, int(self.state["auto_edit_iters"])
                )
                self.ui[self._u_key("auto_console_iters")] = server.gui.add_slider(
                    "Console Iterations",
                    1,
                    50,
                    1,
                    int(self.state["auto_console_iters"]),
                )
                self.ui[self._u_key("auto_gen_buffer")] = server.gui.add_slider(
                    "Gen buffer", 1, 10, 1, int(self.state["auto_gen_buffer"])
                )
                self.ui[self._u_key("auto_ctrl_dims_start")] = server.gui.add_slider(
                    "Ctrl D start", 1, maxD, 1, int(self.state["auto_ctrl_dims_start"])
                )
                self.ui[self._u_key("auto_ctrl_dims_cap")] = server.gui.add_slider(
                    "Ctrl D cap", 1, maxD, 1, int(self.state["auto_ctrl_dims_cap"])
                )
                self.ui[self._u_key("auto_ctrl_strength_start")] = (
                    server.gui.add_slider(
                        "Ctrl strength start",
                        0.0,
                        1.0,
                        0.01,
                        float(self.state["auto_ctrl_strength_start"]),
                    )
                )
                self.ui[self._u_key("auto_ctrl_strength_cap")] = server.gui.add_slider(
                    "Ctrl strength cap",
                    0.0,
                    1.0,
                    0.01,
                    float(self.state["auto_ctrl_strength_cap"]),
                )
                self.ui[self._u_key("auto_rgb_noise_start")] = server.gui.add_slider(
                    "RGB noise start",
                    0.0,
                    1.0,
                    0.01,
                    float(self.state["auto_rgb_noise_start"]),
                )
                self.ui[self._u_key("auto_rgb_noise_cap_edit")] = server.gui.add_slider(
                    "RGB noise cap",
                    0.0,
                    1.0,
                    0.01,
                    float(self.state["auto_rgb_noise_cap_edit"]),
                )
                self.ui[self._u_key("auto_rgb_noise_cap_console")] = (
                    server.gui.add_slider(
                        "RGB noise cap (console)",
                        0.0,
                        1.0,
                        0.01,
                        float(self.state["auto_rgb_noise_cap_console"]),
                    )
                )
                self.ui[self._u_key("auto_max_sh_degree")] = server.gui.add_slider(
                    "Max SH degree", 0, 5, 1, int(self.state["auto_max_sh_degree"])
                )
                self.ui[self._u_key("auto_geometry_lr_scale_min")] = (
                    server.gui.add_slider(
                        "Geom LR scale min",
                        0.1,
                        10.0,
                        0.1,
                        float(self.state["auto_geometry_lr_scale_min"]),
                    )
                )
                self.ui[self._u_key("auto_geometry_lr_scale_max")] = (
                    server.gui.add_slider(
                        "Geom LR scale max",
                        0.1,
                        5.0,
                        0.1,
                        float(self.state["auto_geometry_lr_scale_max"]),
                    )
                )
                self.ui[self._u_key("auto_2d_mask_dilate_start")] = (
                    server.gui.add_slider(
                        "2D mask dilate start",
                        0.0,
                        0.2,
                        0.005,
                        float(self.state["auto_2d_mask_dilate_start"]),
                    )
                )
                self.ui[self._u_key("auto_2d_mask_dilate_cap")] = server.gui.add_slider(
                    "2D mask dilate cap",
                    0.0,
                    0.2,
                    0.01,
                    float(self.state["auto_2d_mask_dilate_cap"]),
                )
                self.ui[self._u_key("auto_samples_per_iter")] = server.gui.add_slider(
                    "Samples/iter", 1, 64, 1, int(self.state["auto_samples_per_iter"])
                )
                self.ui[self._u_key("auto_train_steps")] = server.gui.add_slider(
                    "Train steps/iter",
                    100,
                    1000,
                    100,
                    int(self.state["auto_train_steps"]),
                )
                self.ui[self._u_key("auto_densify")] = server.gui.add_checkbox(
                    "Densify during edit", bool(self.state["auto_densify"])
                )
                self.ui[self._u_key("update_seg_anchors")] = server.gui.add_checkbox(
                    "Update seg anchors", bool(self.state["update_seg_anchors"])
                )
                self.ui[self._u_key("render_videos")] = server.gui.add_checkbox(
                    "Render videos", bool(self.state["render_videos"])
                )
                self.ui["auto_mode"] = server.gui.add_button("Auto mode (gen+opt)")

            # ---- Bindings (centralized) ----
            self._auto_bind_all()

            # ---- Actions ----
            @self.ui["sample_poses_btn"].on_click
            def _(_e):
                self._run_action("Sample View Poses", self.run_camera_sampling)

            @self.ui["generate_btn"].on_click
            def _(_e):
                def _do():
                    timestamp = time.strftime("%H_%M_%S")
                    self.result_dir = f"{self.editing_datadir}/{self.scene.name}/manual_edit/{timestamp}/"
                    self.run_generation()
                    if self.state.get("opt_after_edit", False):
                        self.run_optimization()

                self._run_action("Generate Samples", _do)

            @self.ui["optimize_btn"].on_click
            def _(_e):
                self._run_action("Optimize", self.run_optimization)

            @self.ui["auto_mode"].on_click
            def _(_e):
                self._run_action("Auto Edit", self.run_auto_edit)

            @self.ui["stop_btn"].on_click
            def _(_e):
                self._request_stop()

            @self.ui["load_btn"].on_click
            def _(_e):
                self._run_action("Load File", self.load_scene_or_preset)

            # Finalize
            self._update_ui_visibility()
            self.state["busy"] = False
            self._log("SceneEditingModule ready.")
            self.ui["status_text"].content = f"Status:\t **Ready**"

    # ------------------------------------------------------------------ #
    # Public: generation & optimization entry points
    # ------------------------------------------------------------------ #
    def load_scene_or_preset(self, path: str = None) -> None:
        if path is None:
            path = self.state.get("ckpt_path", "")
        path = str(path).strip()
        path = Path(path)

        # Relative path: look under configs/
        if not path.is_dir() and not path.is_file():
            cfg_path = Path(__file__).parent.parent / "configs"
            path = cfg_path / path

        # Directory: load all presets/checkpoints inside
        if path.is_dir():
            file_names = sorted(os.listdir(path))
            # New editing dir for the set of presets
            new_data_dir = Path(self.editing_datadir).parent
            preset_dir_name = path.stem
            self.editing_datadir = (
                new_data_dir
                / f"{preset_dir_name}_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}"
            ) if self.name is None else new_data_dir / f"{self.name}/{preset_dir_name}"
            if len(file_names) == 0:
                self._log(f"No preset or checkpoint files found in directory: {path}")
                return
            for fn in file_names:
                if ".json" in fn.lower():
                    self.auto_presets.append(path / fn)
            self._log(
                f"Found {len(file_names)} preset/checkpoint files in directory: {path}"
            )
            path = self.auto_presets.pop(0)  # load the first one now

        if path.suffix.lower() == ".json":
            # Single-file entry point: apply 0_param_adjust.json from the same
            # directory first (if present and not the file itself) so the
            # priority chain is: base_parameters < param_adjust < edit_config.
            param_adjust = path.parent / "0_param_adjust.json"
            if param_adjust.is_file() and param_adjust.resolve() != path.resolve():
                self.auto_presets.insert(0, path)
                self.load_preset(str(param_adjust))
            else:
                self.load_preset(str(path))
        elif path.suffix.lower() == ".pt" or path.suffix.lower() == ".pth":
            self.load_scene(str(path))
        else:
            self._log(f"Unrecognized file type for loading: {path}")

    def load_preset(self, preset_path: str) -> None:
        """Load a preset JSON file and apply to state."""
        preset_path = Path(preset_path)
        if not preset_path.is_file():
            self._log(f"Preset file not found: {preset_path}")
            return
        try:
            with open(preset_path, "r") as f:
                preset_params = json.load(f)
            # Cache the preset name so we can use it as the output dir later
            self.state["preset_name"] = preset_path.stem
            # Make the default to auto-edit after loading preset
            self.state["auto_edit"] = True
            # Apply preset params to state
            self.state.update(preset_params)
            self._log(f"Loaded preset from {preset_path}: {preset_params}")
            self._update_ui_visibility()
            # Load the scene
            if "ckpt_path" not in preset_params:
                if "ckpt_path" not in self.state or self.state["ckpt_path"] == "":
                    self._log(
                        "Preset missing 'ckpt_path' and no existing path set; cannot load scene."
                    )
                    return
                else:
                    self._log("Preset missing 'ckpt_path', using state ckpt_path.")
            self.load_scene(ckpt_dir=self.state["ckpt_path"])
            # Load the segmentation anchors if provided
            if self.state["mode"] == "local":
                if "segmentation" not in preset_params:
                    self._log(
                        "WARNING: No segmentation file in preset. Using state segmentation."
                    )
                if "segmentation" not in self.state:
                    self._log(
                        "ERROR: No segmentation in state; cannot load segmentation."
                    )
                    return
            if "segmentation" in self.state:
                seg_path = Path(self.state["segmentation"])
                if not seg_path.is_file():
                    seg_path = Path(self.reconstruction_dir) / seg_path if self.reconstruction_dir is not None else seg_path
                if not seg_path.is_file():
                    seg_path = Path(os.environ.get("TASE_ROOT", "")) / seg_path
                if not seg_path.is_file():
                    self._log(f"Segmentation file not found: {seg_path}")
                    return
                self.viewer.segmentation_module.import_files(str(seg_path))
        except Exception as ex:
            self._log(f"Failed to load preset from {preset_path}: {ex}")

        if self.state.get("auto_edit", True):
            self._log("Auto-edit preset loaded; starting auto edit.")
            try:
                self.run_auto_edit()
            except Exception as ex:
                self._log(f"Auto edit failed: {ex}")
        elif len(self.auto_presets) > 0:
            self._log("Loading next auto-edit preset.")
            self.load_preset(self.auto_presets.pop(0))

    def load_scene(self, ckpt_dir: str) -> None:
        ckpt_dir = Path(str(ckpt_dir).strip())
        if not ckpt_dir.is_file():
            ckpt_dir = Path(self.reconstruction_dir) / ckpt_dir if self.reconstruction_dir is not None else ckpt_dir
        if not ckpt_dir.is_file():
            ckpt_dir = Path(os.environ.get("TASE_ROOT", "")) / ckpt_dir
        if not ckpt_dir.is_file():
            self._log(f"Checkpoint file not found: {ckpt_dir}")
            return
        ckpt_dir = str(ckpt_dir)
        old_ckpt_dir = str(self.state.get("old_ckpt_path", "")).strip()
        self._log(f"Loading scene from original ({ckpt_dir}).")
        if (
            False and
            ckpt_dir is not None
            and old_ckpt_dir is not None
            and ckpt_dir == old_ckpt_dir
        ):
            ############ Quick reload ############
            # preserve the segmentation anchors
            anchors = self.viewer.segmentation_module.get_anchors()
            # only reload the scene, parser and samples stay the same
            scene = Scene.load_from_ckpt(
                ckpt_dir,
                device=self.device,
                require_grad=True,
                renormalize_coordinates=True,
                flip_z=self.flip_z
            )
            parser, samples = None, None
        else:
            anchors = None
            scene, parser, data_dir, samples = _load_ckpt_and_dataset(
                ckpt_dir, self.loaded_datadir, device=self.device, renormalize_coordinates=True, flip_z=self.flip_z
            )
        if scene is None:
            self._log(f"Failed to load scene from {ckpt_dir}.")
            return
        self.state["old_ckpt_path"] = ckpt_dir
        self.state["ckpt_path"] = ckpt_dir
        self.scene = scene
        self._splat_cache = None
        self._cache_edited_splats = None
        self.viewer.set_scene(self.scene)
        self.sample_gen.scene = scene
        if parser is not None:
            self.parser = parser
            self.sample_gen.parser = parser
        if samples is not None:
            self.viewer.elements = samples
            self.viewer._add_scene_elements()
        if anchors is not None:
            self.viewer.segmentation_module.set_anchors(anchors)
        self.viewer.rerender(None)

    @staticmethod
    def clone_param_dict(
        pd: Dict[str, torch.Tensor | torch.nn.Parameter],
        *,
        preserve_requires_grad: bool = True,
        track_grad: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.ParameterDict[str, torch.nn.Parameter]:
        """
        Clone a dict of {name: nn.Parameter or Tensor} into fresh nn.Parameter objects.

        Args:
        preserve_requires_grad: keep each param's requires_grad flag (ignored if track_grad is not None)
        track_grad: force requires_grad for all (e.g., True for MAML inner-loop); overrides preserve_requires_grad
        """
        out = torch.nn.ParameterDict()
        for k, v in pd.items():
            t = v.data if isinstance(v, torch.nn.Parameter) else v
            device = device if device is not None else t.device
            dty = dtype if dtype is not None else t.dtype
            t2 = t.detach().clone().to(device=device, dtype=dty)
            if track_grad is None:
                req = (
                    (v.requires_grad if isinstance(v, torch.nn.Parameter) else False)
                    if preserve_requires_grad
                    else False
                )
            else:
                req = bool(track_grad)
            out[k] = torch.nn.Parameter(t2, requires_grad=req)
        return out

    @staticmethod
    def sample_in_ellipsoid(
        center: torch.Tensor, axes: torch.Tensor, n_points: int, uniform: bool = False
    ) -> torch.Tensor:
        """
        Sample points in / from a 3D ellipsoid.

        Args:
            center: (3,) tensor, ellipsoid center.
            axes:   (3,3) tensor, whose COLUMNS are the semi-axis vectors (orientation + lengths).
            n_points: number of samples.
            uniform: if True -> uniform in ellipsoid volume; else -> elliptical Gaussian (unbounded).

        Returns:
            pts: (n_points, 3) tensor of sampled points.
        """
        assert center.shape == (3,), "center must be (3,)"
        assert axes.shape == (
            3,
            3,
        ), "axes must be (3,3) with columns as semi-axis vectors"

        device = center.device
        dtype = center.dtype

        if uniform:
            # --- Uniform in unit ball via normalized Gaussian directions + radius^(1/3) ---
            # directions ~ uniform on S^2
            dirs = torch.randn(n_points, 3, device=device, dtype=dtype)
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)

            # radii so that volume is uniform in 3D: r = U(0,1)^(1/3)
            r = torch.rand(n_points, 1, device=device, dtype=dtype).pow(1.0 / 3.0)

            u = dirs * r  # (n,3), uniform in unit ball
        else:
            # --- Elliptical Gaussian (centered), i.e., u ~ N(0, I) then x = center + A u ---
            u = torch.randn(n_points, 3, device=device, dtype=dtype)

        # Map unit samples to ellipsoid: x = c + A u
        pts = center[None, :] + u @ axes.T
        return pts

    @torch.no_grad()
    def add_random_gaussians(self, percent: float, selected_mask: torch.Tensor) -> None:
        """Randomly add new splats to the scene to help with geometry generation."""
        if percent <= 0.0:
            return

        orig_splats = self.scene.splats

        num_existing = orig_splats["means"][selected_mask].shape[0]
        num_to_add = int(num_existing * percent)
        self._log(
            f"Adding {num_to_add} random splats ({percent*100:.2f}%) to the scene."
        )
        # sample means within the bounding ellipsoid
        if self.viewer.segmentation_module._bounding_elipsoid is None:
            self._log("Bounding ellipsoid not found. Cannot add random splats.")
            return
        center, axes = self.viewer.segmentation_module._bounding_elipsoid  # (3,), (3,3)
        new_means = SceneEditingModule.sample_in_ellipsoid(
            center, axes, num_to_add, uniform=True
        )  # (num_to_add, 3)
        # sample the other parameters based on existing selected splats
        # local neighborhoods would probably be better, but this is a start
        # I could also create opacities and scales like the random init strategy of the original gsplat script
        """
        for key in orig_splats.keys():
            existing_param = orig_splats[key]
            if key == "means":
                new_param = new_means
            else:
                mean_val = existing_param[selected_mask].mean().item()
                std_val = existing_param[selected_mask].std().item()
                new_param = (
                    torch.randn(
                        (num_to_add, *existing_param.shape[1:]), device=self.device
                    )
                    * std_val
                    + mean_val
                )
            # concatenate to existing
            updated_param = torch.cat([existing_param, new_param], dim=0)
            self.scene.splats[key] = torch.nn.Parameter(
                updated_param, requires_grad=True
            )
            """
        # Pick the closest splat's params to each new splat
        existing_means = orig_splats["means"][selected_mask]  # (M,3)
        existing_params = {k: v[selected_mask] for k, v in orig_splats.items()}
        dists = torch.cdist(new_means, existing_means)  # (num_to_add, M)
        closest_idx = dists.argmin(dim=1)  # (num_to_add,)
        for key in orig_splats.keys():
            existing_param = existing_params[key]  # (M, ...)
            new_param = existing_param[closest_idx]  # (num_to_add, ...)
            # concatenate to existing
            updated_param = torch.cat([orig_splats[key], new_param], dim=0)
            self.scene.splats[key] = torch.nn.Parameter(
                updated_param, requires_grad=True
            )

    def modify_selected_splats(self) -> None:
        """Modify the selected splats in-place, e.g. scale them up/down."""
        self._run_action("Modify Selected Splats", self._modify_selected_splats)

    def get_selection_norm(self) -> None:
        """Renormalize the splats so that the selected ones fit in a unit sphere."""
        if self.state["mode"] != "local":
            self.state["selection_extent"] = 1.0
            return 1.0
        if self.scene is None:
            self._log("No splats found in the scene. Cannot normalize splats.")
            return
        # Get the current maximum distance in the selection
        selection = self.scene.selection
        if selection is None or selection.numel() == 0:
            self._log("No selection found. Cannot normalize splats.")
            return
        selected_mask = (selection > 0).squeeze(-1)  # (N,) bool
        num_selected = selected_mask.sum().item()
        if num_selected == 0:
            self._log("No splats selected. Cannot normalize splats.")
            return
        selected_means = self.scene.means[selected_mask]  # (M,3)
        center = selected_means.mean(dim=0, keepdim=True)  # (1,3)
        dists = (selected_means - center).norm(dim=1)  # (M,)
        max_dist = dists.max().item()
        if max_dist == 0.0:
            self._log("Selected splats are all at the same position. Cannot normalize.")
            return
        scale_factor = 1.0 / max_dist
        self.state["selection_extent"] = scale_factor
        self._log(f"Selection norm for learning rates: {scale_factor:.4f}")
        return scale_factor

    @torch.no_grad()
    def _modify_selected_splats(self) -> None:
        """Modify the selected splats in-place, e.g. scale them up/down."""
        if self.scene is None:
            self._log("No splats found in the scene. Cannot modify splats.")
            return
        if self._splat_cache is None:
            self._log("Caching original splats.")
            self._splat_cache = SceneEditingModule.clone_param_dict(self.scene.splats)
        else:
            # Restore from cache before applying new modifications
            self.scene.splats = SceneEditingModule.clone_param_dict(
                self._splat_cache, preserve_requires_grad=True
            )

        scale_offset = self.state.get("dilate_3d", None)
        spread = self.state.get("perturb_3d", None)
        random_init_percent = self.state.get("random_gaussian_init", 0.0)

        if scale_offset is None and spread is None:
            return

        selection = self.scene.selection
        if selection is None or selection.numel() == 0:
            self._log("No selection found. Cannot modify splats.")
            return
        selected_mask = (selection > 0).squeeze(-1)  # (N,) bool
        num_selected = selected_mask.sum().item()
        if num_selected == 0:
            self._log("No splats selected. Cannot modify splats.")
            return

        if scale_offset is not None:
            # Apply offset only to the two smaller axes of each selected splat (leave the largest axis unchanged)
            orig_scales = self._splat_cache["scales"]
            if scale_offset != 0.0:
                # Selected subset (M,3)
                sel_scales = orig_scales[selected_mask].clone()

                # Find index of largest axis per splat
                max_idx = sel_scales.argmax(dim=1, keepdim=True)  # (M,1)

                # Boolean mask for the two smaller axes
                smaller_axes_mask = torch.ones_like(sel_scales, dtype=torch.bool)
                smaller_axes_mask.scatter_(1, max_idx, False)

                # Add offset only to smaller axes
                sel_scales = sel_scales + scale_offset * smaller_axes_mask.float()

                # (Optional) prevent negatives if shrinking
                # sel_scales.clamp_(min=1e-6)

                self.scene.scales[selected_mask] = sel_scales

        if spread is not None:
            new_means = self._splat_cache["means"]
            if spread > 0.0:
                selected_means = new_means[selected_mask]  # (M,3)
                noise = torch.randn_like(selected_means)  # (M,3)
                noise = F.normalize(noise, dim=-1) * spread
                new_means = selected_means + noise
                self.scene.means[selected_mask] = new_means

        if random_init_percent > 0.0:
            self.add_random_gaussians(
                random_init_percent,
                selected_mask=selected_mask,
            )

        # renormalize splats so that the selected ones fit in a unit sphere
        self.get_selection_norm()

        self.viewer.rerender(None)

    @torch.no_grad()
    def show_original_splats(self, show_original: Optional[bool] = None) -> None:
        """Toggle between original and modified splats."""
        # NOTE: The splat caching is not done 100% cleanly, I need to add a scene hash to avoid issues with the segmentation especially
        if show_original is None:
            show_original = self.state.get("show_original", False)
        if show_original:
            if self._splat_cache is None:
                self._log("No cached original splats found.")
                return
            self._log("Showing original splats.")
            self._cache_edited_splats = SceneEditingModule.clone_param_dict(
                self.scene.splats, preserve_requires_grad=True
            )
            self.scene.splats = SceneEditingModule.clone_param_dict(
                self._splat_cache, preserve_requires_grad=True
            )
        else:
            if self._cache_edited_splats is None:
                self._log("No cached edited splats found.")
                return
            self._log("Showing edited splats.")
            self.scene.splats = SceneEditingModule.clone_param_dict(
                self._cache_edited_splats, preserve_requires_grad=True
            )
        self.viewer.rerender(None)

    def update_trainer_cfg(self, cfg: RunnerConfig) -> None:
        cfg.learning_rates.update(
            {
                "means": float(
                    self.state["lr_scale_geometry"]
                    * self.state.get("selection_extent", 1.0)
                )
                * cfg.learning_rates.get("means", 1e-3),
                "scales": float(
                    self.state["lr_scale_geometry"]
                    * self.state.get("selection_extent", 1.0)
                )
                * cfg.learning_rates.get("scales", 1e-3),
                "quats": float(
                    self.state["lr_scale_geometry"]
                    * self.state.get("selection_extent", 1.0)
                )
                * cfg.learning_rates.get("quats", 1e-3),
                "opacities": cfg.learning_rates.get("opacities", 1e-3)
                * self.state.get("opacity_lr_scale_factor", 1.0),
                "sh0": float(self.state["lr_scale_color"])
                * cfg.learning_rates.get("sh0", 1e-3),
                "shN": float(self.state["lr_scale_color"])
                * cfg.learning_rates.get("shN", 1e-3),
                "features": float(self.state["lr_scale_color"])
                * cfg.learning_rates.get("features", 1e-3),
                "background": float(self.state["lr_scale_color"])
                * cfg.learning_rates.get("background", 1e-3),
                "feature_bg": float(self.state["lr_scale_color"])
                * cfg.learning_rates.get("feature_bg", 1e-3),
            }
        )
        cfg.geometry_change_penalty = float(
            self.state.get("geometry_change_penalty", cfg.geometry_change_penalty)
        )
        cfg.opacity_decay = (
            float(self.state.get("opacity_decay", cfg.opacity_decay))
            # * self.state["rgb_guidance_noise_frac"]
            # ** 2  # decay scaled by square of rgb guidance noise fraction
            # * self.state["lr_scale_geometry"]
        )
        cfg.opacity_reg = self.state.get("opacity_reg", cfg.opacity_reg)
        cfg.color_desaturate = (
            float(self.state.get("color_desaturate", cfg.color_desaturate))
            * self.state["lr_scale_color"]
            # * (1 - self.state["rgb_guidance_noise_frac"])
        )
        cfg.target_opacity = self.state.get("target_opacity", cfg.target_opacity)
        cfg.target_color = self.state.get("target_color", cfg.target_color)
        cfg.color_reg = float(self.state.get("color_reg", cfg.color_reg))
        cfg.sh_degree = int(self.state.get("sh_degree", cfg.sh_degree))
        cfg.result_dir = self.result_dir + "/splat_ckpts/"
        cfg.feature_loss = self.state["use_feature_loss"]
        cfg.rgb_loss = self.state["use_photometric"]
        cfg.depth_loss = self.state["use_pseudo_depth"]
        cfg.use_loss_mask = self.state.get("feature_alpha_mask", False)
        cfg.optimize_background = self.state.get("mode", "global") == "global"
        cfg.strategy.densify_max_num_rel = float(
            self.state.get("densify_max_num_rel", 1.0)
        )
        cfg.prune_only = self.state.get("prune_only", False)
        cfg.local_alpha_reg_interval = self.state.get("local_alpha_reg_interval", -1)

    def run_camera_sampling(self) -> None:
        k = int(self.state["num_samples"])
        strat = str(self.state["camera_sample_strategy"])
        rot_std = float(self.state["rotation_std"])
        trans_std = float(self.state["translation_std"])
        min_cam_dist = float(self.state.get("min_cam_dist", 1.0))
        max_cam_dist = float(self.state.get("max_cam_dist", 2.0))

        self._set_status(phase="sampling")
        bounding_ellipsoid = self.viewer.segmentation_module._bounding_elipsoid

        c2ws, names = self.sample_gen.sample_view_poses(
            num_samples=k,
            camera_sample_strategy=strat,
            bounding_ellipsoid=bounding_ellipsoid,
            rotation_std=rot_std,
            translation_std=trans_std,
            min_cam_dist=min_cam_dist,
            max_cam_dist=max_cam_dist,
            viewer=self.viewer,
            name_prefix="edit_pose",
        )

        self._sampled_c2ws = c2ws
        self._sampled_cam_names = names
        self._log(f"Sampled {len(c2ws)} poses with '{strat}'.")

    def run_generation(self) -> None:
        """
        Global mode: use sampled poses if available, otherwise sample now; then generate images.
        Local mode: (to be implemented next).
        """
        mode = str(self.state["mode"])
        masked = mode == "local"

        # Ensure we have poses
        c2ws = self._sampled_c2ws
        pose_names = self._sampled_cam_names
        k = int(self.state["num_samples"])

        if not c2ws or len(c2ws) != k:
            # Auto-sample now
            k = int(self.state["num_samples"])
            strat = str(self.state["camera_sample_strategy"])
            rot_std = float(self.state["rotation_std"])
            trans_std = float(self.state["translation_std"])
            c2ws, pose_names = self.sample_gen.sample_view_poses(
                num_samples=k,
                camera_sample_strategy=strat,
                rotation_std=rot_std,
                translation_std=trans_std,
                viewer=self.viewer,
                name_prefix="edit_pose",
            )
            self.state["sampled_c2ws"] = c2ws
            self.state["sampled_pose_names"] = pose_names

        self._set_status(phase="generating")

        # Pull generation params from state
        prompt = str(self.state["prompt"])
        seed = int(self.state["seed"])
        ctrl_dims = int(self.state["ctrl_dims"])
        ctrl_strength = float(self.state["ctrl_strength"])
        text_strength = float(self.state["text_strength"])
        dynamic_q = float(self.state["dynamic_q"])
        rgb_guidance_noise_frac = float(self.state["rgb_guidance_noise_frac"])
        mask_scheduling = bool(self.state["mask_scheduling"])
        mask_dilation = float(self.state["dilate_2d"])
        mask_amplify = float(self.state["mask_amplify"])
        feature_alpha_mask = bool(self.state["feature_alpha_mask"])
        update_seg_anchors = bool(self.state.get("update_seg_anchors", False))
        predict_depth = bool(self.state["use_pseudo_depth"])
        segmentation_dims = int(
            self.viewer.segmentation_module.state.get("trunc_dim", ctrl_dims)
        )

        if feature_alpha_mask and masked:
            # get the anchors from the segmentation module
            # NOTE: This does not reapply the segmentation, just updates the cached anchors and retrieves them for loss mask generation
            if update_seg_anchors:
                segmentation_anchors = (
                    self.viewer.segmentation_module.recluster_anchors(apply=False)
                )
            else:
                segmentation_anchors = self.viewer.segmentation_module.get_anchors()
        else:
            segmentation_anchors = None

        artifact_removal = (
            float(self.state["background_artifact_removal_prob"]) if masked else 0.0
        )

        debug = bool(self.state.get("debug", False))

        if artifact_removal > 0.0:
            self.scene.split_frozen_splats()

        self._log(f"Generating {len(c2ws)} frames")
        self.new_scene, self.gen_cnt = self.sample_gen.generate_samples_from_poses(
            c2ws=c2ws,
            pose_names=pose_names,
            feature_ctrl_strength=ctrl_strength,
            feature_ctrl_dims=ctrl_dims,
            text_ctrl_strength=text_strength,
            rgb_guide_noise_frac=rgb_guidance_noise_frac,
            dynamic_q=dynamic_q if dynamic_q > 0.0 else None,
            masked=masked,
            mask_scheduling=mask_scheduling,
            mask_dilation=mask_dilation,
            mask_amplify=mask_amplify,
            segmentation_anchors=segmentation_anchors,
            segmentation_dims=segmentation_dims,
            predict_features=True,
            predict_depth=predict_depth,
            debug=debug,  # Set this to true to save intermediate debug images
            artifact_removal=artifact_removal,
            seed=seed,
            new_data_dir=self.result_dir,
            prompt=prompt,
            viewer=self.viewer,
        )
        self._log(f"Done. New scene metadata: {self.new_scene}")

        if artifact_removal > 0.0:
            self.scene.merge_frozen_splats()

        if not self.viewer.cancel and self.state["opt_after_edit"]:
            self._log("Starting optimization as requested…")
            self.run_optimization()

        self.viewer.cancel = False  # Load cancel flag

    def run_optimization(
        self,
        gen_buffer=None,  # Use the last n generated sample generations
    ) -> None:
        """
        Scene optimization restricted to selected Gaussians:
          - build 1D gaussian mask from selection and local_target policy
          - configure runner param groups (freeze non-selected)
          - set losses and LR scales; train for state['train_steps']
        """
        if self.gen_cnt <= 0 or not self.new_scene:
            self._log("No new samples generated yet. Generate samples first.")
            self._set_status(phase="idle")
            return

        ckpt_path = ""
        cfg = RunnerConfig.from_yaml(self.state["runner_config"])
        masked = self.state["mode"] == "local"
        self.update_trainer_cfg(cfg)
        self.scene.sh_degree = cfg.sh_degree
        # update to the latest scene metadata
        cfg.data_dir = self.result_dir
        cfg.scene = self.new_scene
        # active generations
        if gen_buffer is None:
            act_gens = [self.gen_cnt]
        else:
            act_gens = list(
                range(max(1, self.gen_cnt - gen_buffer + 1), self.gen_cnt + 1)
            )
            gen_weights = {
                g: 1.0 * self.state["auto_gen_loss_decay_factor"] ** i
                for i, g in enumerate(act_gens)
            }
        self._set_status(phase="optimizing")
        # Copy the original geometry for change penalty
        self.scene.store_original_geometry()
        # Split out the frozen splats
        if masked:
            self.scene.split_frozen_splats()
        runner = Runner(
            cfg=cfg,
            scene=self.scene,
            parser=self.parser,
            viewer=self.viewer,
            load_dataset=True,  # need data for optimization
            active_sample_gens=act_gens,
        )
        densify = bool(self.state["densify"])
        self._log(
            f"Optimizing with sample generations {act_gens}, densify={densify}, masked={masked}"
        )
        try:
            ckpt_path = runner.train(
                steps=int(self.state["train_steps"]),
                generational_weights=gen_weights if gen_buffer is not None else None,
                densify=densify,
            )
            self._log("Optimization finished.")
        except Exception as ex:
            tb_str = traceback.format_exc()
            self._log(f"Optimization failed: {ex}\n{tb_str}")
            self._set_status(action="Optimization", phase="error", detail=str(ex))
            if self.viewer.lock.locked():
                self.viewer.lock.release()
        finally:
            self.viewer.cancel = False
            self.scene.merge_frozen_splats()
            self.viewer.rerender(None)  # ensure we see the latest
            self.viewer.set_mode("rendering")

        return ckpt_path

    def run_auto_edit(self) -> None:
        """
        Auto mode: iterate (sample -> generate -> optimize) with scheduled HPs,
        then a final consolidation round.
        """
        preset_name = str(self.state.get("preset_name", None))
        if preset_name is None:
            preset_name = time.strftime("%H_%M_%S")
        self.result_dir = f"{self.editing_datadir}/auto_edit/{preset_name}"
        if os.path.exists(self.result_dir):
            # append timestamp to avoid overwriting
            timestamp = time.strftime("%H_%M_%S")
            self.result_dir += f"_{timestamp}"
        self.gen_cnt = 0
        self.parser.clear_generated()  # start fresh
        self._log("Auto mode, result dir: " + self.result_dir)
        self._dump_params()
        RunnerConfig.copy_yaml(
            self.state["runner_config"],
            os.path.join(self.result_dir, "auto_edit_runner_config.yaml"),
        )
        ckpt_path = ""

        # apply the local splat modifications if any
        if self.state["mode"] == "local":
            self._modify_selected_splats()

        # dump the segmentation anchors
        if self.state["mode"] == "local":
            self.viewer.segmentation_module.export_anchors(
                self.result_dir + "/segmentation/object.anchor"
            )
            self.viewer.segmentation_module.export_segmentation(
                self.result_dir + "/segmentation/object.seg"
            )

        # Hide the selection
        self.viewer.segmentation_module.set_show(False)

        self.show_original_splats(True)

        # dump the original splats
        self.scene.save_ckpt(self.result_dir + "/splat_ckpts/orig_scene.pt")

        # render video of original scene
        medium_cam_dist = float(
            (self.state.get("min_cam_dist", 1.0) + self.state.get("max_cam_dist", 2.0))
            / 2.0
        )
        if self.state["render_videos"]:
            self._log("Sampling video trajectory poses.")
            self.scene.sh_degree = 2
            try:
                if "pose_dir" in self.state and self.state["pose_dir"] != "":
                    # load poses from directory — resolve relative to TASE_ROOT
                    pose_dir = resolve_path(self.state["pose_dir"])
                    pose_file = pose_dir / "edited_scene_cam_poses.npy"
                    self._video_c2ws = np.load(str(pose_file))  # (N,4,4)
                else:
                    self._video_c2ws, _ = self.sample_gen.sample_view_poses(
                        num_samples=240,
                        camera_sample_strategy="orbit",
                        viewer=self.viewer,
                        name_prefix="video_pose",
                        bounding_ellipsoid=self.viewer.segmentation_module._bounding_elipsoid,
                        min_cam_dist=medium_cam_dist,
                        max_cam_dist=medium_cam_dist,
                    )
                # render the video of the original scene
                self.scene.render_video(
                    self.result_dir + "/videos/orig_scene",
                    self._video_c2ws,
                    (self.sample_gen.W, self.sample_gen.H),
                    fov=0.7,
                    depth=True,
                )
            except Exception as e:
                self._log(f"Video rendering failed: {e}")

        # render images also
        if self.state["render_images"]:
            try:
                if "pose_dir" in self.state and self.state["pose_dir"] != "":
                    # load poses from directory
                    pose_dir = str(self.state["pose_dir"]) + "/_cam_poses.npy"
                    self._image_c2ws = np.load(pose_dir)  # (N,4,4)
                else:
                    self._image_c2ws, _ = self.sample_gen.sample_view_poses(
                        num_samples=(3, 8),
                        camera_sample_strategy="grid",
                        viewer=self.viewer,
                        name_prefix="image_pose",
                        bounding_ellipsoid=self.viewer.segmentation_module._bounding_elipsoid,
                        min_cam_dist=medium_cam_dist,
                        max_cam_dist=medium_cam_dist,
                    )
                self.scene.render_images(
                    self.result_dir + "/renders/orig_scene/",
                    self._image_c2ws,
                    (self.sample_gen.W, self.sample_gen.H),
                    fov=0.7,
                    depth=True,
                )
            except Exception as e:
                self._log(f"Image rendering failed: {e}")

        self.show_original_splats(False)

        # --- Start editing timer and VRAM tracking ---
        _edit_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            _device_name = "cpu"

        # Pull schedule HPs
        edit_iters = int(self.state["auto_edit_iters"])
        console_iters = int(self.state["auto_console_iters"])
        total_iters = edit_iters + console_iters
        gen_buf = int(self.state["auto_gen_buffer"])

        D = int(self.state["auto_ctrl_dims_start"])
        D_cap = int(self.state["auto_ctrl_dims_cap"])
        D_scale = (D_cap / max(1, D)) ** (1.0 / max(1, edit_iters))

        ctrl_strength = float(self.state["auto_ctrl_strength_start"])
        ctrl_cap = float(self.state["auto_ctrl_strength_cap"])
        if ctrl_cap < ctrl_strength:
            self._log(
                f"Warning: ctrl_strength_cap ({ctrl_cap}) < ctrl_strength_start ({ctrl_strength}). Setting both to {ctrl_strength}."
            )
            ctrl_cap = ctrl_strength
        ctrl_strength_incr = (ctrl_cap - ctrl_strength) / max(1, console_iters)

        geometry_lr_min = float(self.state["auto_geometry_lr_scale_min"])
        geometry_lr_max = float(self.state["auto_geometry_lr_scale_max"])
        if geometry_lr_min > geometry_lr_max:
            self._log(
                f"Warning: geometry_lr_scale_cap ({geometry_lr_min}) > geometry_lr_scale_start ({geometry_lr_max}). Setting both to {geometry_lr_min}."
            )
            geometry_lr_max = geometry_lr_min
        # sine geometry learning rate over all iterations
        if self.state["auto_geometry_lr_schedule"] == "sine":
            self.state["lr_scale_geometry"] = geometry_lr_min
        else:
            self.state["lr_scale_geometry"] = geometry_lr_max

        mask_dilate = float(self.state["auto_2d_mask_dilate_start"])
        mask_dilate_cap = float(self.state["auto_2d_mask_dilate_cap"])
        if mask_dilate_cap > mask_dilate:
            self._log(
                f"Warning: 2d_mask_dilate_cap ({mask_dilate_cap}) > 2d_mask_dilate_start ({mask_dilate}). Setting both to {mask_dilate}."
            )
            mask_dilate_cap = mask_dilate
        mask_dilate_incr = (mask_dilate_cap - mask_dilate) / max(
            1, edit_iters + console_iters
        )
        self.state["dilate_2d"] = mask_dilate

        rgb_noise = float(self.state["auto_rgb_noise_start"])
        rgb_cap_edit = float(self.state["auto_rgb_noise_cap_edit"])
        rgb_cap_console = float(self.state["auto_rgb_noise_cap_console"])
        if rgb_cap_edit < rgb_cap_console:
            self._log(
                f"Warning: rgb_noise_cap_edit ({rgb_cap_edit}) < rgb_noise_cap_console ({rgb_cap_console}). Setting both to {rgb_cap_edit}."
            )
            rgb_cap_console = rgb_cap_edit
        rgb_scale_edit = (rgb_cap_edit / max(1e-3, rgb_noise)) ** (
            1.0 / max(1, edit_iters)
        )
        rgb_scale_console = (rgb_cap_console / max(1e-3, rgb_cap_edit)) ** (
            1.0 / max(1, console_iters)
        )

        k_per_iter = int(self.state["auto_samples_per_iter"])
        steps_per_iter = int(self.state["auto_train_steps"])
        densify = bool(self.state["auto_densify"])
        self.state["prune_only"] = False

        # Apply to state and UI (single place)
        self.state["ctrl_dims"] = max(
            1, min(D, self.sample_gen.max_ctrl_channels if self.sample_gen else 64)
        )
        self.state["rgb_guidance_noise_frac"] = float(rgb_noise)
        self.state["num_samples"] = k_per_iter
        self.state["train_steps"] = steps_per_iter
        self.state["densify"] = densify
        self.state["ctrl_strength"] = ctrl_strength
        self.state["sh_degree"] = 0  # start with SH0
        self._update_ui_visibility()

        # seed the RNG of the camera sampler for reproducibility
        camera_seed = int(self.state["seed"])
        self.scene.set_seed(camera_seed)

        ### DEBUG ###
        # list_cuda_tensors()

        for i in range(edit_iters):
            if self.state["cancel"]:
                break

            self._set_status(detail=f"Edit {i+1}/{edit_iters}")

            self.run_camera_sampling()
            if self.state["cancel"]:
                break

            self.run_generation()
            if self.state["cancel"]:
                break

            self.run_optimization(gen_buffer=gen_buf)

            # Schedule updates
            self.state["ctrl_dims"] = min(D_cap, self.state["ctrl_dims"] * D_scale)
            # self.state["ctrl_strength"] = min(ctrl_cap, self.state["ctrl_strength"] + ctrl_strength_incr)
            self.state["rgb_guidance_noise_frac"] = max(
                rgb_cap_edit, self.state["rgb_guidance_noise_frac"] * rgb_scale_edit
            )
            self.state["dilate_2d"] = max(
                mask_dilate_cap, self.state["dilate_2d"] + mask_dilate_incr
            )
            # learning rate schedule for the geometry
            cur_iter = i + 1
            if self.state["auto_geometry_lr_schedule"] == "sine":
                cur_geometry_lr_scale = (
                    torch.sin(
                        torch.tensor(float(cur_iter) / edit_iters * 3.1415926 / 2)
                    ).item()
                    * (geometry_lr_max - geometry_lr_min)
                    + geometry_lr_min
                )
            elif self.state["auto_geometry_lr_schedule"] == "linear":
                cur_geometry_lr_scale = (
                    float(cur_iter) / total_iters * (geometry_lr_max - geometry_lr_min)
                    + geometry_lr_min
                )
            self.state["lr_scale_geometry"] = cur_geometry_lr_scale

            # render images
            if self.state["render_images"] and not self.viewer.cancel:
                self._log(f"Rendering images at edit iteration {i+1}.")
                try:
                    self.scene.render_images(
                        self.result_dir + f"/renders/edited_iter_{i+1}/",
                        self._image_c2ws,
                        (self.sample_gen.W, self.sample_gen.H),
                        fov=0.7,
                        depth=True,
                    )
                except Exception as e:
                    self._log(f"Image rendering failed: {e}")

            # reflect to UI
            self._update_ui_visibility()

        # N rounds of consolidation, decreasing the noise level and increasing the samples
        console_ctrl_cap = int(self.state.get("auto_ctrl_dims_cap_console", 64))
        self.state["ctrl_dims"] = min(
            self.sample_gen.max_ctrl_channels if self.sample_gen else 64,
            console_ctrl_cap,
        )
        self.state["rgb_guidance_noise_frac"] = float(rgb_cap_edit)
        self.state["densify"] = True  # always densify during consolidation
        max_sh_degree = int(self.state["auto_max_sh_degree"])
        increase_sh_every = max(1, console_iters // (max_sh_degree + 1))

        for i in range(console_iters):
            if self.state["cancel"]:
                break

            self._set_status(detail=f"Console {i+1}/{console_iters}")

            self.run_camera_sampling()
            if self.state["cancel"]:
                break

            self.run_generation()
            if self.state["cancel"]:
                break

            self.run_optimization(gen_buffer=gen_buf)

            # Schedule updates
            self.state["rgb_guidance_noise_frac"] = max(
                rgb_cap_console,
                self.state["rgb_guidance_noise_frac"] * rgb_scale_console,
            )
            self.state["ctrl_strength"] = min(
                ctrl_cap, self.state["ctrl_strength"] + ctrl_strength_incr
            )
            self.state["dilate_2d"] = max(
                mask_dilate_cap, self.state["dilate_2d"] + mask_dilate_incr
            )
            self.state["sh_degree"] = min(
                max_sh_degree,
                i // increase_sh_every,
            )
            # Stop the densification after half of the console iters
            if i == console_iters // 2:
                self.state["prune_only"] = True
            # learning rate schedule for the geometry
            if self.state["auto_geometry_lr_schedule"] == "sine":
                cur_geometry_lr_scale = (
                    torch.sin(
                        torch.tensor(
                            float(i + 1) / console_iters * 3.1415926 / 2 + 3.1415926 / 2
                        )
                    ).item()
                    * (geometry_lr_max - geometry_lr_min)
                    + geometry_lr_min
                )
            elif self.state["auto_geometry_lr_schedule"] == "linear":
                cur_geometry_lr_scale = (
                    float(cur_iter) / total_iters * (geometry_lr_max - geometry_lr_min)
                    + geometry_lr_min
                )
            self.state["lr_scale_geometry"] = cur_geometry_lr_scale

            # render images
            if self.state["render_images"] and not self.viewer.cancel:
                self._log(f"Rendering images at console iteration {i+1}.")
                try:
                    self.scene.render_images(
                        self.result_dir + f"/renders/edited_console_{i+1}/",
                        self._image_c2ws,
                        (self.sample_gen.W, self.sample_gen.H),
                        fov=0.7,
                        depth=True,
                    )
                except Exception as e:
                    self._log(f"Image rendering failed: {e}")

            # reflect to UI
            self._update_ui_visibility()

        self.state["auto_edit"] = False

        # --- Save editing stats ---
        self._save_editing_stats(
            edit_start=_edit_start,
            device_name=_device_name,
            cancelled=bool(self.viewer.cancel or self.state.get("cancel", False)),
        )

        if not self.viewer.cancel and not self.state["cancel"]:
            # Save the final checkpoint
            ckpt_path = self.scene.save_ckpt(self.result_dir + "/splat_ckpts/last.pt")
            # render the video of the edited scene
            if self.state["render_videos"]:
                self._log("Rendering video of edited scene.")
                try:
                    self.scene.render_video(
                        self.result_dir + "/videos/edited_scene",
                        self._video_c2ws,
                        (self.sample_gen.W, self.sample_gen.H),
                        fov=0.7,
                        depth=True,
                    )
                except Exception as e:
                    self._log(f"Video rendering failed: {e}")
            if self.state["render_images"] and not self.viewer.cancel:
                self._log(f"Rendering images at edit iteration {i+1}.")
                try:
                    self.scene.render_images(
                        self.result_dir + f"/renders/edited/",
                        self._image_c2ws,
                        (self.sample_gen.W, self.sample_gen.H),
                        fov=0.7,
                        depth=True,
                    )
                except Exception as e:
                    self._log(f"Image rendering failed: {e}")
            self._log(f"Auto mode finished, final ckpt: {ckpt_path}")
            self.viewer.cancel = False
            if len(self.auto_presets) > 0:
                self._log("Applying next auto preset.")
                self.load_scene_or_preset(self.auto_presets.pop(0))
        else:
            self._log("Auto mode cancelled, clearing auto queue.")
            self.auto_presets = []
            self.viewer.cancel = False

    def _save_editing_stats(
        self,
        edit_start: float,
        device_name: str,
        cancelled: bool = False,
    ) -> None:
        """Compute elapsed time and peak VRAM, save to result_dir/editing_stats.json and print."""
        elapsed = time.time() - edit_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        human_time = f"{hours:02d}h {minutes:02d}m {seconds:05.2f}s"

        stats: Dict[str, object] = {
            "editing_time_seconds": round(elapsed, 2),
            "editing_time_human": human_time,
            "device": device_name,
            "cancelled": cancelled,
        }
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            stats["peak_vram_gb"] = round(peak_bytes / 1024 ** 3, 3)

        # Save to file
        try:
            os.makedirs(self.result_dir, exist_ok=True)
            stats_path = Path(self.result_dir) / "editing_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as ex:
            self._log(f"Failed to save editing stats: {ex}")

        # Print to console and internal log
        vram_str = f", peak VRAM: {stats['peak_vram_gb']:.3f} GB" if "peak_vram_gb" in stats else ""
        status_str = " (cancelled)" if cancelled else ""
        msg = f"Editing time{status_str}: {human_time}, device: {device_name}{vram_str}"
        print(f"[SceneEditingModule] {msg}")
        self._log(msg)

    def _update_ui_visibility(self) -> None:
        """
        mode = str(self.state.get("mode", "global"))               # "global" | "local"
        control_mode = str(self.state.get("control_mode", "manual"))  # "manual" | "auto"

        # Local-only params visible if mode == local
        local_keys = ["local_target_dropdown", "dilate_3d", "mask_erode", "mask_feather"]
        self._set_visible(local_keys, mode == "local")

        # Camera sampling always relevant for global & SDS; still useful for local in future, keep visible.
        cam_keys = ["num_samples", "cam_sample_dropdown", "rot_std", "trans_std"]
        self._set_visible(cam_keys, True)

        # Manual vs Auto action buttons
        manual_action_keys = ["sample_poses_btn", "generate_btn", "optimize_btn"]
        auto_action_keys   = ["auto_mode"]
        self._set_visible(manual_action_keys, control_mode == "manual")
        self._set_visible(auto_action_keys,   control_mode == "auto")

        # Auto-mode HPs panel visibility
        auto_hp_keys = [
            "auto_iters", "auto_gen_buffer", "auto_init_seed",
            "auto_ctrl_dims_start", "auto_ctrl_dims_scale", "auto_ctrl_dims_cap",
            "auto_rgb_noise_start", "auto_rgb_noise_decay", "auto_rgb_noise_cap",
            "auto_samples_per_iter", "auto_final_samples",
        ]
        self._set_visible(auto_hp_keys, control_mode == "auto")
        """

        # Keep UI elements consistent with internal state when we change them programmatically
        # (do not force-write frequently; only when different)
        pairs = [(self._u_key(k), v) for (k, v) in self.state.items()]
        for ui_key, val in pairs:
            if ui_key in self.ui and getattr(self.ui[ui_key], "value", None) != val:
                self._sync_value(ui_key, val)

    # ---- Status + log helpers ---------------------------------------- #
    def _set_status(
        self,
        action: str | None = None,
        phase: str | None = None,
        detail: str | None = None,
    ):
        """Update the status markdown from current state."""
        if action is not None:
            self.state["status_action"] = action
        if phase is not None:
            self.state["status_phase"] = phase
        if detail is not None:
            self.state["status_detail"] = detail

        act = self.state.get("status_action", "Ready")
        ph = self.state.get("status_phase", "")
        dt = self.state.get("status_detail", "")

        parts = [f"**{act}**"]
        if ph:
            parts.append(f"\n_{ph}_")
            if dt:
                parts.append(f" — {dt}")
        if "status_text" in self.ui:
            self.ui["status_text"].content = "Status:\t" + " ".join(parts)

    def _log(self, msg: str, *, truncate_to: int | None = None) -> None:
        print(f"[SceneEditingModule] {msg}")
        limit = (
            int(self.state.get("log_truncate", 180))
            if truncate_to is None
            else int(truncate_to)
        )
        disp = msg if len(msg) <= limit else (msg[: max(0, limit - 1)] + "…")
        self.state["last_log"] = msg
        if "last_log_text" in self.ui:
            self.ui["last_log_text"].content = f"{disp}"
