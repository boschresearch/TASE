# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
import os

from TASE.splatting.util.utils import compute_knn, rotmat_to_quat


class SegmentationModule3D:
    """
    3D point segmentation using cosine similarity to POS/NEG anchors.

    - Works on a dict-like `splats` with:
        splats["means"]      : (N,3) point positions (torch/numpy)
        splats["features"]   : (N,C) per-point feature embeddings (unit norm recommended)
        splats["selection"]  : (N,) int8 in {-1,0,1} (written by this module)
    - Anchors are selected in **2D** via rectangle selection on the viewer.
      Each selection triggers a **fresh render** of per-pixel features, and the
      anchor is the mean feature over the selected box.
    - Segmentation is **similarity-only** (no spatial distance yet).
    - Conflicts resolved by larger similarity: if both pass, pick the side with higher relative similarity.

    UI (subset of the 2D module; only relevant params):
      - Add POS/NEG anchor, Clear anchors
      - Truncation D (channels kept) — dims >= D are zeroed, then L2-renormalize
      - Relative thresholds τ⁺/τ⁻ in [0,1]; or auto via ratios (top-k)
      - Pos/Neg ratios for auto τ
      - Apply segmentation (writes splats["selection"])
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        server,
        viewer,
        splats: Dict[str, object],
        *,
        mode: Literal["simple", "full"] = "full",
        pmin: float = 1.0,  # percentile low for normalization of similarity
    ) -> None:
        self.server = server
        self.viewer = viewer
        self.splats = splats
        self.feature_key = "features"
        self.mode = "simple" if mode == "simple" else "full"
        self.pmin = float(pmin)
        self.show = False

        # Internal state (UI + anchors)
        self.state: Dict[str, object] = {
            "ui_mode": "simple",  # "simple" | "advanced" | "expert"
            "pos_anchors": [],  # list[tensor (C,)]
            "neg_anchors": [],  # list[tensor (C,)]
            "last_anchor_pos": [],  # caches wether last anchor was pos/neg
            "C": None,  # feature dimension
            "trunc_dim": 16,  # None or int<D
            "render_res": 512,  # Just for feature rendering, reduce memory
            "tau_pos_rel": 0.9,  # None => auto by ratio
            "tau_neg_rel": 0.9,  # None => auto by ratio
            "pos_ratio": 0.05,
            "neg_ratio": 0.05,
            "prop_sim_thresh": 0.8,  # cosine similarity threshold in [0,1]
            "prop_max_dist": 0.02,  # max 3D distance (same units as splats["means"])
            "floaters_k": 16,  # neighbors used for majority vote (excluding self)
            "prop_k": 16,  # neighbors per seed
            "prop_iters": 10,  # used in expert
            "floaters_iters": 10,  # used in expert
        }
        self.ui: Dict[str, object] = {}  # UI widgets
        # kNN cache reused across methods; positions & N assumed constant
        self._knn_cache = {"K": 0, "idx": None, "d2": None}
        self._bounding_elipsoid: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None  # (center (3,), axes (3,3))
        )
        self._viewer_bounding_elipsoid = None
        self._viewer_bounding_coords = None

        self.prepare_state()

    def prepare_state(self) -> None:
        """Call once before attach() to init state from splats."""
        feat = self._get_point_features()
        self.state["C"] = feat.shape[1] if feat is not None else 0
        if self.state["trunc_dim"] > self.state["C"]:
            self.state["trunc_dim"] = self.state["C"]
        self.gui_folder = None

    # --------------------------------------------------------------------- #
    # Public: attach panel + callbacks
    # --------------------------------------------------------------------- #
    def detach(self) -> None:
        if self.gui_folder is not None:
            self.gui_folder.remove()

    def attach(self) -> None:
        server, viewer = self.server, self.viewer

        with server.gui.add_folder("3D Segmentation", expand_by_default=False) as f:
            self.gui_folder = f
            # --- Mode ---
            self.ui["mode_dropdown"] = server.gui.add_dropdown(
                "UI mode",
                options=["simple", "advanced", "expert"],
                initial_value=self.state.get("ui_mode", "simple"),
            )
            # render the selection
            self.ui["show_selection"] = server.gui.add_checkbox(
                "Show selection", initial_value=self.show
            )
            # import / export anchors
            self.ui["import_file_btn"] = server.gui.add_button("Import file")
            self.ui["export_anchors_btn"] = server.gui.add_button("Export anchors")
            self.ui["export_segmentation_btn"] = server.gui.add_button(
                "Export segmentation"
            )
            self.ui["anchor_path"] = server.gui.add_text(
                "Anchor path", initial_value=str("object.anchor")
            )

            # --- Buttons ---
            self.ui["add_pos"] = server.gui.add_button("Add POS anchor")
            self.ui["add_neg"] = server.gui.add_button("Add NEG anchor")
            self.ui["remove_last"] = server.gui.add_button("Remove last anchor")
            self.ui["clear_btn"] = server.gui.add_button("Clear")
            self.ui["refine_btn"] = server.gui.add_button("Refine segmentation")

            # advanced & expert
            self.ui["floaters_btn"] = server.gui.add_button("Remove floaters")
            self.ui["prop_btn"] = server.gui.add_button("Propagate labels")
            self.ui["recluster_btn"] = server.gui.add_button("Recluster anchors")  # NEW

            # --- Feature cosine group ---
            C = int(self.state["C"])
            self.ui["trunc_slider"] = server.gui.add_slider(
                "Truncation D (channels kept)",
                min=1,
                max=max(1, C),
                step=1,
                initial_value=self.state.get("trunc_dim", C),
            )
            # Advanced shared
            self.ui["tau_shared_slider"] = server.gui.add_slider(
                "τ (pos & neg)",
                min=0.7,
                max=1.0,
                step=0.01,
                initial_value=float(self.state.get("tau_pos_rel", 0.9)),
            )
            self.ui["ratio_shared_slider"] = server.gui.add_slider(
                "Ratio (pos & neg)",
                min=0.01,
                max=0.50,
                step=0.01,
                initial_value=float(self.state.get("pos_ratio", 0.05)),
            )
            # Expert individual
            self.ui["tau_pos_slider"] = server.gui.add_slider(
                "Pos threshold (relative 0–1)",
                min=0.7,
                max=1.0,
                step=0.01,
                initial_value=float(self.state.get("tau_pos_rel", 0.9)),
            )
            self.ui["tau_neg_slider"] = server.gui.add_slider(
                "Neg threshold (relative 0–1)",
                min=0.7,
                max=1.0,
                step=0.01,
                initial_value=float(self.state.get("tau_neg_rel", 0.9)),
            )
            self.ui["pos_ratio_slider"] = server.gui.add_slider(
                "Pos ratio (auto τ⁺)",
                min=0.01,
                max=0.50,
                step=0.01,
                initial_value=float(self.state.get("pos_ratio", 0.05)),
            )
            self.ui["neg_ratio_slider"] = server.gui.add_slider(
                "Neg ratio (auto τ⁻)",
                min=0.01,
                max=0.50,
                step=0.01,
                initial_value=float(self.state.get("neg_ratio", 0.05)),
            )

            # --- KNN group ---
            # Visible in advanced & expert
            self.ui["prop_sim_slider"] = server.gui.add_slider(
                "Similarity threshold",
                min=0.5,
                max=1.0,
                step=0.01,
                initial_value=float(self.state.get("prop_sim_thresh", 0.85)),
            )
            self.ui["prop_dist_slider"] = server.gui.add_slider(
                "Max distance (3D)",
                min=0.0,
                max=0.1,
                step=0.01,
                initial_value=float(self.state.get("prop_max_dist", 0.02)),
            )
            # Advanced shared K and iters
            self.ui["k_slider"] = server.gui.add_slider(
                "Neighbors k (shared)",
                min=1,
                max=64,
                step=1,
                initial_value=int(self.state.get("prop_k", 16)),
            )
            self.ui["knn_iter_slider"] = server.gui.add_slider(
                "Propagation iterations (shared)",
                min=1,
                max=20,
                step=1,
                initial_value=int(self.state.get("knn_iters", 5)),
            )
            # Expert individual
            self.ui["prop_iters_slider"] = server.gui.add_slider(
                "Propagate iters",
                min=1,
                max=20,
                step=1,
                initial_value=int(self.state.get("prop_iters", 5)),
            )
            self.ui["floaters_iters_slider"] = server.gui.add_slider(
                "Floaters iters",
                min=1,
                max=20,
                step=1,
                initial_value=int(self.state.get("floaters_iters", 5)),
            )
            self.ui["floaters_k_slider"] = server.gui.add_slider(
                "Neighbors k (floaters)",
                min=1,
                max=16,
                step=1,
                initial_value=int(self.state.get("floaters_k", 16)),
            )
            self.ui["prop_k_slider"] = server.gui.add_slider(
                "Neighbors k (propagate)",
                min=1,
                max=16,
                step=1,
                initial_value=int(self.state.get("prop_k", 16)),
            )

            # ---------- helpers ----------
            def _show(widget, visible: bool):
                if hasattr(widget, "visible"):
                    widget.visible = visible
                elif hasattr(widget, "disabled"):
                    widget.disabled = not visible

            def _apply_ui_mode():
                mode = self.state.get("ui_mode", "simple")

                # Always visible: anchor buttons & action buttons
                for w in (
                    self.ui["add_pos"],
                    self.ui["add_neg"],
                    self.ui["remove_last"],
                    self.ui["clear_btn"],
                    self.ui["refine_btn"],
                    self.ui["import_file_btn"],
                    self.ui["export_anchors_btn"],
                    self.ui["export_segmentation_btn"],
                    self.ui["trunc_slider"],
                    self.ui["tau_shared_slider"],
                ):
                    _show(w, True)

                # advanced & expert
                _show(self.ui["floaters_btn"], mode != "simple")
                _show(self.ui["prop_btn"], mode != "simple")
                _show(self.ui["recluster_btn"], mode != "simple")

                # Feature knobs
                # shared (advanced)
                _show(self.ui["ratio_shared_slider"], mode == "advanced")

                # individual (expert)
                indiv = [
                    self.ui["tau_pos_slider"],
                    self.ui["tau_neg_slider"],
                    self.ui["pos_ratio_slider"],
                    self.ui["neg_ratio_slider"],
                ]
                for w in indiv:
                    _show(w, mode == "expert")

                # KNN base knobs (both adv & expert)
                base_knn = [self.ui["prop_sim_slider"], self.ui["prop_dist_slider"]]
                for w in base_knn:
                    _show(w, mode in ("advanced", "expert"))

                # advanced shared K/iters
                _show(self.ui["k_slider"], mode == "advanced")
                _show(self.ui["knn_iter_slider"], mode == "advanced")

                # expert individual K/iters
                expert_knn = [
                    self.ui["prop_k_slider"],
                    self.ui["floaters_k_slider"],
                    self.ui["prop_iters_slider"],
                    self.ui["floaters_iters_slider"],
                ]
                for w in expert_knn:
                    _show(w, mode == "expert")

                # enforce coupling in advanced
                if mode == "advanced":
                    # couple τ and ratios immediately
                    tau = float(self.ui["tau_shared_slider"].value)
                    self.state["tau_pos_rel"] = tau
                    self.state["tau_neg_rel"] = tau
                    ratio = float(self.ui["ratio_shared_slider"].value)
                    self.state["pos_ratio"] = ratio
                    self.state["neg_ratio"] = ratio
                    # couple Ks & iters
                    k_shared = int(self.ui["k_slider"].value)
                    self.state["prop_k"] = k_shared
                    self.state["floaters_k"] = k_shared
                    iters = int(self.ui["knn_iter_slider"].value)
                    self.state["prop_iters"] = iters
                    self.state["floaters_iters"] = iters

            # ---------- button callbacks ----------
            @self.ui["import_file_btn"].on_click
            def _(_e):
                path = self.ui["anchor_path"].value.strip()
                if path == "":
                    self._log("Please specify a valid anchor path.")
                    return
                try:
                    self.import_files(path)
                    self._log(f"Imported anchors from {path}.")
                except Exception as ex:
                    self._log(f"Failed to import anchors: {ex}")

            @self.ui["export_anchors_btn"].on_click
            def _(_e):
                path = self.ui["anchor_path"].value.strip()
                if path == "":
                    path = "object.anchor"
                    self._log(f"No path specified. Using default: {path}")
                try:
                    self.export_anchors(path)
                    self._log(f"Exported anchors to {path}.")
                except Exception as ex:
                    self._log(f"Failed to export anchors: {ex}")

            @self.ui["export_segmentation_btn"].on_click
            def _(_e):
                path = self.ui["anchor_path"].value.strip()
                if path == "":
                    path = "object.seg"
                    self._log(f"No path specified. Using default: {path}")
                try:
                    self.export_segmentation(path)
                except Exception as ex:
                    self._log(f"Failed to export segmentation: {ex}")

            @self.ui["add_pos"].on_click
            def _(event):
                self.ui["add_pos"].disabled = True
                self.set_show(True)

                @event.client.scene.on_pointer_event(event_type="rect-select")
                def _on_rect(e):
                    event.client.scene.remove_pointer_callback()
                    self._fresh_feature_render_and_anchor(e, positive=True)
                    self.apply_segmentation()
                    self.viewer.rerender(None)
                    self.ui["add_pos"].disabled = False

                @event.client.scene.on_pointer_callback_removed
                def _():
                    self.ui["add_pos"].disabled = False

            @self.ui["add_neg"].on_click
            def _(event):
                self.ui["add_neg"].disabled = True

                @event.client.scene.on_pointer_event(event_type="rect-select")
                def _on_rect(e):
                    event.client.scene.remove_pointer_callback()
                    self._fresh_feature_render_and_anchor(e, positive=False)
                    self.apply_segmentation()
                    self.viewer.rerender(None)
                    self.ui["add_neg"].disabled = False

                @event.client.scene.on_pointer_callback_removed
                def _():
                    self.ui["add_neg"].disabled = False

            @self.ui["remove_last"].on_click
            def _(_e):
                if len(self.state["last_anchor_pos"]) == 0:
                    self._log("No anchors to remove.")
                    return
                last_pos = self.state["last_anchor_pos"].pop()
                if last_pos and len(self.state["pos_anchors"]) > 0:
                    self.state["pos_anchors"].pop()
                elif (not last_pos) and len(self.state["neg_anchors"]) > 0:
                    self.state["neg_anchors"].pop()
                n_pos = len(self.state["pos_anchors"])
                n_neg = len(self.state["neg_anchors"])
                self._log(f"Removed last anchor. Now P={n_pos}, N={n_neg}.")
                if n_pos > 0:
                    self.apply_segmentation()
                else:
                    Fpts = self._get_point_features()
                    self.splats["selection"] = torch.zeros(
                        (Fpts.shape[0], 1), dtype=Fpts.device, device=Fpts.device
                    )
                    self.set_show(False)
                self.viewer.rerender(None)

            @self.ui["clear_btn"].on_click
            def _(_e):
                self.state["pos_anchors"].clear()
                self.state["neg_anchors"].clear()
                self.state["last_anchor_pos"].clear()
                self._log("Cleared anchors.")
                Fpts = self._get_point_features()
                self.splats["selection"] = torch.zeros(
                    (Fpts.shape[0], 1), dtype=Fpts.dtype, device=Fpts.device
                )
                self.set_show(False)
                if self._viewer_bounding_elipsoid is not None:
                    self._viewer_bounding_elipsoid.remove()
                self._viewer_bounding_elipsoid = None
                self.viewer.rerender(None)

            @self.ui["recluster_btn"].on_click
            def _(_e):
                self.recluster_anchors(k_pos=-1, k_neg=-1)
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["show_selection"].on_update
            def _(e):
                self.set_show(bool(e.target.value))

            # ---------- feature knob callbacks ----------
            @self.ui["trunc_slider"].on_update
            def _(e):
                C = int(self.state["C"])
                D = int(e.target.value)
                self.state["trunc_dim"] = None if D >= C else D
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["tau_pos_slider"].on_update
            def _(e):
                self.state["tau_pos_rel"] = float(e.target.value)
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["tau_neg_slider"].on_update
            def _(e):
                self.state["tau_neg_rel"] = float(e.target.value)
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["pos_ratio_slider"].on_update
            def _(e):
                self.state["pos_ratio"] = float(e.target.value)
                self.state["tau_pos_rel"] = None  # switch to auto τ via ratio
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["neg_ratio_slider"].on_update
            def _(e):
                self.state["neg_ratio"] = float(e.target.value)
                self.state["tau_neg_rel"] = None  # switch to auto τ via ratio
                self.apply_segmentation()
                self.viewer.rerender(None)

            # Advanced shared callbacks
            @self.ui["tau_shared_slider"].on_update
            def _(e):
                tau = float(e.target.value)
                self.state["tau_pos_rel"] = tau
                self.state["tau_neg_rel"] = tau
                self.apply_segmentation()
                self.viewer.rerender(None)

            @self.ui["ratio_shared_slider"].on_update
            def _(e):
                ratio = float(e.target.value)
                self.state["pos_ratio"] = ratio
                self.state["neg_ratio"] = ratio
                # enable auto-τ behavior
                self.state["tau_pos_rel"] = None
                self.state["tau_neg_rel"] = None
                self.apply_segmentation()
                self.viewer.rerender(None)

            # ---------- knn knob callbacks ----------
            @self.ui["prop_sim_slider"].on_update
            def _(e):
                self.state["prop_sim_thresh"] = float(e.target.value)

            @self.ui["prop_dist_slider"].on_update
            def _(e):
                self.state["prop_max_dist"] = float(e.target.value)

            # Advanced shared K/iters
            @self.ui["k_slider"].on_update
            def _(e):
                k = int(e.target.value)
                self.state["prop_k"] = k
                self.state["floaters_k"] = k

            @self.ui["knn_iter_slider"].on_update
            def _(e):
                iters = int(e.target.value)
                self.state["prop_iters"] = iters
                self.state["floaters_iters"] = iters

            # Expert individual K/iters
            @self.ui["prop_k_slider"].on_update
            def _(e):
                self.state["prop_k"] = int(e.target.value)

            @self.ui["floaters_k_slider"].on_update
            def _(e):
                self.state["floaters_k"] = int(e.target.value)

            @self.ui["prop_iters_slider"].on_update
            def _(e):
                self.state["prop_iters"] = int(e.target.value)

            @self.ui["floaters_iters_slider"].on_update
            def _(e):
                self.state["floaters_iters"] = int(e.target.value)

            # ---------- action buttons (respect mode) ----------
            @self.ui["prop_btn"].on_click
            def _(_e):
                iters = int(self.state["prop_iters"])
                n = self.propagate_labels(iters)
                self._log(f"Propagated labels to {n} labels.")
                self.viewer.rerender(None)

            @self.ui["floaters_btn"].on_click
            def _(_e):
                iters = int(self.state["floaters_iters"])
                n = self.remove_floaters(iters)
                self._log(f"Floater removal changed {n} labels.")
                self.viewer.rerender(None)

            @self.ui["refine_btn"].on_click
            def _(_e):
                self.auto_refine()

            # ---------- mode dropdown ----------
            @self.ui["mode_dropdown"].on_update
            def _(e):
                self.state["ui_mode"] = e.target.value
                _apply_ui_mode()
                self.viewer.rerender(None)

            # initialize visibility/coupling once
            _apply_ui_mode()

    def set_show(self, show: bool) -> None:
        if self.show == show:
            return
        self.show = show
        if "show_selection" in self.ui and self.ui["show_selection"].value != show:
            self.ui["show_selection"].value = show
        if self._viewer_bounding_elipsoid is not None:
            self._viewer_bounding_elipsoid.visible = show
        self.viewer.rerender(None)

    def auto_refine(self) -> None:
        prop_iters = int(self.state["prop_iters"])
        floaters_iters = int(self.state["floaters_iters"])
        n_ = 0
        # NOTE: Hardcoded max iterations of (floaters + prop) for now
        for _ in range(5):
            n = self.propagate_labels(prop_iters)
            n += self.remove_floaters(floaters_iters)
            n_ += n
            if n == 0:
                break
        self.compute_bounding_ellipsoid()
        self._log(f"Refinement changed {n_} labels.")
        self.viewer.rerender(None)

    def get_anchors(
        self, from_gaussians: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current anchors as (pos, neg) as (N, C) tensors."""
        pos = (
            torch.stack(self.state["pos_anchors"], dim=0)
            if len(self.state["pos_anchors"]) > 0
            else torch.empty((0, self.state["C"]))
        )
        neg = (
            torch.stack(self.state["neg_anchors"], dim=0)
            if len(self.state["neg_anchors"]) > 0
            else torch.empty((0, self.state["C"]))
        )
        return pos, neg

    def set_anchors(self, anchors: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set current anchors from (pos, neg) as (N, C) tensors."""
        pos, neg = anchors
        if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
            self._log("Invalid anchor shapes.")
            return
        if self.state["C"] is not None and (
            pos.shape[1] != self.state["C"] or neg.shape[1] != self.state["C"]
        ):
            self._log("Anchor feature dimension mismatch.")
            return
        self.state["pos_anchors"] = [pos[i] for i in range(pos.shape[0])]
        self.state["neg_anchors"] = [neg[i] for i in range(neg.shape[0])]
        self.state["last_anchor_pos"] = [True] * len(self.state["pos_anchors"]) + [
            False
        ] * len(self.state["neg_anchors"])
        self._log(
            f"Set anchors. Now P={len(self.state['pos_anchors'])}, N={len(self.state['neg_anchors'])}."
        )
        if len(self.state["pos_anchors"]) > 0:
            self.apply_segmentation()
            self.auto_refine()
            self.set_show(True)

    def export_anchors(self, path: str | Path) -> None:
        """Export current anchors as torch tensors"""
        pos, neg = self.get_anchors()
        export_dict = {
            "pos_anchors": pos.detach().cpu(),
            "neg_anchors": neg.detach().cpu(),
        }
        path = Path(path)
        if path.suffix != ".anchor":
            path = path.with_suffix(".anchor")
        if path.parent != "":
            os.makedirs(path.parent, exist_ok=True)
        torch.save(export_dict, str(path))

    def export_segmentation(self, path: str | Path) -> None:
        """Export current segmentation as torch tensors"""
        selection = self.splats.get("selection", None)
        if selection is None:
            self._log("No segmentation to export.")
            return
        export_dict = {
            "selection": selection.detach().cpu(),
        }
        path = Path(path)
        if not path.is_absolute():
            # save to scene checkpoint directory if relative
            scene_dir = Path(self.viewer.scene_ckpt_dir).parent
            path = scene_dir / path
        if path.suffix != ".seg":
            path = path.with_suffix(".seg")
        if path.parent != "":
            os.makedirs(path.parent, exist_ok=True)
        self._log(f"Exporting segmentation to {path}.")
        torch.save(export_dict, str(path))

    def import_files(self, path: str | Path) -> bool:
        """Import anchor or segmentation files based on extension"""
        path = Path(path)
        if not path.is_file():
            # try to find in config directory
            cfg_dir = Path(__file__).parent.parent / "configs"
            path = cfg_dir / path
        if not path.is_file():
            self._log(f"File {path} does not exist.")
            return False
        if path.suffix == ".anchor":
            return self.import_anchors(path)
        if path.suffix == ".seg":
            return self.import_segmentation(path)
        self._log(f"Unsupported file extension: {path.suffix}. Trying as anchors.")
        return self.import_anchors(path)

    def import_segmentation(self, path: str | Path) -> bool:
        """Import segmentation from torch tensors saved via export_segmentation()"""
        if not Path(path).is_file():
            self._log(f"Segmentation file {path} does not exist.")
            return False
        data = torch.load(str(path))
        selection = data.get("selection", None)
        if selection is None:
            self._log("Invalid segmentation file.")
            return False
        number_of_points = self.splats.get("means", None).shape[0]
        if selection.shape[0] != number_of_points:
            self._log(
                f"Segmentation size mismatch: expected {number_of_points}, got {selection.shape[0]}."
            )
            return False
        self.splats["selection"] = selection.to(
            self.splats["means"].device, dtype=self.splats["means"].dtype
        )
        self._log("Imported segmentation.")
        self.compute_bounding_ellipsoid()
        self.set_show(True)
        return True

    def import_anchors(self, path: str | Path) -> bool:
        """Import anchors from torch tensors saved via export_anchors()"""
        if not Path(path).is_file():
            self._log(f"Anchor file {path} does not exist.")
            return False
        data = torch.load(str(path))
        pos = data.get("pos_anchors", None)
        neg = data.get("neg_anchors", None)
        if pos is None or neg is None:
            self._log("Invalid anchor file.")
            return False
        self.set_anchors((pos.to(torch.float32).cuda(), neg.to(torch.float32).cuda()))
        return True

    @torch.no_grad()
    def set_bounding_ellipsoid(self, center: torch.Tensor, axes: torch.Tensor) -> None:
        # Record
        self._bounding_elipsoid = (center, axes)

        # Remove any previous visualization
        if self._viewer_bounding_elipsoid is not None:
            try:
                self.viewer.server.scene.remove(self._viewer_bounding_elipsoid)
                self.viewer.server.scene.remove(self._viewer_bounding_coords)
            except Exception:
                pass
            self._viewer_bounding_elipsoid = None
            self._viewer_bounding_coords = None

        if center is None or axes is None:
            return

        try:
            with torch.no_grad():
                c = center.detach().cpu().float().view(3).numpy()
                A = (
                    axes.detach().cpu().float().numpy()
                )  # (3,3) rows = axis vectors (already scaled)

                # Parameterize a unit sphere with a modest resolution
                rings = 18  # latitude samples (>=3)
                sectors = 28  # longitude samples (>=3)

                u = np.linspace(0.0, np.pi, rings)  # polar angle
                v = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)  # azimuth
                uu, vv = np.meshgrid(u, v, indexing="ij")

                # Unit sphere directions
                xs = np.sin(uu) * np.cos(vv)
                ys = np.sin(uu) * np.sin(vv)
                zs = np.cos(uu)
                sph = np.stack([xs, ys, zs], axis=-1)  # (rings, sectors, 3)

                # Map to ellipsoid: linear combination of axis vectors (rows of A)
                # point = c + x*a1 + y*a2 + z*a3
                ellip = (
                    sph[..., 0, None] * A[0]
                    + sph[..., 1, None] * A[1]
                    + sph[..., 2, None] * A[2]
                ) + c  # (rings, sectors, 3)

                verts = ellip.reshape(-1, 3).astype(np.float32)

                # Build triangle faces from the lat/long grid
                faces = []
                for i in range(rings - 1):
                    base0 = i * sectors
                    base1 = (i + 1) * sectors
                    for j in range(sectors):
                        jn = (j + 1) % sectors
                        v0 = base0 + j
                        v1 = base1 + j
                        v2 = base1 + jn
                        v3 = base0 + jn
                        faces.append((v0, v1, v2))
                        faces.append((v0, v2, v3))
                faces = np.asarray(faces, dtype=np.int32)

                # Add mesh to viewer
                self._viewer_bounding_elipsoid = (
                    self.viewer.server.scene.add_mesh_simple(
                        name="bounding_ellipsoid",
                        vertices=verts,
                        faces=faces,
                        color=(100, 200, 100),
                        opacity=0.25,
                        side="double",
                        wireframe=False,
                        flat_shading=False,
                    )
                )
                quat = rotmat_to_quat(torch.from_numpy(A.T))  # columns of A are the rotation axes
                self._viewer_bounding_coords = (
                    self.viewer.server.scene.add_frame(
                        name="bounding_ellipsoid_axes",
                        position=c,
                        wxyz=(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()),
                    )
                )
                self.viewer.rerender(None)
        except Exception as ex:
            self._log(f"Ellipsoid viz failed: {ex}")

    @torch.no_grad()
    def compute_bounding_ellipsoid(
        self, percentile: float = 0.99, enlarge_factor: float = 1.2, z_axis: Optional[torch.Tensor] = None
    ) -> None:
        """
        Fit a weighted bounding ellipsoid to currently SELECTED splats.

        center:
            Weighted center of mass (weights ~ volume * opacity).
        axes:
            Three (scaled) axis vectors a1,a2,a3 (rows of shape (3,)) obtained from the
            weighted covariance (principal components). Each axis length is the
            given percentile of |projection| of points onto that eigenvector,
            then multiplied by enlarge_factor.

        percentile:
            Percentile (0–1] used to define half-length along each principal axis.
            (e.g. 0.95 ≡ covers 95% (by distance) of points along that axis)
        enlarge_factor:
            Post scaling inflation factor for all three half-lengths.
        """
        indices = self.splats.get("selection", None)  # (N,1)
        if indices is None:
            self._log("No selection tensor present.")
            return
        selected = (indices > 0).squeeze(-1)
        if selected.sum() == 0:
            self._log("No splats selected. Cannot set bounding ellipsoid.")
            return

        means = self.splats["means"][selected]  # (M,3)
        scales = self.splats["scales"][selected]  # (M,3)
        opacities = self.splats["opacities"][selected]  # (M,1) or (M,)
        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)

        # Weights ~ (approx volume) * opacity
        weights = (scales.prod(dim=1) * opacities).clamp_min(1e-8)
        weights = weights / weights.sum()

        # Weighted center of mass
        center = (weights[:, None] * means).sum(dim=0)  # (3,)

        # Centered coordinates
        X = means - center[None, :]  # (M,3)

        # Weighted covariance (no 1/(sum w) needed beyond normalization above)
        # cov = Σ w_i x_i x_i^T
        cov = (X * weights[:, None]).T @ X  # (3,3)

        # Eigen decomposition (principal axes). eigh -> ascending order.
        evals, evecs = torch.linalg.eigh(cov)  # evals asc
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]  # columns are eigenvectors

        # For each principal direction, compute percentile of absolute projections
        axes = []
        for k in range(3):
            v = evecs[:, k]
            # Normalize (guard against numerical drift)
            v = v / v.norm().clamp_min(1e-12)
            proj = X @ v  # (M,)
            # Use absolute distance from center along this axis
            half_len = torch.quantile(proj.abs(), percentile).item()
            # Fallback if degenerate
            if not np.isfinite(half_len) or half_len < 1e-9:
                half_len = evals[k].clamp_min(0).sqrt().item() + 1e-6
            half_len *= enlarge_factor
            axes.append(v * half_len)

        axes = torch.stack(axes, dim=0)  # (3,3) rows are scaled axis vectors (x, y, z order)

        if z_axis is not None:
            z_axis = z_axis.to(axes.device, dtype=axes.dtype)
            # Override the z-axis with the provided vector (after scaling to match half-length)
            z_axis = z_axis / z_axis.norm().clamp_min(1e-12)
            half_len_z = axes[2].norm().item()
            axes[2] = z_axis * half_len_z
            # Recompute the y axis to be orthogonal to the new z-axis while keeping the same x-axis direction
            x_axis = axes[0] / axes[0].norm().clamp_min(1e-12)
            y_axis = torch.cross(z_axis, x_axis, dim=0)
            y_axis = y_axis / y_axis.norm().clamp_min(1e-12)
            half_len_y = axes[1].norm().item()
            axes[1] = y_axis * half_len_y
            # Recompute the x axis to be orthogonal to the new y and z axes
            x_axis = torch.cross(y_axis, z_axis, dim=0)
            x_axis = x_axis / x_axis.norm().clamp_min(1e-12)
            half_len_x = axes[0].norm().item()  
            axes[0] = x_axis * half_len_x

        self.set_bounding_ellipsoid(center=center, axes=axes)
        # self._log(f"Set bounding ellipsoid: center={center.detach().cpu().numpy()}, axes={axes.detach().cpu().numpy()}")
        self.viewer.rerender(None)

    # --------------------------------------------------------------------- #
    # Public: run segmentation and write to splats["selection"]
    # --------------------------------------------------------------------- #
    @staticmethod
    def _anchor_from_rect_pool(
        feat_hwc: torch.Tensor,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        method: str = "geom_median",
        **kwargs,
    ) -> torch.Tensor:
        """
        Pool features inside [y0:y1, x0:x1) using the chosen method, then L2-normalize.
        kwargs are forwarded to _pool_vectors (e.g., iters=20 for geom_median).
        """
        H, W, C = feat_hwc.shape
        y0, y1 = max(0, y0), min(H, y1)
        x0, x1 = max(0, x0), min(W, x1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        if x1 <= x0:
            x1 = min(W, x0 + 1)

        # Flatten spatial region to M×C
        v = feat_hwc[y0:y1, x0:x1].reshape(-1, C)
        v = SegmentationModule3D._pool_vectors(v, method=method, **kwargs)
        return v.detach().clone().contiguous()

    # ---------- rectangle anchor picking ----------
    def _current_resolution(self, client):
        cam = self.viewer.get_camera_state(client)
        H = int(self.state["render_res"])
        return (int(H * cam.aspect), H)

    @torch.no_grad()
    def _fresh_feature_render_and_anchor(self, event, positive: bool):
        """Fresh 2D feature render, then compute rect-mean anchor from the user rect."""
        client = event.client
        cam_state = self.viewer.get_camera_state(client)
        res = self._current_resolution(client)

        # Get features (H,W,C) float32 HWC
        feats = self.viewer._feature_render_fn(cam_state, res, raw=True)
        feat_hwc = self._to_hwc(feats)

        # Rect to anchor (rectangle mean)
        (x0, y0), (x1, y1) = event.screen_pos
        H, W, C2 = feat_hwc.shape
        if C2 != int(self.state["C"]):
            self._log(f"Warning: 2D feat C={C2} != point feat C={self.state['C']}.")
        y0i, y1i, x0i, x1i = self._rect_to_bounds(H, W, (x0, y0), (x1, y1))
        vec = self._anchor_from_rect_pool(feat_hwc, y0i, y1i, x0i, x1i)  # (C,)
        if positive:
            self.state["pos_anchors"].append(vec)
        else:
            self.state["neg_anchors"].append(vec)
        self.state["last_anchor_pos"].append(positive)

        self._log(
            f"Added {'POS' if positive else 'NEG'} anchor. \nNow P={len(self.state['pos_anchors'])}, N={len(self.state['neg_anchors'])}."
        )

    @torch.no_grad()
    def apply_segmentation(self) -> Dict[str, torch.Tensor | float]:
        """
        Computes per-point similarities to POS/NEG anchors (with truncation & renorm),
        applies relative thresholds (or auto via ratios), and writes:
          splats["selection"][i] =  1  if pos_pass and not neg_pass
                                = -1  if neg_pass
                                =  0  otherwise
        Returns diagnostic dict with thresholds and raw/normalized scores.
        """
        Fpts = self._get_point_features()  # (N,C) float32
        N, C = Fpts.shape

        # Truncate + renormalize features
        Fpts_tr = self._truncate_and_unit_normalize(Fpts, self.state["trunc_dim"])

        # Prepare anchors in same truncated space
        def _stack(lst: List[torch.Tensor]) -> Optional[torch.Tensor]:
            if len(lst) == 0:
                return None
            A = torch.stack(lst, dim=0).to(Fpts_tr.device, Fpts_tr.dtype)
            if self.state["trunc_dim"] is not None and int(self.state["trunc_dim"]) < C:
                D = int(self.state["trunc_dim"])
                A = A.clone()
                A[..., D:] = 0.0
            return F.normalize(A, dim=-1, eps=1e-6)

        posA = _stack(self.state["pos_anchors"])
        negA = _stack(self.state["neg_anchors"])

        # handle no positive anchors
        if posA is None or posA.numel() == 0:
            # nothing to do
            selection = torch.zeros((N, 1), dtype=Fpts.dtype, device=Fpts.device)
            self.splats["selection"] = selection
            return {
                "tau_pos_rel": None,
                "tau_neg_rel": None,
                "pos_raw": None,
                "neg_raw": None,
                "pos_rel": None,
                "neg_rel": None,
                "selection": selection,
            }

        # Cosine maps (raw in [-1,1]) – take max over anchors
        pos_raw = (Fpts_tr @ posA.t()).amax(dim=1)  # (N,)
        if negA is None:
            neg_raw = torch.full_like(pos_raw, -1.0)
        else:
            neg_raw = (Fpts_tr @ negA.t()).amax(dim=1)  # (N,)

        # Normalize to [0,1] by percentiles (robust)
        pos_rel = self._percentile_norm(pos_raw, self.pmin)
        neg_rel = self._percentile_norm(neg_raw, self.pmin)

        # Thresholds (fixed relative τ or auto via top-k)
        tau_pos_rel = self.state["tau_pos_rel"]
        tau_neg_rel = self.state["tau_neg_rel"]
        if tau_pos_rel is None:
            k_pos = max(
                1,
                int(round(N * self._clamp_ratio(float(self.state["pos_ratio"]), N, 1))),
            )
            tau_pos_rel = torch.topk(pos_rel, k=k_pos).values[-1].item()
        if tau_neg_rel is None:
            k_neg = max(
                1,
                int(round(N * self._clamp_ratio(float(self.state["neg_ratio"]), N, 1))),
            )
            tau_neg_rel = torch.topk(neg_rel, k=k_neg).values[-1].item()

        pos_pass = pos_rel >= float(tau_pos_rel)
        neg_pass = (
            neg_rel >= float(tau_neg_rel)
            if (negA is not None and negA.numel() > 0)
            else torch.zeros_like(pos_pass, dtype=torch.bool)
        )

        selection = torch.zeros((N, 1), dtype=Fpts_tr.dtype, device=Fpts_tr.device)

        only_pos = pos_pass & (~neg_pass)
        only_neg = (~pos_pass) & neg_pass
        both = pos_pass & neg_pass

        # straightforward cases
        selection[only_pos] = 1.0
        selection[only_neg] = -1.0

        # conflict: choose the side with the larger *relative* similarity
        # (ties go to NEG only if exactly equal)
        if both.any():
            selection[both, 0] = torch.where(
                pos_rel[both] > neg_rel[both],
                torch.tensor(1.0, device=Fpts_tr.device),
                torch.tensor(-1.0, device=Fpts_tr.device),
            )

        # Write to splats (preserve original container type; keep torch)
        self.splats["selection"] = selection

        # Set the bounding ellipsoid to the selected splats
        self.compute_bounding_ellipsoid()

        return {
            "tau_pos_rel": float(tau_pos_rel),
            "tau_neg_rel": float(tau_neg_rel),
            "pos_raw": pos_raw.detach(),
            "neg_raw": neg_raw.detach(),
            "pos_rel": pos_rel.detach(),
            "neg_rel": neg_rel.detach(),
            "selection": selection,
        }

    @torch.no_grad()
    def propagate_labels(self, k_max=1) -> dict:
        n_updated = 0
        for i in range(k_max):
            res = self.propagate_labels_iter()
            updated = int(res.get("updated", 0))
            n_updated += updated
            if updated == 0:
                break
        return n_updated

    @torch.no_grad()
    def propagate_labels_iter(self) -> dict:
        """
        PyTorch3D-based propagation:
        - kNN in 3D with pytorch3d.ops.knn_points
        - keep neighbors within max 3D distance AND above cosine similarity threshold
        - conflicts resolved by higher similarity; ties -> NEG wins
        - protects opposite-signed seeds from being overwritten
        """
        means = torch.as_tensor(self.splats["means"])  # (N,3)
        feats = F.normalize(self._get_point_features(), dim=-1, eps=1e-6)  # (N,C)
        N = means.shape[0]

        sel = torch.as_tensor(self.splats["selection"]).view(-1)  # (N,)
        seeds_pos = torch.nonzero(sel == 1, as_tuple=False).flatten()
        seeds_neg = torch.nonzero(sel == -1, as_tuple=False).flatten()

        if seeds_pos.numel() == 0 and seeds_neg.numel() == 0:
            self._log("propagate_labels: no +/- seeds to expand from.")
            return {"updated": 0}

        k = int(self.state["prop_k"])
        sim_thr = float(self.state["prop_sim_thresh"])
        dmax = float(self.state["prop_max_dist"])
        dmax2 = dmax * dmax

        # buffers of best proposed similarity per target index (init: no proposal)
        prop_pos_sim = torch.full((N,), -1.0, dtype=feats.dtype, device=feats.device)
        prop_neg_sim = torch.full((N,), -1.0, dtype=feats.dtype, device=feats.device)

        def propose_from_side(
            seed_idx: torch.Tensor, into_sim_buf: torch.Tensor
        ) -> int:
            """Use knn_points on seeds -> means; write max(sim) per neighbor into into_sim_buf."""
            if seed_idx.numel() == 0:
                return 0

            q_feat = feats[seed_idx]  # (S,C), L2-normalized

            # PyTorch3D knn: shapes (1,S,k) after removing batch dim
            # d2: squared L2 distances; idx: neighbor indices into means
            idx_all, d2_all = self._get_knn(k)  # (N,k), (N,k)
            idx = idx_all[seed_idx]  # (S,k)
            d2 = d2_all[seed_idx]  # (S,k)

            # (optional) exclude self-matches if any seeds are present in the ref set
            if seed_idx.numel() > 0:
                # broadcast seed indices for comparison
                self_mask = idx == seed_idx.view(-1, 1)
            else:
                self_mask = torch.zeros_like(d2, dtype=torch.bool)

            within_dist = (d2 <= dmax2) & (~self_mask)

            # gather neighbor features -> (S,k,C), compute cosine to q_feat
            nbr_feat = feats[idx]  # advanced indexing makes (S,k,C)
            sim = (nbr_feat * q_feat[:, None, :]).sum(
                dim=-1
            )  # (S,k); feats are unit-norm

            good = within_dist & (sim >= sim_thr)
            if not good.any():
                return 0

            # compress valid proposals
            s_idx, k_idx = torch.nonzero(good, as_tuple=True)
            targ = idx[s_idx, k_idx]  # neighbor ids (flattened list)
            vals = sim[s_idx, k_idx]  # their proposed sims

            # write max(sim) per target using scatter_reduce if available; else fallback
            if hasattr(torch, "scatter_reduce"):
                into_sim_buf.scatter_reduce_(
                    0, targ, vals, reduce="amax", include_self=True
                )
            else:
                order = torch.argsort(targ)
                targ = targ[order]
                vals = vals[order]
                start = 0
                T = targ.numel()
                while start < T:
                    end = start + 1
                    while end < T and targ[end].item() == targ[start].item():
                        end += 1
                    into_sim_buf[targ[start]] = torch.maximum(
                        into_sim_buf[targ[start]], vals[start:end].max()
                    )
                    start = end

            return int(targ.numel())

        num_prop_pos = propose_from_side(seeds_pos, prop_pos_sim)
        num_prop_neg = propose_from_side(seeds_neg, prop_neg_sim)

        # acceptors (points that received at least one valid proposal)
        acc_pos = prop_pos_sim >= 0.0
        acc_neg = prop_neg_sim >= 0.0

        # decide sign per point: compare max proposed sims; NEG wins on ties
        choose_pos = acc_pos & (~acc_neg | (prop_pos_sim > prop_neg_sim))
        choose_neg = acc_neg & (~acc_pos | (prop_neg_sim >= prop_pos_sim))

        # protect opposite seeds from overwrite
        protect_pos = sel == 1
        protect_neg = sel == -1
        choose_pos &= ~protect_neg
        choose_neg &= ~protect_pos

        before = sel.clone()
        sel[choose_pos & (sel <= 0)] = 1
        sel[choose_neg] = -1  # negative wins

        self.splats["selection"] = sel.view(N, 1)

        updated = int((sel != before).sum().item())

        return {
            "updated": updated,
            "num_prop_pos": num_prop_pos,
            "num_prop_neg": num_prop_neg,
            "k": k,
        }

    @torch.no_grad()
    def remove_floaters(self, k_max=1) -> dict:
        n_updated = 0
        for i in range(k_max):
            res = self.remove_floaters_iter()
            updated = int(res.get("updated", 0))
            n_updated += updated
            if updated == 0:
                break
        return n_updated

    @torch.no_grad()
    def remove_floaters_iter(self, tie_policy: str = "keep") -> dict:
        """
        Reassign each point to the majority label among its k nearest neighbors (excluding self).
        Labels are in {-1, 0, 1}; if most neighbors are 0, the point becomes 0.
        tie_policy:
        - "keep": keep original label on ties
        - "zero": force 0 on ties
        - "negwins": prefer -1 > 0 > +1 on ties (deterministic)
        """
        sel = torch.as_tensor(
            self.splats["selection"],
            dtype=torch.float32,
            device=self.splats["means"].device,
        ).view(
            -1
        )  # (N,)
        N = sel.numel()
        if N == 0:
            return {"updated": 0}

        k = int(self.state["floaters_k"])
        k = max(1, min(k, max(1, N - 1)))  # exclude self; need at least 1 neighbor

        # Use cached kNN once; neighbors 1..k (skip self at [:,0])
        idx_all, _ = self._get_knn(k + 1)
        nbr_idx = idx_all[:, 1 : (k + 1)]  # (N, k)

        # Neighbor labels: (N, k) in {-1,0,1}
        nbr = sel[nbr_idx].to(torch.int8)

        # Counts per class [-1,0,1] -> columns [0,1,2]
        cnt_neg = (nbr == -1).sum(dim=1)
        cnt_zero = (nbr == 0).sum(dim=1)
        cnt_pos = (nbr == 1).sum(dim=1)
        counts = torch.stack([cnt_neg, cnt_zero, cnt_pos], dim=1)  # (N,3)

        # Majority winner index in [0,1,2]
        winners = counts.argmax(dim=1)  # breaks ties by first occurrence

        # Optional tie handling
        if tie_policy in ("keep", "zero", "negwins"):
            maxc = counts.gather(1, winners.unsqueeze(1)).squeeze(1)
            ties = (counts == maxc.unsqueeze(1)).sum(dim=1) > 1
            if ties.any():
                if tie_policy == "keep":
                    # map current sel {-1,0,1} -> [0,1,2] then back below
                    cur_idx = (sel.to(torch.int64) + 1).clamp(0, 2)  # -1->0, 0->1, 1->2
                    winners[ties] = cur_idx[ties]
                elif tie_policy == "zero":
                    winners[ties] = 1  # index of label 0
                elif tie_policy == "negwins":
                    # enforce priority order: -1 > 0 > +1
                    # Already true with argmax if we bias counts: add tiny eps decreasing with class index
                    eps = torch.tensor([3e-7, 2e-7, 1e-7], device=counts.device)
                    winners = (counts + eps).argmax(dim=1)

        # Map [0,1,2] -> [-1,0,1]
        label_map = torch.tensor([-1, 0, 1], device=sel.device, dtype=torch.int8)
        new_sel = label_map[winners].to(torch.float32)

        updated = int((new_sel != sel).sum().item())
        self.splats["selection"] = new_sel.view(N, 1)
        return {"updated": updated, "k": k}

    def recluster_anchors(
        self,
        k_pos: Optional[int] = None,  # omit or -1 for auto
        k_neg: Optional[int] = None,  # -1 for auto, omit to keep current neg anchors
        *,
        seed: Optional[int] = None,
        reinit_empty: bool = True,
        center_method: str = "geom_median",
        shift_factor_pos: float = 0.1,
        shift_factor_neg: float = 0.1,
        apply: bool = True,
    ) -> None:
        """
        Recompute POS (and NEG) anchors by clustering currently selected splats.

        Improvements:
        - Initialize clusters from existing anchors (robust warm start).
        - Limit how far each anchor may move from its original position:
              max_shift = shift_factor * std(selection_features)
          where std is sqrt(mean(||x - mean||^2)) over the (pos/neg) selected set.
        - Uses robust center updates (geom_median default).

        shift_factor_*:
            0 -> keep original anchors (no movement)
            1 -> allow movement up to one global std of that selection set
        """
        if "selection" not in self.splats:
            self._log("recluster_anchors: no selection tensor.")
            return
        feat_all = self._get_point_features()  # (N,C)
        if feat_all.numel() == 0:
            self._log("recluster_anchors: empty feature set.")
            return

        sel = self.splats["selection"].view(-1)
        pos_mask = sel > 0.5
        neg_mask = sel < -0.5
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos == 0:
            self._log("recluster_anchors: no positive points to cluster.")
            return

        trunc_dim = self.state.get("trunc_dim", None)
        feats_proc = self._truncate_and_unit_normalize(feat_all, trunc_dim)

        # --- POSITIVE ---
        pos_feats = feats_proc[pos_mask]  # (M_pos,C)
        if k_pos is None or k_pos < 0:
            k_pos = len(self.state["pos_anchors"]) or 4
        k_pos = max(1, min(k_pos, pos_feats.shape[0]))

        # Compute global std of positive selection
        pos_mean = pos_feats.mean(dim=0)
        pos_var = ((pos_feats - pos_mean) ** 2).sum(dim=1).mean()
        pos_std = float(torch.sqrt(pos_var.clamp_min(0.0)).detach().cpu())

        # Build init centers from existing anchors (truncate + normalize)
        init_pos = None
        if len(self.state["pos_anchors"]) > 0:
            init_list = []
            for a in self.state["pos_anchors"][:k_pos]:
                vec = a.clone()
                if trunc_dim is not None and int(trunc_dim) < vec.numel():
                    D = int(trunc_dim)
                    v2 = torch.zeros_like(vec)
                    v2[:D] = vec[:D]
                    vec = v2
                init_list.append(F.normalize(vec, dim=-1, eps=1e-6))
            # If fewer than k_pos, pad with random positives
            if len(init_list) < k_pos:
                need = k_pos - len(init_list)
                rand_idx = torch.randperm(pos_feats.shape[0])[:need]
                init_list.extend([pos_feats[i].clone() for i in rand_idx])
            init_pos = torch.stack(init_list, dim=0)

        max_shift_pos = shift_factor_pos * pos_std if shift_factor_pos > 0 else 0.0

        try:
            pos_centers = SegmentationModule3D.cosine_kmeans(
                pos_feats,
                k_pos,
                seed=seed,
                reinit_empty=reinit_empty,
                center_method=center_method,
                init_centers=init_pos,
                max_shift=max_shift_pos,
            )
        except Exception as e:
            self._log(f"recluster_anchors: POS clustering failed ({e}).")
            return
        new_pos = [c.detach().clone() for c in pos_centers]
        log_msg = f"Reclustered anchors (center_method={center_method}): "
        log_msg += f"n POS: {len(new_pos)} POS: std={pos_std:.4f}, max_shift={max_shift_pos:.4f}"

        # --- NEGATIVE (optional) ---
        if n_neg > 0 and (k_neg is not None):
            new_neg: list[torch.Tensor] = []
            neg_feats = feats_proc[neg_mask]
            if k_neg < 0:
                k_neg = len(self.state["neg_anchors"]) or min(4, max(1, n_neg))
            k_neg = max(1, min(k_neg, neg_feats.shape[0]))

            neg_mean = neg_feats.mean(dim=0)
            neg_var = ((neg_feats - neg_mean) ** 2).sum(dim=1).mean()
            neg_std = float(torch.sqrt(neg_var.clamp_min(0.0)))

            init_neg = None
            if len(self.state["neg_anchors"]) > 0:
                init_list = []
                for a in self.state["neg_anchors"][:k_neg]:
                    vec = a.clone()
                    if trunc_dim is not None and int(trunc_dim) < vec.numel():
                        D = int(trunc_dim)
                        v2 = torch.zeros_like(vec)
                        v2[:D] = vec[:D]
                        vec = v2
                    init_list.append(F.normalize(vec, dim=-1, eps=1e-6))
                if len(init_list) < k_neg:
                    need = k_neg - len(init_list)
                    rand_idx = torch.randperm(neg_feats.shape[0])[:need]
                    init_list.extend([neg_feats[i].clone() for i in rand_idx])
                init_neg = torch.stack(init_list, dim=0)
            max_shift_neg = shift_factor_neg * neg_std if shift_factor_neg > 0 else 0.0

            try:
                neg_centers = SegmentationModule3D.cosine_kmeans(
                    neg_feats,
                    k_neg,
                    seed=seed,
                    reinit_empty=reinit_empty,
                    center_method=center_method,
                    init_centers=init_neg,
                    max_shift=max_shift_neg,
                )
                new_neg = [c.detach().clone() for c in neg_centers]
            except Exception as e:
                self._log(f"recluster_anchors: NEG clustering failed ({e}); skipping.")
                new_neg = []
            log_msg += f", n NEG: {len(new_neg)} NEG: std={neg_std:.4f}, max_shift={max_shift_neg:.4f}"
        else:
            # keep old negatives
            new_neg = self.state["neg_anchors"]

        # Install
        if apply:
            self.state["pos_anchors"] = new_pos
            self.state["neg_anchors"] = new_neg
            self.state["last_anchor_pos"] = [True] * len(new_pos) + [False] * len(
                new_neg
            )

        self._log(log_msg)

        # stack as tensors for return
        C = feat_all.shape[1]
        new_pos = (
            torch.stack(new_pos, dim=0)
            if len(new_pos) > 0
            else torch.empty((0, C), dtype=feat_all.dtype, device=feat_all.device)
        )
        new_neg = (
            torch.stack(new_neg, dim=0)
            if len(new_neg) > 0
            else torch.empty((0, C), dtype=feat_all.dtype, device=feat_all.device)
        )

        return new_pos, new_neg

    # --------------------------------------------------------------------- #
    # Internals: splats, math, and small helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _rect_to_bounds(H: int, W: int, rect0, rect1) -> tuple[int, int, int, int]:
        """
        Convert normalized [0,1] rectangle to integer pixel bounds [y0:y1, x0:x1) (y1/x1 exclusive).
        Guarantees at least 1 pixel in each dimension.
        """
        (x0, y0), (x1, y1) = rect0, rect1
        # clamp + sort
        xa, xb = float(np.clip(min(x0, x1), 0, 1)), float(np.clip(max(x0, x1), 0, 1))
        ya, yb = float(np.clip(min(y0, y1), 0, 1)), float(np.clip(max(y0, y1), 0, 1))
        # map to pixels (inclusive on the right/low edge via floor, then make exclusive with +1)
        xi0 = int(np.floor(xa * (W - 1)))
        xi1 = int(np.floor(xb * (W - 1)))
        yi0 = int(np.floor(ya * (H - 1)))
        yi1 = int(np.floor(yb * (H - 1)))
        # ensure non-empty
        xi1 = max(xi1, xi0)
        yi1 = max(yi1, yi0)
        return yi0, yi1 + 1, xi0, xi1 + 1  # y0, y1_excl, x0, x1_excl

    @staticmethod
    def _pool_vectors(
        v: torch.Tensor, method: str = "geom_median", iters: int = 20, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Pool a set of feature vectors v ∈ R^{M×C} into a single anchor vector.
        Returns an L2-normalized (C,) tensor.

        method:
        - "mean": arithmetic mean (current default)
        - "median": coordinate-wise median (robust; cheap)
        - "geom_median": geometric median via Weiszfeld iterations (most robust; pricier)

        iters: number of Weiszfeld iterations for geom_median
        eps  : small epsilon to avoid division by zero
        """
        assert v.dim() == 2, "v should be (M,C)"
        M, C = v.shape
        if M == 0:
            # Fallback guard: return small jitter to avoid NaN
            return F.normalize(
                torch.zeros(C, dtype=v.dtype, device=v.device) + 1e-12, dim=-1, eps=1e-6
            )

        if method == "mean":
            a = v.mean(dim=0)

        elif method == "median":
            # Coordinate-wise median; good robustness, very fast
            a = v.median(dim=0).values

        elif method == "geom_median":
            # Weiszfeld's algorithm
            # init with mean (or coord-median also works)
            a = v.mean(dim=0)
            for _ in range(iters):
                # distances to current estimate
                d = torch.norm(v - a[None, :], dim=1).clamp_min(eps)  # (M,)
                w = 1.0 / d  # weights
                a_new = (w[:, None] * v).sum(dim=0) / w.sum()  # weighted mean
                # early stop if converged
                if torch.norm(a_new - a) < 1e-6:
                    a = a_new
                    break
                a = a_new
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        return F.normalize(a + 1e-12, dim=-1, eps=1e-6)

    def _ensure_knn(self, K: int) -> None:
        """Cache kNN (including self as NN #0). Recompute only if we need a larger K."""
        means = self.splats["means"]  # (N, 3)
        N = means.shape[0]
        K = int(min(K, N))

        if self._knn_cache["idx"] is None or self._knn_cache["K"] < K:
            with torch.no_grad():
                xyz = means.to(dtype=torch.float32)

                # backend-agnostic knn
                d2, idx = compute_knn(xyz, K)

                self._knn_cache = {
                    "K": int(K),
                    "idx": idx,  # (N, K)
                    "d2": d2,  # (N, K)
                }

    def _get_knn(self, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (idx, d2) of shape (N,K). Guaranteed K rows available."""
        self._ensure_knn(K)
        idx = self._knn_cache["idx"][:, :K]
        d2 = self._knn_cache["d2"][:, :K]
        return idx, d2

    @torch.no_grad()
    def _get_point_features(self) -> torch.Tensor:
        return self.splats[self.feature_key]

    @staticmethod
    def _to_hwc(t):
        if isinstance(t, np.ndarray):
            x = torch.from_numpy(t)
        else:
            x = t
        if x.dim() == 4 and x.shape[0] == 1:
            x = x[0]  # remove batch dim
        if x.dim() != 3:
            raise ValueError("Expected 3D feature image (H,W,C) or (C,H,W).")
        # x = x.permute(1, 2, 0).contiguous()  # CHW->HWC
        return x

    @staticmethod
    def _truncate_and_unit_normalize(
        feat: torch.Tensor, trunc_dim: Optional[int]
    ) -> torch.Tensor:
        N, C = feat.shape
        if trunc_dim is None or int(trunc_dim) >= C:
            return F.normalize(feat, dim=-1, eps=1e-6)
        D = int(trunc_dim)
        out = feat.clone()
        out[..., D:] = 0.0
        return F.normalize(out, dim=-1, eps=1e-6)

    @staticmethod
    def _percentile_norm(
        x: torch.Tensor, pmin: float, pmax: float = 100.0
    ) -> torch.Tensor:
        # Normalize to [0,1] using percentile range (robust to outliers).
        if pmin > 0 and pmin <= pmax:
            xmin = float(torch.quantile(x, pmin / 100.0))
        else:
            xmin = float(x.min())
        if pmax >= pmin and pmax < 100:
            xmax = float(torch.quantile(x, pmax / 100.0))
        else:
            xmax = float(x.max())
        denom = max(xmax - xmin, 1e-8)
        return ((x - xmin) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _clamp_ratio(r: float, N: int, _dummyW: int) -> float:
        # For points, "image size" is N; reuse same safety clamp.
        r = float(r)
        r_min = 1.0 / max(1, N)
        return max(r_min, min(r, 0.99))

    def _log(self, msg: str) -> None:
        print(f"[Seg3D] {msg}")

    @staticmethod
    def cosine_kmeans(
        embeddings: torch.Tensor,
        k: int,
        *,
        iters: int = 50,
        init: str = "kmeans++",
        seed: Optional[int] = None,
        eps: float = 1e-6,
        reinit_empty: bool = True,
        return_assignments: bool = False,
        center_method: str = "geom_median",
        init_centers: Optional[torch.Tensor] = None,  # NEW: warm start centers (m<=k,C)
        max_shift: Optional[
            float
        ] = None,  # NEW: clamp center displacement from original init
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Spherical (cosine) k-means with robust center updates.

        init_centers:
            Tensor (m,C), m<=k. Provided rows are used as the first m initial centers
            (already assumed roughly unit norm; will be normalized). Remaining centers
            initialized via chosen 'init' strategy (kmeans++ / random) on the residual data.

        max_shift:
            If not None, limit each final center's displacement from its ORIGINAL
            initialization (per cluster) to <= max_shift (Euclidean in embedding space).
            Applied after each update (center clipped toward its original).
        """
        if center_method not in ("mean", "median", "geom_median"):
            raise ValueError(
                "center_method must be one of {'mean','median','geom_median'}"
            )
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be (N,C)")
        N, C = embeddings.shape
        if k <= 0:
            raise ValueError("k must be > 0")
        if N == 0:
            empty = embeddings.new_empty((0, C))
            return (
                (empty, embeddings.new_empty((0,), dtype=torch.long))
                if return_assignments
                else empty
            )
        if k >= N:
            uniq = F.normalize(embeddings, dim=-1, eps=eps)
            if k > N:
                pad = uniq[-1:].expand(k - N, -1)
                uniq = torch.cat([uniq, pad], dim=0)
            return (
                (uniq, torch.arange(uniq.shape[0], device=uniq.device))
                if return_assignments
                else uniq
            )

        dev = embeddings.device
        rng = torch.Generator(device=dev)
        if seed is not None:
            rng.manual_seed(int(seed))

        X = F.normalize(embeddings, dim=-1, eps=eps)  # (N,C)

        centers = torch.empty((k, C), device=dev, dtype=X.dtype)

        m = 0
        if init_centers is not None and init_centers.numel() > 0:
            if init_centers.shape[1] != C:
                raise ValueError("init_centers has wrong dimensionality")
            m = min(k, init_centers.shape[0])
            centers[:m] = F.normalize(init_centers[:m], dim=-1, eps=eps)

        if m < k:
            remain = k - m
            if init not in ("kmeans++", "random"):
                raise ValueError("init must be 'kmeans++' or 'random'")
            if init == "random":
                perm = torch.randperm(N, generator=rng, device=dev)[:remain]
                centers[m:] = X[perm]
            else:
                # kmeans++ for remaining slots
                # If some centers exist, compute current best_sim; else start fresh
                if m == 0:
                    idx0 = torch.randint(0, N, (1,), generator=rng, device=dev).item()
                    centers[0] = X[idx0]
                    m = 1
                best_sim = (X @ centers[:m].t()).max(dim=1).values  # (N,)
                for ci in range(m, k):
                    dist = (1.0 - best_sim.clamp(-1.0, 1.0)).clamp_min(0)
                    prob = dist / dist.sum().clamp_min(eps)
                    nxt = torch.multinomial(prob, 1, generator=rng).item()
                    centers[ci] = X[nxt]
                    sim_new = (X @ centers[ci].unsqueeze(1)).squeeze(1)
                    best_sim = torch.maximum(best_sim, sim_new)

        centers = F.normalize(centers, dim=-1, eps=eps)
        original_centers = centers.clone()  # for max_shift constraint
        assignments = torch.empty(N, dtype=torch.long, device=dev)
        prev_centers = centers.clone()
        tol = 1e-4

        for _it in range(iters):
            sims = X @ centers.t()  # (N,k)
            assignments = sims.argmax(dim=1)

            for ci in range(k):
                mask = assignments == ci
                if not mask.any():
                    if reinit_empty:
                        with torch.no_grad():
                            dist_like = 1.0 - sims.max(dim=1).values
                            far_idx = torch.argmax(dist_like)
                        centers[ci] = X[far_idx]
                    continue
                cluster_pts = X[mask]
                new_center = SegmentationModule3D._pool_vectors(
                    cluster_pts,
                    method="mean" if center_method == "mean" else center_method,
                )
                # Enforce max_shift (relative to original center)
                if max_shift is not None and max_shift >= 0:
                    orig = original_centers[ci]
                    disp = new_center - orig
                    d = torch.norm(disp).item()
                    if d > max_shift and max_shift > 0:
                        new_center = F.normalize(
                            orig + disp * (max_shift / d), dim=-1, eps=eps
                        )
                    elif max_shift == 0:
                        new_center = orig
                centers[ci] = new_center

            shift = torch.norm(centers - prev_centers, dim=1).max().item()
            prev_centers.copy_(centers)
            if shift < tol:
                break

        if return_assignments:
            return centers, assignments
        return centers
