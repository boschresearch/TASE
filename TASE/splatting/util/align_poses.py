# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Interactive pose alignment for cross-baseline evaluation.

The user aligns camera coordinate frames between the TASE (Ours) scene and a
baseline scene by navigating the baseline to match static reference images
rendered from the Ours scene.

Workflow (2D overlay matching):
  1. A reference image is rendered from the Ours scene at one of the test poses.
     It appears as a 2D overlay on the live baseline render and as a thumbnail.
  2. The user navigates the **baseline scene** until it matches the reference.
  3. "Confirm Match" records the baseline camera pose.
  4. Repeat for 3+ test poses, then solve and save.

The overlay is purely 2D — the reference is a static snapshot; only the
baseline scene renders live.

Usage::

    python -m TASE.splatting.util.align_poses \\
        --scene bear --method DreamCatalyst [--port 9000]

See ``TASE/splatting/util/ALIGNMENT_GUIDE.md`` for full instructions.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import viser

from TASE.splatting.common.rendering import Scene
from TASE.splatting.viewer.viewer import Viewer, ColorMapper, CameraState
from TASE.splatting.util.data_util import pose_to_rotation_translation

from TASE.splatting.util.evaluation import (
    POSE_DIR,
    IMG_WH,
    SCENE_NAME_MAP_INV,
    _load_poses,
    _load_scene_for_method,
    _find_dc_base,
    _find_ge_base,
    _find_es_base,
    _find_ours_base,
)

logger = logging.getLogger(__name__)

# Thumbnail size displayed in the sidebar image widget
_THUMB_WH = (320, 180)


# ---------------------------------------------------------------------------
# Similarity transform solver
# ---------------------------------------------------------------------------

def solve_similarity_from_poses(
    poses_a: List[np.ndarray],
    poses_b: List[np.ndarray],
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Solve similarity + optional flip from matched c2w pose pairs.

    Given anchor pairs ``(c2w_a[i], c2w_b[i])``, solves for ``(R, s, t, F)``
    such that::

        c2w_b[:3,:3] ≈ R @ c2w_a[:3,:3] @ F
        c2w_b[:3, 3] ≈ s * (R @ c2w_a[:3,3]) + t

    Args:
        poses_a: Camera poses in frame A (e.g. TASE / Ours).
        poses_b: Matching camera poses in frame B (e.g. baseline).

    Returns:
        ``(R, s, t, F)`` — rotation, scale, translation, camera-axis flip.
    """
    pos_a = np.array([p[:3, 3] for p in poses_a])
    pos_b = np.array([p[:3, 3] for p in poses_b])

    if len(pos_a) < 3:
        raise ValueError(f"Need >= 3 anchor pairs, got {len(pos_a)}")

    # Procrustes / Umeyama on positions
    mu_a = pos_a.mean(0)
    mu_b = pos_b.mean(0)
    ca = pos_a - mu_a
    cb = pos_b - mu_b
    H = ca.T @ cb
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    rotated = (R @ ca.T).T
    s = np.sum(cb * rotated) / np.sum(rotated * rotated)
    t = mu_b - s * (R @ mu_a)

    # Detect camera-axis flip
    FLIP = np.diag([1.0, -1.0, -1.0])
    err_no_flip = sum(
        np.linalg.norm(R @ a[:3, :3] - b[:3, :3], "fro")
        for a, b in zip(poses_a, poses_b)
    )
    err_flip = sum(
        np.linalg.norm(R @ a[:3, :3] @ FLIP - b[:3, :3], "fro")
        for a, b in zip(poses_a, poses_b)
    )
    F = FLIP if err_flip < err_no_flip else np.eye(3)

    return R, s, t, F


def apply_similarity_to_c2w(
    c2w: np.ndarray,
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Transform a c2w pose from frame A to frame B."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3].astype(np.float64) @ F
    out[:3, 3] = s * (R @ c2w[:3, 3].astype(np.float64)) + t
    return out


def check_anchor_spacing(
    anchor_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[bool, str]:
    """Verify anchor positions are well-distributed (not collinear).

    Returns ``(is_ok, message)``.
    """
    if len(anchor_pairs) < 3:
        return False, f"Need >= 3 anchors, have {len(anchor_pairs)}"
    positions = np.array([a[:3, 3] for a, _ in anchor_pairs])
    centered = positions - positions.mean(0)
    _, S, _ = np.linalg.svd(centered)
    ratio = S[-1] / max(S[0], 1e-12)
    if ratio < 0.01:
        return False, (
            f"Anchors are nearly collinear (spread ratio {ratio:.4f}). "
            "Add viewpoints from more diverse angles."
        )
    return True, f"Anchor spread ratio: {ratio:.3f} (good)"


# ---------------------------------------------------------------------------
# Aligned pose I/O
# ---------------------------------------------------------------------------

def save_aligned_poses(
    scene: str,
    method: str,
    test_c2ws: List[np.ndarray],
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    F: np.ndarray,
    anchor_pairs: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Optional[Path] = None,
) -> Path:
    """Save aligned test poses and the similarity transform.

    File: ``data/poses/{scene}/aligned_{method}.npz``
    """
    if out_dir is None:
        out_dir = POSE_DIR / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"aligned_{method}.npz"
    np.savez(
        path,
        test_c2ws=np.array(test_c2ws),
        R=R,
        s=np.array(s),
        t=t,
        F=F,
        anchors_ours=np.array([a for a, _ in anchor_pairs]),
        anchors_baseline=np.array([b for _, b in anchor_pairs]),
        scene=scene,
        method=method,
    )
    logger.info("Saved aligned poses to %s", path)
    return path


def load_aligned_poses(scene: str, method: str) -> Optional[Dict]:
    """Load aligned poses if available.

    Returns dict with keys ``test_c2ws``, ``R``, ``s``, ``t``, ``F``,
    or ``None`` if no alignment file exists.
    """
    path = POSE_DIR / scene / f"aligned_{method}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "test_c2ws": list(data["test_c2ws"]),
        "R": data["R"],
        "s": float(data["s"]),
        "t": data["t"],
        "F": data["F"],
    }


# ---------------------------------------------------------------------------
# Reference image rendering
# ---------------------------------------------------------------------------


@torch.no_grad()
def _render_reference(
    ours_scene: Scene,
    c2w: np.ndarray,
    img_wh: Tuple[int, int] = IMG_WH,
) -> np.ndarray:
    """Render the Ours scene at ``c2w``.  Returns float32 (H, W, 3) in [0, 1]."""
    W, H = img_wh
    cam = CameraState(c2w=c2w, aspect=W / H, fov=0.7)
    img = ours_scene.render_fn(cam, img_wh)
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Alignment state
# ---------------------------------------------------------------------------

@dataclass
class AlignmentState:
    """Mutable state shared between the render function and GUI callbacks."""

    scene_name: str
    method_name: str
    # All test poses in the Ours / TASE coordinate frame
    test_c2ws: List[np.ndarray]
    # Index into test_c2ws for the currently active reference
    current_idx: int = 0
    # Alpha for 2D overlay (0 = only baseline, 1 = only reference)
    alpha: float = 0.5
    # Confirmed (c2w_ours, c2w_baseline) pairs
    anchor_pairs: List = field(default_factory=list)
    # Solved transform
    solved: bool = False
    R: Optional[np.ndarray] = None
    s: Optional[float] = None
    t: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    # Internal: rendered references at IMG_WH, keyed by index — populated lazily
    _render_cache: Dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def set_reference(self, idx: int, img: np.ndarray) -> None:
        """Store a rendered reference for *idx*."""
        with self._lock:
            self._render_cache[idx] = img

    def get_reference_for_display(
        self, img_wh: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Return reference resized to *img_wh*, or None if not yet rendered."""
        ref = self._render_cache.get(self.current_idx)
        if ref is None:
            return None
        W, H = img_wh
        if ref.shape[:2] == (H, W):
            return ref
        return cv2.resize(ref, (W, H)).astype(np.float32)

    def get_thumb(self) -> Optional[np.ndarray]:
        """Return uint8 (H, W, 3) thumbnail for the sidebar."""
        ref = self._render_cache.get(self.current_idx)
        if ref is None:
            return None
        W, H = _THUMB_WH
        return np.clip(cv2.resize(ref, (W, H)) * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def _status_text(state: AlignmentState, extra: str = "") -> str:
    n = len(state.anchor_pairs)
    n_total = len(state.test_c2ws)
    text = (
        f"**Pose {state.current_idx + 1} / {n_total}** "
        f"\u2014 **Anchors: {n} / 3+**"
    )
    if state.solved:
        text += f"  \u2713 Solved (s={state.s:.4f})"
    if extra:
        text += f"\n\n{extra}"
    return text


def _add_alignment_gui(
    viewer: Viewer,
    state: AlignmentState,
    ours_scene: Scene,
) -> None:
    """Attach the 2D-overlay alignment GUI to *viewer*."""

    def _ensure_reference(idx: int) -> None:
        """Render and cache the reference for *idx* if not already done."""
        if idx not in state._render_cache:
            logger.info(
                "Rendering reference %d/%d...", idx + 1, len(state.test_c2ws)
            )
            img = _render_reference(ours_scene, state.test_c2ws[idx])
            state.set_reference(idx, img)

    # Pre-render the first reference so the overlay is ready immediately
    _ensure_reference(state.current_idx)
    initial_thumb = state.get_thumb()

    with viewer.server.gui.add_folder("Pose Alignment", expand_by_default=True):

        # --- Thumbnail of the current Ours reference view ---
        thumb_handle = None
        if initial_thumb is not None:
            try:
                thumb_handle = viewer.server.gui.add_image(initial_thumb)
            except Exception as e:
                logger.warning("add_image widget unavailable: %s", e)

        status_md = viewer.server.gui.add_markdown(_status_text(state))

        def _refresh_status(extra: str = "") -> None:
            status_md.content = _status_text(state, extra)

        def _refresh_thumb() -> None:
            if thumb_handle is None:
                return
            thumb = state.get_thumb()
            if thumb is None:
                return
            try:
                thumb_handle.image = thumb
            except AttributeError:
                try:
                    thumb_handle.content = thumb
                except Exception:
                    pass

        viewer.server.gui.add_markdown("---")

        # --- Test pose navigation ---
        viewer.server.gui.add_markdown("**Select Reference Pose:**")
        nav_group = viewer.server.gui.add_button_group(
            "", ["\u25c0 Prev", "\u25b6 Next"]
        )
        nav_group.value = ""

        @nav_group.on_click
        def _nav(event):
            n = len(state.test_c2ws)
            if event.target.value == "\u25c0 Prev":
                state.current_idx = (state.current_idx - 1) % n
            elif event.target.value == "\u25b6 Next":
                state.current_idx = (state.current_idx + 1) % n
            event.target.value = ""
            _ensure_reference(state.current_idx)
            _refresh_thumb()
            _refresh_status()
            viewer.rerender(None)

        # --- Overlay alpha ---
        alpha_slider = viewer.server.gui.add_slider(
            "Overlay alpha", min=0.0, max=1.0, step=0.05,
            initial_value=state.alpha,
        )

        @alpha_slider.on_update
        def _alpha(event):
            state.alpha = event.target.value
            viewer.rerender(None)

        viewer.server.gui.add_markdown("---")

        # --- Confirm match ---
        confirm_btn = viewer.server.gui.add_button("\u2713 Confirm Match")

        @confirm_btn.on_click
        def _confirm(event):
            c2w_base = viewer.get_camera_state(event.client).c2w.copy()
            c2w_ours = state.test_c2ws[state.current_idx].copy()
            state.anchor_pairs.append((c2w_ours, c2w_base))
            n = len(state.anchor_pairs)
            viewer._log(
                f"Anchor {n} confirmed (test pose {state.current_idx + 1})."
            )

            # Mark anchor in the 3D scene
            q_b, t_b = pose_to_rotation_translation(c2w_base)
            viewer.server.scene.add_camera_frustum(
                name=f"alignment/anchor_{n}",
                fov=0.7, aspect=1.5, position=t_b, wxyz=q_b,
                scale=0.15, color=np.array([1.0, 0.3, 0.0]),
            )

            # Auto-advance to a well-spaced test pose
            n_total = len(state.test_c2ws)
            state.current_idx = (
                state.current_idx + max(1, n_total // 3)
            ) % n_total
            _ensure_reference(state.current_idx)
            _refresh_thumb()
            _refresh_status()
            viewer.rerender(None)

        undo_btn = viewer.server.gui.add_button("\u21a9 Undo Last")

        @undo_btn.on_click
        def _undo(event):
            if not state.anchor_pairs:
                return
            n = len(state.anchor_pairs)
            state.anchor_pairs.pop()
            try:
                viewer.server.scene.remove_by_name(f"alignment/anchor_{n}")
            except Exception:
                pass
            _refresh_status()
            viewer._log(f"{len(state.anchor_pairs)} anchors remaining.")

        viewer.server.gui.add_markdown("---")

        # --- Solve ---
        solve_btn = viewer.server.gui.add_button("\u2699 Solve Transform")

        @solve_btn.on_click
        def _solve(event):
            is_ok, msg = check_anchor_spacing(state.anchor_pairs)
            if not is_ok:
                viewer._log(msg)
                _refresh_status(msg)
                return
            poses_a = [a for a, _ in state.anchor_pairs]
            poses_b = [b for _, b in state.anchor_pairs]
            R, s, t_vec, F = solve_similarity_from_poses(poses_a, poses_b)
            state.R, state.s, state.t, state.F = R, s, t_vec, F
            state.solved = True
            errs = [
                np.linalg.norm(
                    apply_similarity_to_c2w(pa, R, s, t_vec, F)[:3, 3] - pb[:3, 3]
                )
                for pa, pb in state.anchor_pairs
            ]
            has_flip = np.allclose(F, np.diag([1, -1, -1]))
            result_msg = (
                f"s={s:.4f}, flip={has_flip}\n"
                f"pos err: mean={np.mean(errs):.6f}, max={np.max(errs):.6f}"
            )
            viewer._log(f"Solved: {result_msg}")
            _refresh_status(result_msg)

        # --- Save ---
        save_btn = viewer.server.gui.add_button("\U0001f4be Save Aligned Poses")

        @save_btn.on_click
        def _save(event):
            if not state.solved:
                viewer._log("Solve the transform first!")
                _refresh_status("Solve first.")
                return
            aligned = [
                apply_similarity_to_c2w(c2w, state.R, state.s, state.t, state.F)
                for c2w in state.test_c2ws
            ]
            path = save_aligned_poses(
                state.scene_name, state.method_name,
                aligned, state.R, state.s, state.t, state.F,
                state.anchor_pairs,
            )
            viewer._log(f"Saved to {path}")
            _refresh_status(f"\u2713 Saved \u2192 `{path.name}`")


# ---------------------------------------------------------------------------
# Auto-discover base checkpoints
# ---------------------------------------------------------------------------

def _discover_base_ckpts(
    scene: str,
    method: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-discover Ours and baseline base-model checkpoints.

    Returns ``(ours_ckpt_path, baseline_ckpt_path)``.
    """
    ours_ckpt = _find_ours_base(scene)
    scene_ext = SCENE_NAME_MAP_INV.get(scene, scene)

    if method == "DreamCatalyst":
        dc = _find_dc_base(scene_ext)
        base_ckpt = dc[0] if dc else None
    elif method in ("GaussianEditor", "DGE"):
        base_ckpt = _find_ge_base(scene)
    elif method == "EditSplat":
        base_ckpt = _find_es_base(scene_ext)
    else:
        base_ckpt = None

    return ours_ckpt, base_ckpt



# ---------------------------------------------------------------------------
# Alignment viewer launcher
# ---------------------------------------------------------------------------

def launch_alignment_viewer(
    ours_ckpt: Path,
    baseline_ckpt: Path,
    scene_name: str,
    method_name: str,
    port: int = 9000,
    renormalize_ours: bool = False,
    renormalize_baseline: bool = False,
) -> None:
    """Launch the interactive 2D-overlay alignment viewer.

    The BASELINE scene is the live 3D scene in the viewport.  Reference images
    are rendered from the Ours scene on-demand and overlaid on the live render
    as a 2D composite.

    Args:
        ours_ckpt: Path to the unedited Ours checkpoint.
        baseline_ckpt: Path to the unedited baseline checkpoint.
        scene_name: Scene identifier (``"bear"``, ``"bike"``, ``"face"``).
        method_name: Baseline method name (e.g. ``"DreamCatalyst"``).
        port: Viser server port.
        renormalize_ours: Apply PCA renormalization to Ours scene.
        renormalize_baseline: Apply PCA renormalization to baseline scene.
    """
    logger.info("Loading Ours scene from %s", ours_ckpt)
    ours_scene = _load_scene_for_method(
        "Ours", ours_ckpt, renormalize=renormalize_ours,
    )
    if ours_scene is None:
        raise RuntimeError(f"Failed to load Ours scene from {ours_ckpt}")

    logger.info("Loading baseline scene from %s", baseline_ckpt)
    baseline_scene = _load_scene_for_method(
        method_name, baseline_ckpt, renormalize=renormalize_baseline,
    )
    if baseline_scene is None:
        raise RuntimeError(
            f"Failed to load baseline scene from {baseline_ckpt}"
        )

    test_c2ws = _load_poses(scene_name)
    logger.info("Loaded %d test poses for scene '%s'", len(test_c2ws), scene_name)

    server = viser.ViserServer(port=port, verbose=False)

    # The BASELINE is the live 3D scene; Ours is used only for reference renders
    viewer = Viewer(
        server=server,
        scene=baseline_scene,
        cfg=getattr(baseline_scene, "cfg", None),
        mode="rendering",
        colormapper=getattr(baseline_scene, "depth_colormapper", ColorMapper()),
    )

    state = AlignmentState(
        scene_name=scene_name,
        method_name=method_name,
        test_c2ws=test_c2ws,
    )

    # Override render to composite the static Ours reference over live baseline
    _baseline_render = viewer.base_render_fn

    def _composite_render(camera_state, img_wh, **kwargs):
        img_live = _baseline_render(camera_state, img_wh, **kwargs)
        ref = state.get_reference_for_display(img_wh)
        if ref is None or state.alpha <= 0.0:
            return img_live
        blended = (
            (1.0 - state.alpha) * np.asarray(img_live)
            + state.alpha * ref
        )
        return np.clip(blended, 0.0, 1.0).astype(np.float32)

    viewer.base_render_fn = _composite_render

    # Attach GUI (also pre-renders the first reference)
    _add_alignment_gui(viewer, state, ours_scene)

    print(f"\n{'=' * 60}")
    print(f"  Alignment viewer: {scene_name} / {method_name}")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'=' * 60}")
    print(
        "  Live render : BASELINE scene\n"
        "  Overlay     : Ours reference image (use 'Overlay alpha')\n"
        "  Navigate baseline to match, then click 'Confirm Match'.\n"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive pose alignment between Ours and a baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        required=True,
        choices=["bear", "bike", "face"],
        help="Scene name",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Baseline method (e.g. DreamCatalyst, GaussianEditor, DGE, EditSplat)",
    )
    parser.add_argument(
        "--ours-ckpt",
        type=str,
        default=None,
        help="Explicit Ours checkpoint path (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=None,
        help="Explicit baseline checkpoint path (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Viser server port",
    )
    parser.add_argument(
        "--renormalize-ours",
        action="store_true",
        help="Apply PCA renormalization to Ours scene",
    )
    parser.add_argument(
        "--renormalize-baseline",
        action="store_true",
        help="Apply PCA renormalization to baseline scene (use for bear GE/DGE/ES)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Resolve checkpoints
    if args.ours_ckpt:
        ours_ckpt = Path(args.ours_ckpt)
    else:
        ours_ckpt = _find_ours_base(args.scene)
        if ours_ckpt is None:
            raise FileNotFoundError(
                f"No Ours base model found for scene '{args.scene}'. "
                "Pass --ours-ckpt explicitly."
            )
        logger.info("Auto-discovered Ours base: %s", ours_ckpt)

    if args.baseline_ckpt:
        baseline_ckpt = Path(args.baseline_ckpt)
    else:
        _, baseline_ckpt = _discover_base_ckpts(args.scene, args.method)
        if baseline_ckpt is None:
            raise FileNotFoundError(
                f"No {args.method} base model found for scene '{args.scene}'. "
                "Pass --baseline-ckpt explicitly."
            )
        logger.info("Auto-discovered baseline: %s", baseline_ckpt)

    launch_alignment_viewer(
        ours_ckpt=ours_ckpt,
        baseline_ckpt=baseline_ckpt,
        scene_name=args.scene,
        method_name=args.method,
        port=args.port,
        renormalize_ours=args.renormalize_ours,
        renormalize_baseline=args.renormalize_baseline,
    )


if __name__ == "__main__":
    main()
