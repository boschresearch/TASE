# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np
import viser

# Core project imports
from TASE.splatting.common.rendering import Scene
from TASE.splatting.viewer.viewer import Viewer, ColorMapper
from TASE.splatting.trainer.trainer import Runner, Config

# Import your new module skeleton + generator (DI to avoid circulars elsewhere)
from TASE.splatting.common.editing import SceneEditingModule
from TASE.splatting.common.rendering import _load_ckpt_and_dataset
from TASE.splatting.common.generative_helper import SampleGenerator
from TASE.splatting.util.data_util import resolve_path

# --------------------------- Helpers ---------------------------


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive Scene Editing Viewer (Global & Local regeneration with 3D selection)"
    )
    # Required
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to pretrained scene checkpoint (.pt)",
    )
    # Optional dataset root override (where new samples are written)
    parser.add_argument(
        "--new-dataset",
        type=str,
        default=None,
        help="Override dataset root for saving new samples",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="edit_session",
        help="Name for this editing session, used for output directory",
    )
    parser.add_argument(
        "--old-dataset",
        type=str,
        default=None,
        help="Override dataset root to load the original training samples, visualization only.",
    )
    parser.add_argument(
        "--reconstruction", 
        type=str,
        default=None,
        help="Path to the original reconstruction directory."
    )
    # Viewer
    parser.add_argument("--port", type=int, default=8080, help="Viewer TCP port")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    # Config for the splat optimization. Some values will be overridden by the editing module.
    parser.add_argument(
        "--runner-config",
        type=str,
        default="tase_edit_local.yml",
        help="YAML config file for the Runner",
    )
    # Model Checkpoints
    parser.add_argument(
        "--mrl-model-path",
        type=str,
        default=None,
        help="Path to MRL model checkpoint (relative to $TASE_ROOT/WEIGHTS or absolute)",
    )
    parser.add_argument(
        "--pca-path",
        type=str,
        default=None,
        help="Path to PCA model checkpoint (relative to $TASE_ROOT or absolute)",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="If set, uses the PCA model for dimensionality reduction.",
    )
    parser.add_argument(
        "--flip-z",
        action="store_true",
        help="If set, flips the Z-axis during scene loading.",
    )
    parser.add_argument(
        "--ctrl-net-path",
        type=str,
        default=None,
        help="Path to ControlNet model checkpoint (relative to $TASE_ROOT/WEIGHTS or absolute)",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Path to Transformer model checkpoint (if separate finetuned from the original Diffusion model)",
    )
    parser.add_argument(
        "--raw-dino",
        action="store_true",
        help="Use raw DINOv3 features (768 dims) without adapter. Sets max-ctrl-channels to 768.",
    )
    parser.add_argument(
        "--max-ctrl-channels",
        type=int,
        default=64,
        help="Max control channels for ControlNet (default 64, set to 768 for raw DINO).",
    )
    # Preset file(s)
    parser.add_argument(
        "--preset",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to preset file (JSON). If auto-edit is set, these presets will be applied automatically.",
    )
    # Run edit(s) directly
    parser.add_argument(
        "--auto-edit",
        action="store_true",
        help="If set, automatically runs the auto edit. Please provide a preset file for non default edits. Presets for local edits need to contain a path to an anchor point file.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="If set, keeps the viewer alive after auto edit is done.",
    )
    args = parser.parse_args()
    device = _device_from_arg(args.device)

    # ---- Load scene from checkpoint (learnable)
    if args.ckpt is not None:
        ckpt_path = Path(args.reconstruction) / args.ckpt if args.reconstruction is not None else Path(args.ckpt)
        print(f"[Edit] Loading scene from: {ckpt_path}")
        scene, TASE_parser, data_dir, samples = _load_ckpt_and_dataset(
            ckpt_path, args.old_dataset, device=device
        )  # verify checkpoint + dataset
    else:
        scene, TASE_parser, data_dir, samples = None, None, None, None

    # Configs path for relative paths
    cfg_dir = Path(__file__).parent / "configs"

    # ---- Optional Runner (for post-edit optimization)
    # Load config from YAML
    cfg = Path(args.runner_config)
    if not cfg.is_file():
        cfg = cfg_dir / cfg
    if not cfg.is_file():
        raise FileNotFoundError(f"Runner config file not found: {cfg}")

    # ---- Viewer server + viewer
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = Viewer(
        server=server,
        scene=scene,
        cfg=scene.cfg if scene is not None else None,
        mode="rendering",
        initial_pose=None,  # or pass a specific pose if desired
        colormapper=(
            scene.depth_colormapper
            if hasattr(scene, "depth_colormapper")
            else ColorMapper()
        ),
        samples=samples,  # we’ll add/update during generation
        segmentation_module=True,  # let your Viewer mount segmentation if desired
    )

    # ---- Build SampleGenerator (DI)
    max_ctrl_channels = 768 if args.raw_dino else args.max_ctrl_channels
    sample_gen = SampleGenerator(
        scene=scene,
        parser=TASE_parser,
        load_models=True,
        ctrl_net_path=args.ctrl_net_path,
        mrl_model_path=args.mrl_model_path,
        pca_path=args.pca_path if args.use_pca else None,
        use_pca=args.use_pca,
        raw_dino=args.raw_dino,
        transformer_path=args.transformer_path,
        max_ctrl_channels=max_ctrl_channels,
        device=str(device),
        dtype=torch.bfloat16,
    )

    # ---- Construct & attach the SceneEditingModule
    if args.new_dataset is None:
        new_data_dir = parser.datadir + "_EDIT"
    else:
        new_data_dir = args.new_dataset
    new_data_dir = resolve_path(new_data_dir)
    print(f"[Edit] Dataset root for new samples: {new_data_dir}")
    edit_module = SceneEditingModule(
        preset=None,  # Load the default preset first, only relevant parameters will be overridden later
        server=server,
        viewer=viewer,
        scene=scene,
        parser=TASE_parser,
        runner_config=cfg,
        sample_gen=sample_gen,
        old_data_dir=args.old_dataset,
        new_data_dir=new_data_dir,
        reconstruction_dir=args.reconstruction,
        name=args.name,
        flip_z=args.flip_z,
        device=str(device),
        dtype=torch.bfloat16,
    )
    edit_module.attach()

    print(f"[Edit] Viewer running on port {args.port}. Ctrl+C to exit.")
    try:
        if args.auto_edit:
            for preset in args.preset:
                print(f"[Edit] Running automatic edit with preset: {preset} …")
                try:
                    edit_module._run_action(
                        "Auto Edit",
                        lambda p=preset: edit_module.load_scene_or_preset(p),
                    )
                    print(f"[Edit] Auto edit completed for preset {preset}.")
                except Exception as ex:
                    print(f"[Edit] Auto edit failed for preset {preset}: {ex}")
        if not args.auto_edit or args.keep_alive:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful shutdown
        try:
            if hasattr(viewer, "server") and viewer.server is not None:
                viewer.server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
