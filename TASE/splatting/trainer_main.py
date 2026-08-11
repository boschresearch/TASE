# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from gsplat V 1.5.3
#   (https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py)
# Copyright 2025 Nerfstudio Team, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import argparse
import datetime
import os
import shutil
from pathlib import Path

import torch
import debugpy

from TASE.splatting.trainer.trainer import Runner, Config


def _detect_cluster() -> bool:
    try:
        name = torch.cuda.get_device_properties(0).name
        return name not in {"NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"}
    except Exception:
        return False


def _resolve_data_path(p: Path, data_root_env: str | None) -> Path:
    """
    If 'p' is absolute, return as-is.
    If 'p' is relative, join it under $DATA_ROOT.
    """
    if p.is_absolute():
        return p
    if not data_root_env:
        raise ValueError(
            f"Relative data dir '{p}' provided but $DATA_ROOT is not set. "
            "Set DATA_ROOT or pass an absolute path."
        )
    return Path(os.path.expandvars(data_root_env)).expanduser() / p


def _build_result_template(cfg_result_dir: str, result_root_env: Path | None) -> str:
    """
    Always build result dirs under result_root_env if provided.
    'cfg_result_dir' should be a (possibly relative) template with exactly two '{}' placeholders.
    """
    if cfg_result_dir.count("{}") != 2:
        raise ValueError(
            "result_dir in config must have exactly two '{}' placeholders (timestamp, scene)."
        )

    # Normalize: strip leading slashes so join doesn't discard result_root_env
    rel = cfg_result_dir.lstrip("/")

    if result_root_env is not None:
        return str(result_root_env / rel)  # still contains '{}' placeholders
    else:
        # Fall back to the template from YAML, but warn so users know it's not under TASE_ROOT
        print("[WARN] $TASE_ROOT not set. Using result_dir from YAML as-is.")
        return cfg_result_dir


def main():
    parser = argparse.ArgumentParser(description="Train 3DGS scenes with gsplat")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="mipNeRF360.yml",
        help="YAML config file path",
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset root directories (absolute or RELATIVE to $DATA_ROOT)",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        required=True,
        help="Scene identifiers (e.g., bicycle.json)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )
    parser.add_argument("--clean", action="store_true", help="Remove previous results")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Override name (e.g. "debug" or "2025-09-16_12:00:00")',
    )
    parser.add_argument("--disable-viewer", action="store_true", help="Disable viewer")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    cluster = _detect_cluster()
    debug = debugpy.is_client_connected()

    # Load config from YAML
    cfg = Path(args.config)
    if not cfg.is_file():
        cfg = Path(__file__).parent / "configs" / args.config
    cfg: Config = Config.from_yaml(cfg)

    if args.disable_viewer:
        cfg.disable_viewer = True

    print(
        f"Using device: {args.device}, "
        f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Mode: {'cluster' if cluster else 'local'}, debug: {debug}")

    # Build result template under $TASE_ROOT
    tase_root_str = os.environ.get("TASE_ROOT")
    if tase_root_str is None:
        print("[WARN] $TASE_ROOT is not set. Output paths will be relative to cwd.")
        result_root_env = None
        data_root_env = None
    else:
        result_root_env = Path(tase_root_str) / "OUTPUT" / "reconstruction"
        data_root_env = Path(tase_root_str) / "DATA"
    result_dir_template = _build_result_template(cfg.result_dir, result_root_env)
    if result_root_env:
        print(f"TASE_ROOT: {result_root_env}")
    if data_root_env:
        print(f"DATA_ROOT: {data_root_env}")

    # Iterate over provided data roots and scenes
    for data_dir_arg in args.data_dirs:
        data_dir = _resolve_data_path(data_dir_arg, data_root_env)
        cfg.data_dir = str(data_dir)
        print(f"\n== Dataset path: {cfg.data_dir}")

        for scene in args.scenes:

            # Prepare timestamp
            if args.name is not None:
                name_str = args.name
            else:
                name_str = data_dir_arg.name

            cfg.scene = scene
            cfg.result_dir = result_dir_template.format(name_str, scene)

            print(f"Result path: {cfg.result_dir}")
            print(f"Max steps: {cfg.max_steps}")

            rd = Path(cfg.result_dir)
            if rd.exists():
                if args.clean:
                    print("Removing previous results...")
                    shutil.rmtree(rd)
                else:
                    print(
                        "Result dir already exists. Use --clean to remove previous results."
                    )
                    date_time_str = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H:%M:%S"
                    )
                    cfg.result_dir = str(rd.with_suffix(f".{date_time_str}"))
                    print(f"Using new result dir: {cfg.result_dir}")

            runner = Runner(cfg=cfg)
            runner.train()
            print(f"Training finished for scene: {scene}")


if __name__ == "__main__":
    main()

"""
cd TASE/splatting
python trainer_main.py \
  --config mipNeRF360.yml \
  --data-dirs TASE_360_FINAL_AE_GRAM_EQV_MOMENTS \
  --scenes bicycle garden bonsai \
  --clean
"""
