# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Prepare external baseline renders for TASE evaluation.

Converts renders from external baselines (DreamCatalyst, EditSplat, etc.)
into the directory structure expected by compute_directional_similarity():

    ablation_dir/
        prompts.json
        {scene}/
            {method}/
                original_scene/imgs/rgb_000.png, rgb_001.png, ...
                {experiment}/imgs/rgb_000.png, rgb_001.png, ...

Usage:
    python -m TASE.splatting.util.prepare_baseline_eval \
        --baseline DreamCatalyst \
        --renders-root Results/DreamCatalyst/renders \
        --ablation-dir OUTPUT/ablations
"""

import argparse
import logging
import shutil
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Scene name mapping: external baseline name → TASE ablation name
SCENE_NAME_MAP = {
    "bicycle": "bike",
}


def convert_dc_renders(
    renders_root: Path,
    ablation_dir: Path,
    method_name: str = "DreamCatalyst",
) -> None:
    """Convert DreamCatalyst renders into TASE ablation directory structure.

    DreamCatalyst renders live at:
        renders_root/{scene}/{experiment}/train/rgb/frame_XXXXX.jpg
        renders_root/{scene}/original_scene/train/rgb/frame_XXXXX.jpg

    This converts them to:
        ablation_dir/{tase_scene}/{method_name}/{experiment}/imgs/rgb_XXX.png
        ablation_dir/{tase_scene}/{method_name}/original_scene/imgs/rgb_XXX.png
    """
    if not renders_root.exists():
        raise FileNotFoundError(f"DC renders root not found: {renders_root}")

    for scene_dir in sorted(renders_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        tase_scene = SCENE_NAME_MAP.get(scene_name, scene_name)

        for experiment_dir in sorted(scene_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue
            experiment_name = experiment_dir.name

            rgb_dir = experiment_dir / "train" / "rgb"
            if not rgb_dir.exists():
                logger.warning(
                    "No renders found for %s/%s at %s",
                    scene_name, experiment_name, rgb_dir,
                )
                continue

            # Collect and sort source images
            src_images = sorted(rgb_dir.glob("frame_*.jpg"))
            if not src_images:
                logger.warning("No frame_*.jpg files in %s", rgb_dir)
                continue

            out_dir = ablation_dir / tase_scene / method_name / experiment_name / "imgs"
            out_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Converting %d images: %s/%s → %s",
                len(src_images), scene_name, experiment_name, out_dir,
            )

            for idx, src_path in enumerate(src_images):
                dst_path = out_dir / f"rgb_{idx:03d}.png"
                if dst_path.exists():
                    continue
                img = Image.open(src_path)
                img.save(dst_path)

    logger.info("DreamCatalyst render preparation complete.")


def convert_editsplat_renders(
    renders_root: Path,
    ablation_dir: Path,
    method_name: str = "EditSplat",
) -> None:
    """Convert EditSplat renders into TASE ablation directory structure.

    Placeholder for future EditSplat integration.
    """
    raise NotImplementedError("EditSplat render conversion not yet implemented")


BASELINE_CONVERTERS = {
    "DreamCatalyst": convert_dc_renders,
    "EditSplat": convert_editsplat_renders,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare external baseline renders for TASE evaluation"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=list(BASELINE_CONVERTERS.keys()),
        help="Baseline method name",
    )
    parser.add_argument(
        "--renders-root",
        type=Path,
        required=True,
        help="Root directory of baseline renders",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        required=True,
        help="TASE ablation output directory (contains prompts.json)",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Override the method name used in the ablation directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    method_name = args.method_name or args.baseline
    converter = BASELINE_CONVERTERS[args.baseline]
    converter(args.renders_root, args.ablation_dir, method_name)


if __name__ == "__main__":
    main()
