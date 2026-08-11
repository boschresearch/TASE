#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Build the SplatCtrl dataset used for the ControlNet artifact-fixing finetuning
# stage (Stage 2 of the TASE training pipeline).
#
# The builder renders (corrupted-splat, clean-image, feature-map) triplets from
# AnySplat reconstructions of ImageNet images.  It must be executed from within
# the AnySplat repository because it imports AnySplat internals directly.
#
# Prerequisites:
#   - AnySplat cloned and installed in a separate environment
#     (https://github.com/InternRobotics/AnySplat)
#   - $ANYSPLAT_ROOT pointing to the AnySplat repo root
#   - $IMGNET_ROOT pointing to the ImageNet ILSVRC2012 images root
#   - A trained MRL adapter checkpoint (or use --raw-dino)
#
# Usage:
#   bash scripts/build_splatctrl_dataset.sh \
#       --mrl-ckpt WEIGHTS/mrl_adapter/epoch_1.pth \
#       [--num-samples 50000] \
#       [--output-dir DATA/SplatCtrl] \
#       [--raw-dino]
#
# Output is written to:
#   DATA/SplatCtrl/train/<timestamp>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MRL_CKPT=""
NUM_SAMPLES=50000
OUTPUT_DIR="$REPO_ROOT/DATA/SplatCtrl"
RAW_DINO=false
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mrl-ckpt)      MRL_CKPT="$2";     shift 2 ;;
        --num-samples)   NUM_SAMPLES="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";   shift 2 ;;
        --raw-dino)      RAW_DINO=true;     shift   ;;
        *)               EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
if [[ -z "${ANYSPLAT_ROOT:-}" ]]; then
    echo "[ERROR] \$ANYSPLAT_ROOT is not set."
    echo "        Clone AnySplat and export ANYSPLAT_ROOT=<path to repo>."
    echo "        See https://github.com/InternRobotics/AnySplat"
    exit 1
fi

if [[ -z "${IMGNET_ROOT:-}" ]]; then
    echo "[ERROR] \$IMGNET_ROOT is not set."
    echo "        export IMGNET_ROOT=<path to ImageNet ILSVRC2012 images root>"
    exit 1
fi

if ! $RAW_DINO && [[ -z "$MRL_CKPT" ]]; then
    MRL_CKPT="$REPO_ROOT/WEIGHTS/mrl_adapter/epoch_1.pth"
fi

if ! $RAW_DINO && [[ ! -f "$MRL_CKPT" ]]; then
    echo "[ERROR] MRL adapter checkpoint not found: $MRL_CKPT"
    echo "        Pass --mrl-ckpt <path> or use --raw-dino."
    exit 1
fi

# ---------------------------------------------------------------------------
# Copy the builder script to the AnySplat repo (required for its imports)
# ---------------------------------------------------------------------------
BUILDER_SRC="$REPO_ROOT/TASE/image_gen/controlnet/splat_ctrl_dataset_builder.py"
BUILDER_DST="$ANYSPLAT_ROOT/splat_ctrl_dataset_builder.py"

if [[ ! -f "$BUILDER_DST" ]]; then
    echo "Copying dataset builder to AnySplat repo..."
    cp "$BUILDER_SRC" "$BUILDER_DST"
fi

# ---------------------------------------------------------------------------
# Activate the AnySplat Python environment
# ---------------------------------------------------------------------------
if [[ -f "$ANYSPLAT_ROOT/.venv/bin/activate" ]]; then
    source "$ANYSPLAT_ROOT/.venv/bin/activate"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "[WARN] Using active conda environment: $CONDA_DEFAULT_ENV"
else
    echo "[ERROR] No AnySplat virtual environment found at $ANYSPLAT_ROOT/.venv"
    echo "        Activate the AnySplat environment manually before running this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$ANYSPLAT_ROOT"

MRL_ARG=""
RAW_DINO_ARG=""
if $RAW_DINO; then
    RAW_DINO_ARG="--raw-dino"
else
    MRL_ARG="--mrl-model-path $MRL_CKPT"
fi

echo "================================================================"
echo "  Building SplatCtrl dataset"
echo "  AnySplat root : $ANYSPLAT_ROOT"
echo "  ImageNet      : $IMGNET_ROOT"
echo "  Num samples   : $NUM_SAMPLES"
echo "  Output dir    : $OUTPUT_DIR"
echo "  Raw DINO      : $RAW_DINO"
if ! $RAW_DINO; then
    echo "  MRL ckpt      : $MRL_CKPT"
fi
echo "================================================================"

python splat_ctrl_dataset_builder.py \
    $MRL_ARG \
    $RAW_DINO_ARG \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --imagenet-root "$IMGNET_ROOT" \
    --dataset-split train \
    $EXTRA_ARGS

echo "================================================================"
echo "  Done. Dataset written to: $OUTPUT_DIR/train/"
echo "================================================================"
