#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Finetune the FLUX ControlNet on the SplatCtrl dataset to fix splat-render
# artifacts (Stage 2 of the TASE training pipeline).
#
# This stage adapts the Stage-1 ControlNet to the distribution of rendered
# Gaussian splat images, improving generation quality at edit time.
# The transformer (FLUX.1-dev) is jointly finetuned with a lower learning rate.
#
# Prerequisites:
#   - TASE environment activated (uv sync && source .venv/bin/activate)
#   - A completed Stage-1 ControlNet checkpoint
#   - A SplatCtrl dataset built with build_splatctrl_dataset.sh
#
# Usage:
#   bash scripts/finetune_controlnet.sh \
#       --ctrlnet-ckpt OUTPUT/ControlNet/<run>/ckpts/controlnet_final_stage_1_res_512/diffusion_pytorch_model.safetensors \
#       --dataset-root DATA/SplatCtrl/train/<timestamp> \
#       [--steps 500000] \
#       [--raw-dino] \
#       [--multi-gpu]
#
# Outputs are written to:
#   OUTPUT/ControlNet/splat_finetune/<run_id>/ckpts/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CTRLNET_CKPT=""
DATASET_ROOT=""
STEPS=500000
RAW_DINO=false
MULTI_GPU=false
DTYPE="bf16"
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctrlnet-ckpt)  CTRLNET_CKPT="$2";  shift 2 ;;
        --dataset-root)  DATASET_ROOT="$2";  shift 2 ;;
        --steps)         STEPS="$2";         shift 2 ;;
        --raw-dino)      RAW_DINO=true;      shift   ;;
        --multi-gpu)     MULTI_GPU=true;     shift   ;;
        --dtype)         DTYPE="$2";         shift 2 ;;
        *)               EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$CTRLNET_CKPT" ]]; then
    echo "[ERROR] --ctrlnet-ckpt is required."
    echo "        Pass the path to a Stage-1 ControlNet safetensors checkpoint."
    exit 1
fi

if [[ ! -f "$CTRLNET_CKPT" ]]; then
    echo "[ERROR] ControlNet checkpoint not found: $CTRLNET_CKPT"
    exit 1
fi

if [[ -z "$DATASET_ROOT" ]]; then
    echo "[ERROR] --dataset-root is required."
    echo "        Pass the SplatCtrl dataset directory produced by build_splatctrl_dataset.sh."
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[ERROR] SplatCtrl dataset directory not found: $DATASET_ROOT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build optional flags
# ---------------------------------------------------------------------------
RAW_DINO_ARGS=""
if $RAW_DINO; then
    RAW_DINO_ARGS="--raw-dino --max-control-channels 768"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

TRAIN_CMD=(
    python -m TASE.image_gen.controlnet.ctrlnet_trainer
    --dataset-root "$DATASET_ROOT"
    --result-dir "$REPO_ROOT/OUTPUT/ControlNet"
    --resume-from-ckpt "$CTRLNET_CKPT"
    --splat-ctrl-finetuning
    --stages "512,4,$STEPS"
    --dtype "$DTYPE"
    --tb
    $RAW_DINO_ARGS
    $EXTRA_ARGS
)

if $MULTI_GPU; then
    TRAIN_CMD=(accelerate launch --multi-gpu -m TASE.image_gen.controlnet.ctrlnet_trainer
        --dataset-root "$DATASET_ROOT"
        --result-dir "$REPO_ROOT/OUTPUT/ControlNet"
        --resume-from-ckpt "$CTRLNET_CKPT"
        --splat-ctrl-finetuning
        --stages "512,4,$STEPS"
        --dtype "$DTYPE"
        --tb
        $RAW_DINO_ARGS
        $EXTRA_ARGS)
fi

echo "================================================================"
echo "  Finetuning ControlNet on SplatCtrl dataset"
echo "  ControlNet ckpt : $CTRLNET_CKPT"
echo "  Dataset         : $DATASET_ROOT"
echo "  Steps           : $STEPS"
echo "  Raw DINO        : $RAW_DINO"
echo "  multi-GPU       : $MULTI_GPU"
echo "  Output          : $REPO_ROOT/OUTPUT/ControlNet/splat_finetune/"
echo "================================================================"

"${TRAIN_CMD[@]}"

echo "================================================================"
echo "  Done. Checkpoints saved under:"
echo "    $REPO_ROOT/OUTPUT/ControlNet/splat_finetune/"
echo "================================================================"
