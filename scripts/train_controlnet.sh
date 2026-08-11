#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Train the FLUX ControlNet on ImageNet-1k using the MRL adapter as the
# control-signal encoder (Stage 1 of the TASE training pipeline).
#
# Prerequisites:
#   - TASE environment activated (uv sync && source .venv/bin/activate)
#   - $IMGNET_ROOT pointing to the ImageNet ILSVRC2012 images root
#   - A trained MRL adapter checkpoint (see train_mrl_adapter.sh or paper)
#
# Usage:
#   bash scripts/train_controlnet.sh \
#       --mrl-ckpt WEIGHTS/mrl_adapter/epoch_1.pth \
#       [--stages 512,4,1281167] \
#       [--dtype bf16] \
#       [--multi-gpu]
#
# Outputs are written to:
#   OUTPUT/ControlNet/<run_id>/ckpts/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults (can be overridden via CLI flags below)
# ---------------------------------------------------------------------------
MRL_CKPT=""
STAGES="512,4,1281167"    # res,batch_per_gpu,total_images — H200/B200
DTYPE="bf16"
MULTI_GPU=false
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mrl-ckpt)       MRL_CKPT="$2";    shift 2 ;;
        --stages)         STAGES="$2";      shift 2 ;;
        --dtype)          DTYPE="$2";       shift 2 ;;
        --multi-gpu)      MULTI_GPU=true;   shift   ;;
        *)                EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$MRL_CKPT" ]]; then
    # Fall back to conventional WEIGHTS location
    MRL_CKPT="$REPO_ROOT/WEIGHTS/mrl_adapter/epoch_1.pth"
fi

if [[ ! -f "$MRL_CKPT" ]]; then
    echo "[ERROR] MRL adapter checkpoint not found: $MRL_CKPT"
    echo "        Pass --mrl-ckpt <path> or place the checkpoint at WEIGHTS/mrl_adapter/epoch_1.pth"
    exit 1
fi

if [[ -z "${IMGNET_ROOT:-}" ]]; then
    echo "[ERROR] \$IMGNET_ROOT is not set. Export it before running this script:"
    echo "        export IMGNET_ROOT=<path to ImageNet ILSVRC2012 images root>"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

echo "================================================================"
echo "  Training FLUX ControlNet on ImageNet"
echo "  MRL adapter : $MRL_CKPT"
echo "  Stages      : $STAGES"
echo "  dtype       : $DTYPE"
echo "  multi-GPU   : $MULTI_GPU"
echo "  ImageNet    : $IMGNET_ROOT"
echo "  Output      : $REPO_ROOT/OUTPUT/ControlNet/"
echo "================================================================"

if $MULTI_GPU; then
    accelerate launch --multi-gpu \
        -m TASE.image_gen.controlnet.ctrlnet_trainer \
        --dataset-root "$IMGNET_ROOT" \
        --result-dir "$REPO_ROOT/OUTPUT/ControlNet" \
        --mrl-embedding-model-path "$MRL_CKPT" \
        --stages "$STAGES" \
        --ctrl-backbone dinov3 \
        --max-control-channels 64 \
        --dtype "$DTYPE" \
        --tb $EXTRA_ARGS
else
    python -m TASE.image_gen.controlnet.ctrlnet_trainer \
        --dataset-root "$IMGNET_ROOT" \
        --result-dir "$REPO_ROOT/OUTPUT/ControlNet" \
        --mrl-embedding-model-path "$MRL_CKPT" \
        --stages "$STAGES" \
        --ctrl-backbone dinov3 \
        --max-control-channels 64 \
        --dtype "$DTYPE" \
        --tb $EXTRA_ARGS
fi

echo "================================================================"
echo "  Done. Checkpoints saved under: $REPO_ROOT/OUTPUT/ControlNet/"
echo "================================================================"
