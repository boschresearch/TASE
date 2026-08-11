#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Train the MRL (Matryoshka Representation Learning) adapter that compresses
# DINOv3 patch features to a compact, truncatable embedding space (Stage 0 of
# the TASE training pipeline).
#
# The resulting checkpoint is used as the control-signal encoder for the FLUX
# ControlNet and for per-Gaussian feature storage during scene reconstruction.
#
# Prerequisites:
#   - TASE environment activated (uv sync && source .venv/bin/activate)
#   - $IMGNET_ROOT pointing to the ImageNet ILSVRC2012 images root
#
# Usage:
#   bash scripts/train_mrl_adapter.sh \
#       [--max-dims 64] \
#       [--backbone dinov3] \
#       [--batch 64] \
#       [--epochs 1] \
#       [--equivariance-lambda 1000.0] \
#       [--resume <checkpoint.pth>]
#
# Outputs are written to:
#   OUTPUT/models/MRL/<backbone>/<run_id>/ckpt/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_DIMS=64
BACKBONE="dinov3"
BATCH=64
EPOCHS=1
EQV_LAMBDA=1000.0
RESUME=""
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-dims)              MAX_DIMS="$2";    shift 2 ;;
        --backbone)              BACKBONE="$2";   shift 2 ;;
        --batch)                 BATCH="$2";      shift 2 ;;
        --epochs)                EPOCHS="$2";     shift 2 ;;
        --equivariance-lambda)   EQV_LAMBDA="$2"; shift 2 ;;
        --resume)                RESUME="$2";     shift 2 ;;
        *)                       EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "${IMGNET_ROOT:-}" ]]; then
    echo "[ERROR] \$IMGNET_ROOT is not set."
    echo "        export IMGNET_ROOT=<path to ImageNet ILSVRC2012 images root>"
    exit 1
fi

RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
    if [[ ! -f "$RESUME" ]]; then
        echo "[ERROR] Resume checkpoint not found: $RESUME"
        exit 1
    fi
    RESUME_ARG="--ckpt $RESUME"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

echo "================================================================"
echo "  Training MRL adapter (AERoPE)"
echo "  Backbone   : $BACKBONE"
echo "  Max dims   : $MAX_DIMS"
echo "  Batch size : $BATCH"
echo "  Epochs     : $EPOCHS"
echo "  EQV lambda : $EQV_LAMBDA"
echo "  ImageNet   : $IMGNET_ROOT"
echo "  Output     : $REPO_ROOT/OUTPUT/models/MRL/$BACKBONE/"
if [[ -n "$RESUME" ]]; then
    echo "  Resume     : $RESUME"
fi
echo "================================================================"

python -m TASE.image_gen.dimensionality_reduction.mrl_trainer \
    --imagenet_root "$IMGNET_ROOT" \
    --backbone "$BACKBONE" \
    --res 512 \
    --batch "$BATCH" \
    --epochs "$EPOCHS" \
    --max_dims "$MAX_DIMS" \
    --equivariance_lambda "$EQV_LAMBDA" \
    $RESUME_ARG \
    $EXTRA_ARGS

echo "================================================================"
echo "  Done. Checkpoint saved under:"
echo "    $REPO_ROOT/OUTPUT/models/MRL/$BACKBONE/"
echo "================================================================"
