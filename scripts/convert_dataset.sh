#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Convert one or more COLMAP/LLFF scenes to the TASE format and extract
# per-image semantic feature maps using the MRL adapter.  The resulting
# dataset is used as input for scene reconstruction (trainer_main.py).
#
# The script handles the two common workflows:
#   a) MRL adapter features  (default, requires --mrl-ckpt)
#   b) Raw DINOv3 features   (pass --raw-dino, no adapter needed)
#
# Input dataset layout expected under DATA/<source-subdir>/<scene>/:
#   images/ (or images_4/, images_8/, …)   — training images
#   sparse/0/                               — COLMAP reconstruction
#
# Prerequisites:
#   - TASE environment activated (uv sync && source .venv/bin/activate)
#   - $TASE_ROOT set to the repository root
#   - A trained MRL adapter checkpoint (see train_mrl_adapter.sh),
#     OR pass --raw-dino to skip the adapter.
#
# Usage:
#   bash scripts/convert_dataset.sh \
#       --scenes bicycle garden counter \
#       [--source-subdir mipNeRF360] \
#       [--mrl-ckpt WEIGHTS/mrl_adapter/epoch_1.pth] \
#       [--raw-dino] \
#       [--prefer-down 4] \
#       [--val-frac 0.10]
#
# Output is written to:
#   DATA/TASE_360_<run_name>/<scene>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCENES=()
SOURCE_SUBDIR="DATA/mipNeRF360"
MRL_CKPT="WEIGHTS/mrl_adapter/epoch_1.pth"
RAW_DINO=false
PREFER_DOWN=4
VAL_FRAC=0.0
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                SCENES+=("$1"); shift
            done
            ;;
        --source-subdir)  SOURCE_SUBDIR="$2";  shift 2 ;;
        --mrl-ckpt)       MRL_CKPT="$2";       shift 2 ;;
        --raw-dino)       RAW_DINO=true;        shift   ;;
        --prefer-down)    PREFER_DOWN="$2";     shift 2 ;;
        --val-frac)       VAL_FRAC="$2";        shift 2 ;;
        *)                EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "[ERROR] No scenes specified. Pass --scenes <scene1> <scene2> ..."
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
# Build flags
# ---------------------------------------------------------------------------
FEATURE_ARGS=""
if $RAW_DINO; then
    FEATURE_ARGS="--features --raw-dino"
else
    FEATURE_ARGS="--features --feature-ckpt $MRL_CKPT"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

echo "================================================================"
echo "  Converting dataset to TASE format"
echo "  Scenes      : ${SCENES[*]}"
echo "  Source dir  : $SOURCE_SUBDIR"
echo "  Raw DINOv3  : $RAW_DINO"
if ! $RAW_DINO; then
    echo "  MRL ckpt    : $MRL_CKPT"
fi
echo "  Prefer down : $PREFER_DOWN"
echo "  Val frac    : $VAL_FRAC"
echo "  Output dir  : $REPO_ROOT/DATA/TASE_360/"
echo "================================================================"

python -m TASE.splatting.data.mipNerf_tase_converter \
    --scenes "${SCENES[@]}" \
    --old-subdir "$SOURCE_SUBDIR" \
    --new-subdir "$REPO_ROOT/DATA/TASE_360/" \
    --val-frac "$VAL_FRAC" \
    --prefer-down "$PREFER_DOWN" \
    $FEATURE_ARGS \
    $EXTRA_ARGS

echo "================================================================"
echo "  Done. TASE dataset(s) written under: $REPO_ROOT/DATA/"
echo "================================================================"
