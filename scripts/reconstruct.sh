#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Reconstruct one or more scenes as 3D Gaussian splatting models with
# per-Gaussian TASE semantic features.
#
# Input scenes must first be converted to the TASE format via
# convert_dataset.sh.  The dataset subdir passed here should match the
# output of that script (e.g. TASE_360_<run_name>).
#
# Prerequisites:
#   - TASE environment activated (uv sync && source .venv/bin/activate)
#   - $TASE_ROOT set to the repository root
#   - TASE-format dataset available under DATA/<dataset-subdir>/
#
# Usage:
#   bash scripts/reconstruct.sh \
#       --scenes bicycle garden counter \
#       --dataset-subdir TASE_360_epoch_1 \
#       [--config mipNeRF360.yml] \
#       [--name my_run]
#
# Output is written to:
#   OUTPUT/reconstruction/<name>/<scene>/ckpts/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCENES=()
DATASET_SUBDIR=""
CONFIG="mipNeRF360.yml"
RUN_NAME=""
CLEAN=false
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
        --dataset-subdir)  DATASET_SUBDIR="$2"; shift 2 ;;
        --config)          CONFIG="$2";         shift 2 ;;
        --name)            RUN_NAME="$2";        shift 2 ;;
        --clean)           CLEAN=true;           shift   ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "[ERROR] No scenes specified. Pass --scenes <scene1> <scene2> ..."
    exit 1
fi

if [[ -z "$DATASET_SUBDIR" ]]; then
    echo "[ERROR] --dataset-subdir is required."
    echo "        Pass the dataset directory name under DATA/ (e.g. TASE_360_epoch_1)."
    exit 1
fi

DATASET_PATH="$REPO_ROOT/DATA/$DATASET_SUBDIR"
if [[ ! -d "$DATASET_PATH" ]]; then
    echo "[ERROR] Dataset directory not found: $DATASET_PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional flags
# ---------------------------------------------------------------------------
CLEAN_ARG=""
if $CLEAN; then
    CLEAN_ARG="--clean"
fi

NAME_ARG=""
if [[ -n "$RUN_NAME" ]]; then
    NAME_ARG="--name $RUN_NAME"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

echo "================================================================"
echo "  Reconstructing scenes with TASE features"
echo "  Scenes     : ${SCENES[*]}"
echo "  Dataset    : $DATASET_PATH"
echo "  Config     : $CONFIG"
if [[ -n "$RUN_NAME" ]]; then
    echo "  Run name   : $RUN_NAME"
fi
echo "  Output     : $REPO_ROOT/OUTPUT/reconstruction/"
echo "================================================================"

python -m TASE.splatting.trainer_main \
    --config "TASE/splatting/configs/$CONFIG" \
    --data-dirs "$DATASET_SUBDIR" \
    --scenes "${SCENES[@]}" \
    $NAME_ARG \
    $CLEAN_ARG \
    $EXTRA_ARGS

echo "================================================================"
echo "  Done. Checkpoints saved under:"
echo "    $REPO_ROOT/OUTPUT/reconstruction/"
echo "================================================================"
