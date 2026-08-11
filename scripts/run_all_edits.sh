#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Run all edit presets with default settings (no truncation override).
# Each preset directory is passed in full so editor_main.py processes all
# JSON files inside it sequentially in alphabetical order.
# (0_param_adjust.json files are skipped because they set auto_edit=false.)
#
# Output lands in: OUTPUT/editing/<SESSION_NAME>/<preset_dir_name>/
#
# Usage:
#   bash scripts/run_all_edits.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSION_NAME="all_edits"

RECONSTRUCTION="OUTPUT/reconstruction/"
NEW_DATASET="OUTPUT/editing"

MRL_MODEL_PATH="WEIGHTS/mrl_adapter/epoch_1.pth"
CTRL_NET_PATH="WEIGHTS/ctrlnet/spltctrl/diffusion_pytorch_model.safetensors"
TRANSFORMER_PATH="WEIGHTS/transformer/diffusion_pytorch_model.safetensors"

CFG_DIR="$REPO_ROOT/TASE/splatting/configs"

# ---------------------------------------------------------------------------
# Preset directories to process (each directory is one "scene + edit group")
# Comment out any groups you want to skip.
# ---------------------------------------------------------------------------
PRESET_DIRS=(
    "$CFG_DIR/bike_edits"       # bicycle scene, local edits
    "$CFG_DIR/bike_global"      # bicycle scene, global edits
    "$CFG_DIR/garden_global"    # garden scene, global edits
    "$CFG_DIR/face_edits"       # face scene, local edits
    "$CFG_DIR/bear_edits"       # bear scene, local edits
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/TASE/splatting"

for PRESET_DIR in "${PRESET_DIRS[@]}"; do
    DIR_NAME=$(basename "$PRESET_DIR")
    echo ""
    echo "====================================================================="
    echo " Scene/group: $DIR_NAME"
    echo "====================================================================="

    python editor_main.py \
        --name "$SESSION_NAME" \
        --new-dataset "$NEW_DATASET" \
        --reconstruction "$RECONSTRUCTION" \
        --mrl-model-path "$MRL_MODEL_PATH" \
        --ctrl-net-path "$CTRL_NET_PATH" \
        --transformer-path "$TRANSFORMER_PATH" \
        --preset "$PRESET_DIR" \
        --auto-edit

    echo " Done: $DIR_NAME"
done

echo ""
echo "All edits complete. Results in: $NEW_DATASET/$SESSION_NAME/"
