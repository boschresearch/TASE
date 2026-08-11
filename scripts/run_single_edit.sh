#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Run a single edit preset from a specific JSON config file.
#
# Parameter priority (lowest → highest):
#   base_parameters.json  <  0_param_adjust.json (same dir)  <  EDIT_CONFIG
#
# The 0_param_adjust.json in the same directory as EDIT_CONFIG is applied
# automatically before EDIT_CONFIG — no need to specify it manually.
#
# Usage:
#   bash scripts/run_single_edit.sh
#
# To run directly via Python:
#   source .venv/bin/activate && python TASE/splatting/editor_main.py \
#       --name <session_name> \
#       --new-dataset OUTPUT/editing \
#       --reconstruction OUTPUT/reconstruction/ \
#       --mrl-model-path WEIGHTS/mrl_adapter/epoch_1.pth \
#       --ctrl-net-path WEIGHTS/ctrlnet/spltctrl/diffusion_pytorch_model.safetensors \
#       --transformer-path WEIGHTS/transformer/diffusion_pytorch_model.safetensors \
#       --preset TASE/splatting/configs/bear_edits/tiger_test.json \
#       --auto-edit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configuration — edit these to change what runs
# ---------------------------------------------------------------------------
SESSION_NAME="single_edit"
EDIT_CONFIG="$1"

RECONSTRUCTION="OUTPUT/reconstruction"
NEW_DATASET="OUTPUT/editing"

MRL_MODEL_PATH="WEIGHTS/mrl_adapter/epoch_1.pth"
CTRL_NET_PATH="WEIGHTS/ctrlnet/spltctrl/diffusion_pytorch_model.safetensors"
TRANSFORMER_PATH="WEIGHTS/transformer/diffusion_pytorch_model.safetensors"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

echo "Running edit: $(basename "$EDIT_CONFIG")"

python TASE/splatting/editor_main.py \
    --name "$SESSION_NAME" \
    --new-dataset "$NEW_DATASET" \
    --reconstruction "$RECONSTRUCTION" \
    --mrl-model-path "$MRL_MODEL_PATH" \
    --ctrl-net-path "$CTRL_NET_PATH" \
    --transformer-path "$TRANSFORMER_PATH" \
    --preset "$EDIT_CONFIG" \
    --auto-edit

echo "Done. Results in: $NEW_DATASET/$SESSION_NAME/"
