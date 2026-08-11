#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Benchmark the shortest and longest local edits to measure wall-clock runtime.
#
# Shortest: bike_edits/blue_bike        (auto_edit_iters=4,  auto_train_steps=200  → 800  total opt steps)
# Longest:  bike_edits/enduro  (auto_edit_iters=12, auto_train_steps=300  → 3600 total opt steps)
#
# Both edits use the bicycle scene (bike_edits/0_param_adjust.json for scene setup).
# Each edit is run as a fresh process so model-load time is included in both measurements.
#
# Usage:
#   bash scripts/benchmark_edit_runtime.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
export TASE_ROOT="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configuration (mirrors run_all_edits.sh)
# ---------------------------------------------------------------------------
RECONSTRUCTION="OUTPUT/reconstruction/"
NEW_DATASET="OUTPUT/editing"
SESSION_NAME="runtime_benchmark"

MRL_MODEL_PATH="WEIGHTS/mrl_adapter/epoch_1.pth"
CTRL_NET_PATH="WEIGHTS/ctrlnet/spltctrl/diffusion_pytorch_model.safetensors"
TRANSFORMER_PATH="WEIGHTS/transformer/diffusion_pytorch_model.safetensors"

CFG_DIR="$REPO_ROOT/TASE/splatting/configs"
BIKE_PARAM_ADJUST="$CFG_DIR/bike_edits/0_param_adjust.json"

# ---------------------------------------------------------------------------
# Helper: run a single edit and print elapsed time in minutes
# ---------------------------------------------------------------------------
run_and_time() {
    local label="$1"
    local preset_file="$2"

    echo ""
    echo "====================================================================="
    echo " Timing: $label"
    echo "====================================================================="

    local t0
    t0=$(date +%s)

    cd "$REPO_ROOT/TASE/splatting"
    python editor_main.py \
        --name "$SESSION_NAME" \
        --new-dataset "$NEW_DATASET" \
        --reconstruction "$RECONSTRUCTION" \
        --mrl-model-path "$MRL_MODEL_PATH" \
        --ctrl-net-path "$CTRL_NET_PATH" \
        --transformer-path "$TRANSFORMER_PATH" \
        --preset "$BIKE_PARAM_ADJUST" "$preset_file" \
        --auto-edit

    local t1
    t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local minutes
    minutes=$(python3 -c "print(f'{$elapsed/60:.2f}')")

    echo ""
    echo " >>> $label"
    echo "     Elapsed: ${elapsed}s  (${minutes} min)"
}

# ---------------------------------------------------------------------------
# Run both benchmarks sequentially
# ---------------------------------------------------------------------------

# Shortest edit: blue_bike  (4 iters × 200 steps = 800 total opt steps)
run_and_time \
    "shortest — bike_edits/blue_bike  [4 iters × 200 steps = 800 total opt steps]" \
    "$CFG_DIR/bike_edits/blue_bike.json"

# Longest edit: enduro  (12 iters × 300 steps = 3600 total opt steps)
run_and_time \
    "longest  — bike_edits/enduro  [12 iters × 300 steps = 3600 total opt steps]" \
    "$CFG_DIR/bike_edits/enduro.json"

echo ""
echo "====================================================================="
echo " Runtime benchmark complete."
echo "====================================================================="
