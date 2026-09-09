#!/usr/bin/env bash
# Full BioSR LOSO experiment suite (AGENT.md §3 folds, §12 ablations).
#
# Usage (on the training server):
#   export BIOSR_ROOT=/abs/path/to/BioSR
#   bash scripts/run_loso.sh core       # 4 folds x {dfcan, rcan}          -> 8 training runs
#   bash scripts/run_loso.sh ablation   # 4 folds x {grad-off, fourier}   -> 8 runs (run 'core' first)
#   bash scripts/run_loso.sh bicubic    # 4 folds x bicubic reference     -> eval only, no training
#
# Extra overrides applied to every run, e.g. a shorter first pass:
#   EXTRA="training.epochs=100 training.batch_size=8" bash scripts/run_loso.sh core
#
# Long runs: keep them alive in tmux/nohup, e.g.:
#   nohup bash scripts/run_loso.sh core > train_core.log 2>&1 &
#
# Results per run: runs/<timestamp>_<model>_<heldout>/results/test_metrics.json
# (per-level + overall PSNR/SSIM/ZNCC/FRC and structural Precision/Recall/F1).
set -euo pipefail

: "${BIOSR_ROOT:?export BIOSR_ROOT=/abs/path/to/BioSR first}"
export BIOSR_ROOT
MODE="${1:-core}"
PY="${PYTHON:-python}"
EXTRA=(${EXTRA:-})

train() { # <config> <held_out> [overrides...]
  local config="$1" held="$2"
  shift 2
  echo ""
  echo "=== TRAIN $(basename "$config" .yaml) | held-out: $held | extra: $* ==="
  $PY train.py --config "$config" "dataset.held_out_structure=$held" "$@" "${EXTRA[@]}"
}

case "$MODE" in
  core)
    # Fold x backbone matrix (default loss: charbonnier + ssim 0.1 + gradient 0.1).
    for held in CCPs ER Microtubules F-actin; do
      train configs/dfcan.yaml "$held"
      train configs/rcan.yaml "$held"
    done
    ;;
  ablation)
    # AGENT.md §12 - one variable at a time; 'core' is the baseline for these deltas.
    # grad-off: isolates the gradient term (loss.gradient 0.1 -> 0.0)
    # fourier:  adds the optional Fourier loss at the small weight §8 recommends
    # (F: morphology-OOD augmentation is reserved and not implemented yet.)
    for held in CCPs ER Microtubules F-actin; do
      train configs/dfcan.yaml "$held" loss.gradient=0.0
      train configs/dfcan.yaml "$held" loss.fourier=0.1
    done
    ;;
  bicubic)
    # Reference floor: epochs=0 skips training and evaluates the interpolator
    # directly on the held-out fold with the standard per-level metrics.
    for held in CCPs ER Microtubules F-actin; do
      train configs/bicubic.yaml "$held"
    done
    ;;
  *)
    echo "unknown mode: $MODE (use core | ablation | bicubic)" >&2
    exit 1
    ;;
esac

echo ""
echo "All '$MODE' runs complete. See runs/<timestamp>_<model>_<heldout>/results/test_metrics.json"
