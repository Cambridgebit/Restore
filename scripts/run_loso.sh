#!/usr/bin/env bash
# Full BioSR LOSO experiment suite (AGENT.md §3 folds, §12 ablations) - hardened:
#   * each run is an isolated process; a failed run NEVER kills the suite
#   * completed runs are skipped on rerun (resume) - results/test_metrics.json is
#     the completion marker; set SKIP_EXISTING=0 to force a full rerun
#   * CUDA cache is cleared after every run; alloc fragmentation is mitigated via
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#   * ablation variants write to separate output dirs (no skip-name collisions)
#
# Usage:
#   export BIOSR_ROOT=/abs/path/to/BioSR
#   bash scripts/run_loso.sh core        # 4 folds x {dfcan, rcan}   -> runs/
#   bash scripts/run_loso.sh ablation    # 4 folds x {grad-off, fourier}
#                                        #   -> runs_ab_gradoff/ and runs_ab_fourier/
#   bash scripts/run_loso.sh bicubic     # eval-only reference       -> runs/
#
# Env knobs:
#   EXTRA="training.epochs=100 ..."  extra overrides for every run
#   SKIP_EXISTING=0                  force full rerun (default 1 = resume)
#   PYTHON=/path/to/python           interpreter (default: python)
#
# Long runs: nohup bash scripts/run_loso.sh core > train_core.log 2>&1 &
set -uo pipefail

: "${BIOSR_ROOT:?export BIOSR_ROOT=/abs/path/to/BioSR first}"
export BIOSR_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MODE="${1:-core}"
PY="${PYTHON:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA=(${EXTRA:-})
FAILED=()

cleanup_gpu() {
  # Belt-and-braces: the training process already released its CUDA context on
  # exit; this clears any residual allocator cache and confirms a clean GPU.
  "$PY" - <<'PYEOF' >/dev/null 2>&1 || true
import gc

import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYEOF
  echo "=== cleanup: gpu cache cleared ==="
}

gpu_preflight() {
  # Fail fast (before burning a startup cycle) when the GPU is mostly occupied
  # by another process. Threshold overridable via MIN_GPU_FREE_MIB.
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if (( free_mib < ${MIN_GPU_FREE_MIB:-6000} )); then
    echo "!!! GPU has only ${free_mib} MiB free (< ${MIN_GPU_FREE_MIB:-6000}); another process is likely occupying it:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    echo "    Free the GPU first (or set MIN_GPU_FREE_MIB=... lower) and rerun." >&2
    return 1
  fi
  return 0
}

train() { # <config> <held_out> [overrides...]
  local config="$1" held="$2" model out_dir="runs" arg
  model="$(basename "$config" .yaml)"
  shift 2
  for arg in "$@"; do
    case "$arg" in output_dir=*) out_dir="${arg#output_dir=}" ;; esac
  done
  if [[ "$SKIP_EXISTING" == 1 ]] && compgen -G "$out_dir/*_${model}_${held}/results/test_metrics.json" >/dev/null; then
    echo "=== SKIP  $model | $held (completed run in $out_dir; SKIP_EXISTING=0 to retrain) ==="
    return 0
  fi
  echo ""
  echo "=== TRAIN $model | held-out: $held | extra: $* ==="
  if ! gpu_preflight; then
    FAILED+=("$model/$held (gpu occupied)")
    return 0
  fi
  if "$PY" train.py --config "$config" "dataset.held_out_structure=$held" "$@" "${EXTRA[@]}"; then
    echo "=== DONE  $model | $held ==="
  else
    echo "!!! FAILED $model | $held (suite continues)" >&2
    FAILED+=("$model/$held")
  fi
  cleanup_gpu
}

case "$MODE" in
  core)
    for held in CCPs ER Microtubules F-actin; do
      train configs/dfcan.yaml "$held"
      train configs/rcan.yaml "$held"
    done
    ;;
  ablation)
    # AGENT.md §12 - one variable at a time; 'core' is the baseline for these deltas.
    for held in CCPs ER Microtubules F-actin; do
      train configs/dfcan.yaml "$held" output_dir=runs_ab_gradoff loss.gradient=0.0
      train configs/dfcan.yaml "$held" output_dir=runs_ab_fourier loss.fourier=0.1
    done
    ;;
  bicubic)
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
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "SUITE FINISHED WITH ${#FAILED[@]} FAILURE(S): ${FAILED[*]}"
  echo "Rerun the same command to retry only the failed/missing runs (completed runs are skipped)."
  exit 1
fi
echo "SUITE COMPLETE. Results: <output_dir>/<timestamp>_<model>_<heldout>/results/test_metrics.json"
