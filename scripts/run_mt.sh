#!/usr/bin/env bash
# Train every backbone on CCPs + ER + F-actin and evaluate on the unseen
# Microtubules (LOSO fold 3) - STRICTLY SERIAL.
#
#   * models are launched ONE AT A TIME: bash waits for each `train.py` process to
#     exit before the next starts, so two models never share the GPU (no VRAM blow-up)
#   * after every run the GPU cache is cleared and the process memory is released,
#     with a free-memory readout before the next model starts
#   * a per-GPU flock lock (shared with run_loso.sh / run_grid.sh) makes the suites
#     mutually exclusive, and KILL_STALE terminates leftover train.py on this GPU
#
# Usage:
#   export BIOSR_ROOT=/abs/path/to/BioSR
#   bash scripts/run_mt.sh                        # all models, GPU 0, held-out Microtubules
#   GPU=1 bash scripts/run_mt.sh                  # second GPU
#   MODELS="dfcan swinir" bash scripts/run_mt.sh  # subset
#   SKIP_EXISTING=0 bash scripts/run_mt.sh        # force retrain
#
# Long run: nohup bash scripts/run_mt.sh > run_mt.log 2>&1 &
#
# Env knobs:
#   BIOSR_ROOT=...   required data root
#   GPU=1            physical GPU index (default 0), exported as CUDA_VISIBLE_DEVICES
#   HELD=Microtubules  held-out structure evaluated at test time
#   MODELS="..."     models to train (default: all trainable backbones)
#   OUT_DIR=runs_mt  output directory
#   KILL_STALE=0     keep leftover train.py on this GPU (default 1 = terminate)
#   PYTHON=...       interpreter (default: python)
#   SKIP_EXISTING=0  force full rerun (default 1 = resume completed models)
#   EXTRA="..."      extra overrides applied to every run
#   MIN_GPU_FREE_MIB=6000  preflight free-VRAM threshold
set -uo pipefail

: "${BIOSR_ROOT:?export BIOSR_ROOT=/abs/path/to/BioSR first}"
export BIOSR_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

HELD="${HELD:-Microtubules}"
OUT_DIR="${OUT_DIR:-runs_mt}"
PY="${PYTHON:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA=(${EXTRA:-})
MODELS=(${MODELS:-dfcan rcan nafnet swinir mambair wavemixsr restormer flow_matching})
FAILED=()

# --- per-GPU single-instance + stale-process guard (shared lock with the suites) ---
GPU="${GPU:-0}"
KILL_STALE="${KILL_STALE:-1}"
LOCK_FILE="${LOCK_FILE:-/tmp/restore_loso_gpu${GPU}.lock}"
export CUDA_VISIBLE_DEVICES="$GPU"

gpu_pids() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' || true
}

kill_stale_training() {
  # Terminate leftover train.py processes on THIS GPU only (never another GPU's).
  local pid cmd
  local -a targets=()
  for pid in $(gpu_pids); do
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmd" == *train.py* ]] && targets+=("$pid")
  done
  ((${#targets[@]})) || return 0
  echo "!!! stale train.py on GPU$GPU -> terminating: ${targets[*]}" >&2
  kill "${targets[@]}" 2>/dev/null || true
  sleep 2
  for pid in "${targets[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "    still alive, SIGKILL $pid" >&2
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

acquire_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "!!! flock not found (install util-linux); refusing to run without a single-instance guard." >&2
    exit 1
  fi
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "!!! another training suite is already active on GPU$GPU (lock: $LOCK_FILE)." >&2
    echo "    Run one suite per GPU, or wait for the current one." >&2
    exit 1
  fi
}

gpu_preflight() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local free_mib
  free_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if (( free_mib < ${MIN_GPU_FREE_MIB:-6000} )); then
    echo "!!! GPU$GPU has only ${free_mib} MiB free (< ${MIN_GPU_FREE_MIB:-6000}); another process is likely occupying it:" >&2
    nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    return 1
  fi
  return 0
}

cleanup_gpu() {
  # Runs in a fresh process: the training process already freed its own CUDA
  # context and CPU heap on exit; this clears the allocator cache and reports the
  # free VRAM before the next model starts.
  "$PY" - <<'PYEOF' >/dev/null 2>&1 || true
import gc

import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYEOF
  sleep 2
  if command -v nvidia-smi >/dev/null 2>&1; then
    local free_mib
    free_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "=== cleanup: GPU$GPU free ${free_mib} MiB ==="
  else
    echo "=== cleanup: cache cleared ==="
  fi
}

model_config() {
  case "$1" in
    dfcan)         echo "configs/dfcan.yaml" ;;
    rcan)          echo "configs/rcan.yaml" ;;
    nafnet)        echo "configs/nafnet.yaml" ;;
    swinir)        echo "configs/swinir.yaml" ;;
    mambair)       echo "configs/mambair.yaml" ;;
    wavemixsr)     echo "configs/wavemixsr.yaml" ;;
    restormer)     echo "configs/restormer.yaml" ;;
    flow_matching) echo "configs/flowmatching.yaml" ;;
    *) echo "unknown model: $1" >&2; exit 1 ;;
  esac
}

echo "=== run_mt.sh | train: CCPs + ER + F-actin -> test: $HELD | GPU=$GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ==="
echo "=== plan: ${#MODELS[@]} model(s) x 1 fold = ${#MODELS[@]} SERIAL run(s) -> $OUT_DIR/<timestamp>_<model>_${HELD}/ ==="
echo "=== models: ${MODELS[*]} ==="

acquire_lock
if [[ "$KILL_STALE" == 1 ]]; then
  kill_stale_training
fi

for model in "${MODELS[@]}"; do
  config="$(model_config "$model")"
  if [[ "$SKIP_EXISTING" == 1 ]] && compgen -G "$OUT_DIR/*_${model}_${HELD}/results/test_metrics.json" >/dev/null; then
    echo "=== SKIP  $model (completed in $OUT_DIR; SKIP_EXISTING=0 to retrain) ==="
    continue
  fi
  echo ""
  echo "=== TRAIN $model | $config | train CCPs+ER+F-actin -> test $HELD ==="
  if ! gpu_preflight; then
    FAILED+=("$model (gpu occupied)")
    continue
  fi
  if "$PY" train.py --config "$config" "dataset.held_out_structure=$HELD" "output_dir=$OUT_DIR" "${EXTRA[@]}"; then
    echo "=== DONE  $model ==="
  else
    echo "!!! FAILED $model (suite continues)" >&2
    FAILED+=("$model")
  fi
  cleanup_gpu
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "RUN_MT FINISHED WITH ${#FAILED[@]} FAILURE(S): ${FAILED[*]}"
  echo "Rerun to retry only the failed/missing models (completed runs are skipped)."
  exit 1
fi
echo "RUN_MT COMPLETE. Results: $OUT_DIR/<timestamp>_<model>_${HELD}/results/test_metrics.json"
