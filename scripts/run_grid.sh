#!/usr/bin/env bash
# Grid runner: {model} x {loss variant} x {training variant} ablation matrix.
#   * every combo trains from its own base config into its own output dir
#     (runs_grid/<model>__<loss>__<train>) - no cross-combo collisions
#   * each run is an isolated process; a failed run NEVER kills the grid
#   * completed runs are skipped on rerun (resume) - results/test_metrics.json
#     is the completion marker; set SKIP_EXISTING=0 to force a full rerun
#   * per-GPU flock single-instance guard + stale train.py cleanup, identical
#     to scripts/run_loso.sh
#
# Usage:
#   export BIOSR_ROOT=/abs/path/to/BioSR
#   bash scripts/run_grid.sh models     # {dfcan,rcan,nafnet,swinir,mambair,wavemixsr,restormer,flow_matching} x base
#   bash scripts/run_grid.sh losses     # dfcan x {base,dc,ffl,hess,gradvar}     (default)
#   bash scripts/run_grid.sh training   # dfcan x {base,ema,sam,ood,sel_f1}
#   bash scripts/run_grid.sh flow       # dfcan (regression) vs flow_matching, base loss/training
#   bash scripts/run_grid.sh full       # full cross-product (models x losses x trainings)
#
# Env knobs (same as run_loso.sh):
#   BIOSR_ROOT=/abs/path/to/BioSR    required data root
#   GPU=1                            physical GPU index (default 0), exported as
#                                    CUDA_VISIBLE_DEVICES
#   KILL_STALE=0                     keep leftover train.py on this GPU
#                                    (default 1 = terminate them first)
#   PYTHON=/path/to/python           interpreter (default: python)
#   EXTRA="training.epochs=100 ..."  extra overrides applied to every run
#   SKIP_EXISTING=0                  force full rerun (default 1 = resume)
#   LOCK_FILE=/path                  override the per-GPU lock path
#
# Long runs: nohup bash scripts/run_grid.sh full > grid_full.log 2>&1 &
set -uo pipefail

: "${BIOSR_ROOT:?export BIOSR_ROOT=/abs/path/to/BioSR first}"
export BIOSR_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MODE="${1:-losses}"
PY="${PYTHON:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA=(${EXTRA:-})
FAILED=()
HELDS=(CCPs ER Microtubules F-actin)
GRID_ROOT="runs_grid"

# --- per-GPU single-instance + stale-process guard ---------------------------
# Only one run_grid.sh may drive a given GPU: a second launch on the same GPU
# refuses instead of stacking a parallel training run. Leftover train.py
# processes on THIS GPU are terminated first (set KILL_STALE=0 to disable).
GPU="${GPU:-0}"
KILL_STALE="${KILL_STALE:-1}"
LOCK_FILE="${LOCK_FILE:-/tmp/restore_grid_gpu${GPU}.lock}"
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
    echo "!!! another run_grid.sh is already active on GPU$GPU (lock: $LOCK_FILE)." >&2
    echo "    Use GPU=<other> for the second GPU, or wait for the current grid." >&2
    exit 1
  fi
}

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
  # Fail fast (before burning a startup cycle) when the selected GPU is mostly
  # occupied by another process. Threshold overridable via MIN_GPU_FREE_MIB.
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local free_mib
  free_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if (( free_mib < ${MIN_GPU_FREE_MIB:-6000} )); then
    echo "!!! GPU$GPU has only ${free_mib} MiB free (< ${MIN_GPU_FREE_MIB:-6000}); another process is likely occupying it:" >&2
    nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    echo "    Free the GPU first (or set MIN_GPU_FREE_MIB=... lower) and rerun." >&2
    return 1
  fi
  return 0
}

# --- grid axes ----------------------------------------------------------------
# Model axis: one base config per model; model-specific keys never cross configs.
MODELS=(dfcan rcan nafnet swinir mambair wavemixsr restormer flow_matching)
# Loss axis: overrides on the base config (valid keys in build_loss).
LOSSES=(base dc ffl hess gradvar)
# Training axis: overrides (valid keys in train.py).
TRAININGS=(base ema sam ood sel_f1)

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

loss_overrides() { # sets LOSS_ARGS
  case "$1" in
    base)    LOSS_ARGS=() ;;
    dc)      LOSS_ARGS=(loss.data_consistency=0.1) ;;
    ffl)     LOSS_ARGS=(loss.focal_frequency=0.05) ;;
    hess)    LOSS_ARGS=(loss.hessian=0.05) ;;
    gradvar) LOSS_ARGS=(loss.gradient_variance=0.05) ;;
    *) echo "unknown loss variant: $1" >&2; exit 1 ;;
  esac
}

training_overrides() { # sets TRAIN_ARGS
  case "$1" in
    base)   TRAIN_ARGS=() ;;
    ema)    TRAIN_ARGS=(training.ema=true) ;;
    sam)    TRAIN_ARGS=(training.optimizer=sam) ;;
    ood)    TRAIN_ARGS=(augmentation.morphology_ood=true) ;;
    sel_f1) TRAIN_ARGS=(training.select_metric=val_f1) ;;
    *) echo "unknown training variant: $1" >&2; exit 1 ;;
  esac
}

# --- build the combo plan for the requested mode ------------------------------
COMBOS=() # entries: "model:loss:training"
case "$MODE" in
  models)
    for m in "${MODELS[@]}"; do COMBOS+=("$m:base:base"); done
    ;;
  losses)
    for l in "${LOSSES[@]}"; do COMBOS+=("dfcan:$l:base"); done
    ;;
  training)
    for t in "${TRAININGS[@]}"; do COMBOS+=("dfcan:base:$t"); done
    ;;
  flow)
    # regression vs generative on the same protocol
    COMBOS+=("dfcan:base:base")
    COMBOS+=("flow_matching:base:base")
    ;;
  full)
    for m in "${MODELS[@]}"; do
      for l in "${LOSSES[@]}"; do
        for t in "${TRAININGS[@]}"; do COMBOS+=("$m:$l:$t"); done
      done
    done
    ;;
  *)
    echo "unknown mode: $MODE (use models | losses | training | flow | full)" >&2
    exit 1
    ;;
esac
N_RUNS=$(( ${#COMBOS[@]} * ${#HELDS[@]} ))

echo "=== run_grid.sh | mode=$MODE | GPU=$GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ==="
echo "=== plan: ${#COMBOS[@]} combo(s) x ${#HELDS[@]} folds = $N_RUNS training run(s) -> $GRID_ROOT/<model>__<loss>__<train>/ ==="
if [[ "$MODE" == full ]]; then
  echo "!!! mode=full trains the entire cross-product ($N_RUNS runs); expect a multi-day cost." >&2
fi

train() { # <config> <held_out> [overrides...]
  local config="$1" held="$2" model out_dir="runs" arg label
  model="$(basename "$config" .yaml)"
  shift 2
  for arg in "$@"; do
    case "$arg" in output_dir=*) out_dir="${arg#output_dir=}" ;; esac
  done
  label="$(basename "$out_dir")/$held"
  if [[ "$SKIP_EXISTING" == 1 ]] && compgen -G "$out_dir/*_${model}_${held}/results/test_metrics.json" >/dev/null; then
    echo "=== SKIP  $label (completed run in $out_dir; SKIP_EXISTING=0 to retrain) ==="
    return 0
  fi
  echo ""
  echo "=== TRAIN $label | overrides: $* ==="
  if ! gpu_preflight; then
    FAILED+=("$label (gpu occupied)")
    return 0
  fi
  if "$PY" train.py --config "$config" "dataset.held_out_structure=$held" "$@" "${EXTRA[@]}"; then
    echo "=== DONE  $label ==="
  else
    echo "!!! FAILED $label (grid continues)" >&2
    FAILED+=("$label")
  fi
  cleanup_gpu
}

acquire_lock
if [[ "$KILL_STALE" == 1 ]]; then
  kill_stale_training
fi

for combo in "${COMBOS[@]}"; do
  IFS=: read -r model loss tvar <<< "$combo"
  config="$(model_config "$model")"
  loss_overrides "$loss"
  training_overrides "$tvar"
  grid_dir="$GRID_ROOT/${model}__${loss}__${tvar}"
  for held in "${HELDS[@]}"; do
    train "$config" "$held" "${LOSS_ARGS[@]}" "${TRAIN_ARGS[@]}" "output_dir=$grid_dir"
  done
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "GRID FINISHED WITH ${#FAILED[@]} FAILURE(S):"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  echo "Rerun 'bash scripts/run_grid.sh $MODE' to retry only failed/missing runs (completed runs are skipped)."
  exit 1
fi
echo "GRID COMPLETE. Results: $GRID_ROOT/<model>__<loss>__<train>/<timestamp>_<model>_<heldout>/results/test_metrics.json"
