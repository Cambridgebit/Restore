#!/usr/bin/env bash
# High-SNR training arms + cross-SNR migration evaluation + decision table.
#
#   arm "mixed"          all levels (baseline; reused from runs/, skipped when complete)
#   arm "highsnr"        signal_levels=[6]                      -> runs_highsnr/
#   arm "highsnr_noisy"  signal_levels=[6] + noise aug (0.05)   -> runs_highsnr_noisy/
#
# After training, every checkpoint is RE-EVALUATED with ALL levels
# (dataset.signal_levels=all) to build the train-SNR -> inference-SNR transfer
# table. Per run this produces:
#   results/test_metrics_trainlevels.json   (levels used in training, preserved)
#   results/test_metrics_all_levels.json    (full migration table)
#   results/test_metrics.json               (restored marker for resume logic)
#
# Usage:
#   export BIOSR_ROOT=/abs/path/to/BioSR
#   bash scripts/run_highsnr_arms.sh              # train missing runs + eval + table
#   SKIP_EVAL=1 bash scripts/run_highsnr_arms.sh  # training only
#   SKIP_TRAIN=1 bash scripts/run_highsnr_arms.sh # eval + table only
# Long runs: nohup bash scripts/run_highsnr_arms.sh > highsnr_arms.log 2>&1 &
set -uo pipefail

: "${BIOSR_ROOT:?export BIOSR_ROOT=/abs/path/to/BioSR first}"
export BIOSR_ROOT
PY="${PYTHON:-python}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FAILED=()

train_arm() { # <output_dir> [extra overrides...]
  if [[ "$SKIP_TRAIN" == 1 ]]; then
    echo "=== SKIP training (SKIP_TRAIN=1) ==="
    return 0
  fi
  echo ""
  echo "########## ARM: output_dir=$1 ${*:2} ##########"
  EXTRA="dataset.signal_levels=[6] output_dir=$1 ${*:2}" bash scripts/run_loso.sh core || FAILED+=("train/$1")
}

migrate_eval() { # <run_dir>
  local run_dir="$1" results
  results="$run_dir/results"
  if [[ ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "=== SKIP eval $run_dir (no checkpoint) ==="
    return 0
  fi
  if [[ -f "$results/test_metrics_all_levels.json" ]]; then
    echo "=== SKIP eval $run_dir (already done) ==="
    return 0
  fi
  echo ""
  echo "########## MIGRATION EVAL: $run_dir (all levels) ##########"
  cp "$results/test_metrics.json" "$results/test_metrics_trainlevels.json"
  if "$PY" evaluate.py --config "$run_dir/config.yaml" \
      --checkpoint "$run_dir/checkpoints/best.pt" dataset.signal_levels=all; then
    cp "$results/test_metrics.json" "$results/test_metrics_all_levels.json"
    cp "$results/test_metrics_trainlevels.json" "$results/test_metrics.json" # restore resume marker
    echo "=== DONE  $run_dir -> results/test_metrics_all_levels.json ==="
  else
    echo "!!! FAILED migration eval $run_dir (suite continues)" >&2
    FAILED+=("eval/$run_dir")
  fi
}

# ---- 1) training arms (resume-safe; completed runs are skipped) -------------
train_arm runs_highsnr
train_arm runs_highsnr_noisy augmentation.noise_max_sigma=0.05

# ---- 2) migration evaluation over every arm checkpoint ----------------------
if [[ "$SKIP_EVAL" == 1 ]]; then
  echo "=== SKIP migration eval (SKIP_EVAL=1) ==="
else
  for arm_root in runs_highsnr runs_highsnr_noisy; do
    for run_dir in "$arm_root"/*/; do
      [[ -d "$run_dir" ]] && migrate_eval "$run_dir"
    done
  done
fi

# ---- 3) decision table -------------------------------------------------------
"$PY" - <<'PYEOF'
import json
from pathlib import Path

ARMS = [("mixed", Path("runs")), ("highsnr", Path("runs_highsnr")), ("highsnr_noisy", Path("runs_highsnr_noisy"))]
MODELS = ("dfcan", "rcan")


def load(run_dir: Path):
    for name in ("test_metrics_all_levels.json", "test_metrics.json"):
        path = run_dir / "results" / name
        if path.is_file():
            return json.loads(path.read_text())
    return None


rows: dict = {}
for arm, root in ARMS:
    if not root.is_dir():
        continue
    for d in sorted(root.iterdir()):
        parts = d.name.split("_")
        if len(parts) < 3 or parts[1] not in MODELS:
            continue
        m = load(d)
        if m is None:
            continue
        model, held = parts[1], "_".join(parts[2:])
        for level, group in m.get("per_level", {}).items():
            entry = rows.setdefault((model, held, level), {})
            entry[arm] = (group["psnr"]["mean"], group["structural_precision"]["mean"])

for model in MODELS:
    helds = sorted({h for (mo, h, _) in rows if mo == model})
    for held in helds:
        levels = sorted({lv for (mo, h, lv) in rows if mo == model and h == held}, key=int)
        print(f"\n=== {model} | held-out: {held}  (PSNR | precision) ===")
        print(f"{'level':>5} | " + " | ".join(f"{arm:>16}" for arm, _ in ARMS))
        for level in levels:
            cells = []
            for arm, _ in ARMS:
                v = rows.get((model, held, level), {}).get(arm)
                cells.append(f"{v[0]:6.2f} | {v[1]:.3f}" if v else f"{'--':>16}")
            print(f"{level:>5} | " + " | ".join(cells))
print("\n(mixed = all-levels training baseline from runs/; read columns per level to judge")
print(" whether high-SNR-only training transfers to low SNR and whether precision collapses.)")
PYEOF

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FINISHED WITH ${#FAILED[@]} FAILURE(S): ${FAILED[*]}"
  echo "Rerun the same command to retry missing parts (completed parts are skipped)."
  exit 1
fi
echo "HIGH-SNR ARMS COMPLETE."
