# Restore — Structure-Generalizable Microscopy Super-Resolution

Single-image ×2 SR on BioSR with a **leave-one-structure-out (LOSO)** protocol.
The research question is not "sharper images" but: *can a model reconstruct a
morphology it has never seen — without hallucinating structure?*

Priority: **Structural Fidelity > Cross-Structure Generalization > PSNR**.
See `AGENT.md` for the full research protocol and guardrails.

## Repo layout

```text
.
├── AGENT.md                  # research protocol (read first)
├── README.md
├── configs/
│   ├── dfcan.yaml            # main backbone (fold 1 default)
│   └── rcan.yaml             # baseline
├── docs/
│   └── data_protocol.md      # BioSR discovery/pairing rules + BIOSR_ROOT
├── src/
│   ├── data/
│   │   ├── biosr.py          # sample discovery, dataset, root resolution
│   │   ├── splits.py         # LOSO folds, split building, leakage checks
│   │   └── sampling.py       # structure-balanced sampler
│   ├── models/               # DFCAN, RCAN
│   ├── losses/               # charbonnier + SSIM + gradient (+ optional Fourier)
│   ├── metrics/              # PSNR/SSIM/ZNCC/FRC + structural precision/recall
│   └── utils/
│       ├── config.py         # YAML load/override/save
│       ├── seed.py           # global + DataLoader-worker seeding
│       ├── run.py            # run dirs, checkpoints, JSONL logs, git hash
│       └── inference.py      # tiled inference (predict_tiled)
├── train.py                  # config-driven LOSO training loop
├── evaluate.py               # checkpoint evaluation on the held-out structure
└── tests/
```

## Environment

Reuses the sibling virtualenv — do **not** pip install anything here:

```bash
PY=../SuperRestore/.venv/bin/python
$PY -m pytest                     # all tests
$PY -m pytest tests/test_utils.py -q
```

## Data

`docs/data_protocol.md` defines discovery, sample pairing, and normalization.
Root resolution order: `dataset.root` in the config → `$BIOSR_ROOT` env var →
the protocol default (see `src.data.biosr.DEFAULT_BIOSR_ROOT`). Keep configs
with `root: null` and set the env var instead of hardcoding paths:

```bash
export BIOSR_ROOT=/path/to/BioSR
```

## LOSO folds

Split logic lives in `src.data.splits.LOSO_FOLDS`; each fold trains on three
structures and tests on the unseen fourth:

| Fold   | Held-out (test) structure | Train/val structures      |
|--------|---------------------------|---------------------------|
| fold_1 | CCPs                      | ER, Microtubules, F-actin |
| fold_2 | ER                        | CCPs, Microtubules, F-actin |
| fold_3 | Microtubules              | CCPs, ER, F-actin         |
| fold_4 | F-actin                   | CCPs, ER, Microtubules    |

## Tracking & progress

Each run shows live progress: a tqdm bar per epoch in the terminal, a per-epoch
summary line, and (when `wandb.enabled: true`) real-time curves and SR panels at
`https://wandb.ai/<entity>/Restore`:

- logged per epoch: `train_loss`, loss components, `val_psnr`, `val_ssim`, `lr`
- every `training.vis_every` epochs (default 30, plus epoch 0): tiled inference
  on fixed samples — 2 per signal level of the unseen structure + 1 per seen
  structure — saved as `LR-up | SR | GT` TIFFs under `run_dir/vis/` and pushed
  to wandb as image panels
- final test metrics are attached to the wandb summary

wandb is optional: without the package (or with `wandb.enabled: false`) training
runs unchanged and logs only to stdout + `logs/train_log.jsonl`. Local runs
without a wandb login can set `wandb.mode: offline`.

## Training

Full LOSO suite (4 folds x backbones, plus ablation/bicubic modes) via one script:

```bash
export BIOSR_ROOT=/abs/path/to/BioSR
bash scripts/run_loso.sh core       # 4 folds x {dfcan, rcan}
bash scripts/run_loso.sh ablation   # + gradient-off / + fourier (run core first)
bash scripts/run_loso.sh bicubic    # reference floor, eval only (epochs=0)
```

Single runs:

```bash
# Fold 1 (default in configs/*.yaml)
$PY train.py --config configs/dfcan.yaml
# Folds 2-4 (one override per fold)
$PY train.py --config configs/dfcan.yaml dataset.held_out_structure=ER
$PY train.py --config configs/dfcan.yaml dataset.held_out_structure=Microtubules
$PY train.py --config configs/dfcan.yaml dataset.held_out_structure=F-actin
# RCAN baseline, e.g. fold 2
$PY train.py --config configs/rcan.yaml dataset.held_out_structure=ER training.epochs=10
```

Each run writes to `<output_dir>/<timestamp>_<model>_<held_out>/`:
`config.yaml` (with resolved root), `split.json`, `logs/train_log.jsonl`
(epoch, lr, train loss + unweighted parts, val PSNR/SSIM),
`checkpoints/best.pt` + `checkpoints/last.pt`, and
`results/test_metrics.json`. Best model = highest validation PSNR.

## Evaluation

```bash
$PY evaluate.py --config runs/<run>/config.yaml --checkpoint runs/<run>/checkpoints/best.pt
```

Rebuilds the exact test split from the run config (discover → filter → LOSO
split → leakage check), runs tiled inference, and writes
`results/test_metrics.json` next to the run (or `--checkpoint/../results/` when
no run dir is implied). Metrics: PSNR, SSIM, ZNCC, FRC, structural
precision/recall/F1 — aggregated per signal level and overall.

## Config keys (summary)

| Section       | Keys                                                                 |
|---------------|----------------------------------------------------------------------|
| `dataset`     | `root`, `scale`, `structures`, `held_out_structure`, `signal_levels`, `patch_size`, `normalization` |
| `model`       | `name` (`dfcan`/`rcan`), `nf`, `num_groups`, `num_blocks`, `reduction` (RCAN), `gamma` (DFCAN), `scale`, `residual_prediction` |
| `training`    | `seed`, `batch_size`, `learning_rate`, `epochs`, `num_workers`, `val_fraction`, `val_every`, `weight_decay`, `grad_clip`, `device` |
| `loss`        | `charbonnier`, `ssim`, `gradient`, `fourier` (weights), `charbonnier_eps`, `ssim_window` |
| `augmentation`| `hflip`, `vflip`, `rot90`, `intensity_range`, `noise_max_sigma`, `morphology_ood` |
| `eval`        | `tile`, `overlap`, `edge_percentile`, `tolerance_px`, `frc_threshold`, `ssim_window` |
| top level     | `output_dir` |

All experiment parameters live in configs / CLI `key=value` overrides — nothing
is hardcoded in the training code. Any CLI override is a dotted path parsed as
YAML: `training.learning_rate=1e-4`, `augmentation.hflip=false`.

## Leakage rules (summary)

- Split by **cell/FOV** before any patching; all levels and LR realizations of
  one FOV stay in the same split.
- The held-out structure never appears in train/val in any form: no patches, no
  augmentation, no pretraining, no validation, no normalization statistics.
- The structure label is used only for splitting, balanced sampling, and
  evaluation/logging — **never as a model input**.

## Tests

```bash
$PY -m pytest
```

`tests/test_utils.py` is standalone; the train/eval smoke tests run a tiny
end-to-end training + evaluation on a synthetic BioSR tree (see
`tests/conftest.py`) and verify artifacts, leakage-free splits, and metrics.

## Morphology-OOD augmentation

`augmentation.morphology_ood` is a **reserved toggle** for a later ablation
(AGENT.md §9). Setting it to `true` raises `NotImplementedError`; the baseline
uses only flips, 90° rotation, intensity scaling, and mild noise.

## Reproducibility

Every run saves: fixed seed (config), full resolved config, the exact split
(`split.json`), checkpoints (config + epoch + metrics + git hash embedded), and
per-epoch logs. Same config + same data root → same split and model.
