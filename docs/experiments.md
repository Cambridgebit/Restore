# Experiments: models, losses, training methods

Current inventory of the framework. All numbers are from the shipped configs and
were measured with:

```bash
PYTHONPATH=$PWD ../SuperRestore/.venv/bin/python - <<'PY'
import yaml
from pathlib import Path
from src.models import build_model

for name in ["dfcan", "rcan", "nafnet", "swinir", "mambair", "wavemixsr", "restormer", "flowmatching"]:
    cfg = yaml.safe_load((Path("configs") / f"{name}.yaml").read_text())
    count = sum(p.numel() for p in build_model(cfg["model"]).parameters())
    print(f"{name:14s} {count:>12,}")
PY
```

Shared contract for every backbone: input `(N, 1, H, W)` float32 in `[0, 1]`,
output `(N, 1, 2H, 2W)`, `scale=2` only, and residual prediction by default
(`SR = U(LR) + R_theta(LR)`, `U` = antialiased bicubic). All are config-driven and
built by `src.models.build_model`.

## 1. Model zoo

| `model.name` | Config | Params | Family / mechanism | Objective |
|---|---|---|---|---|
| `bicubic` | `configs/bicubic.yaml` | 0 | parameter-free bicubic x2 | reference floor |
| `nafnet` | `configs/nafnet.yaml` | 0.23M | gated convolution (SimpleGate + simplified channel attention) | regression |
| `mambair` | `configs/mambair.yaml` | 0.68M | state space, 2D selective scan in 4 directions (VSS) | regression |
| `swinir` | `configs/swinir.yaml` | 0.90M | shifted-window self-attention | regression |
| `restormer` | `configs/restormer.yaml` | 1.09M | U-Net with channel-transposed attention (MDTA) + gated FFN | regression |
| `wavemixsr` | `configs/wavemixsr.yaml` | 1.65M | Haar-DWT token mixing | regression |
| `dfcan` | `configs/dfcan.yaml` | 3.20M | Fourier channel attention (main backbone) | regression |
| `flow_matching` | `configs/flowmatching.yaml` | 7.58M | conditional flow matching, time-conditioned U-Net velocity field | **generative** |
| `rcan` | `configs/rcan.yaml` | 15.27M | channel attention (baseline) | regression |

`mambair` uses a pure-PyTorch selective scan and needs no `mamba-ssm`; install
`mamba-ssm`/`causal-conv1d` on the server to enable the fused CUDA kernel and a
large speedup.

## 2. Loss components (`src/losses`, `COMPONENT_DEFAULTS`)

The loss is a weighted sum; only components with weight `> 0` are computed.

| Component | Default weight | Role | Needs `lr` |
|---|---|---|---|
| `charbonnier` | 1.0 | robust pixel fidelity | no |
| `ssim` | 0.1 | local structure (`1 - SSIM`) | no |
| `gradient` | 0.1 | central-difference L1 (filaments/edges) | no |
| `fourier` | 0.0 | amplitude-spectrum L1 | no |
| `data_consistency` | 0.0 | `||downsample(SR) - LR||_1` (anti-hallucination) | **yes** |
| `residual` | 0.0 | `||SR - bicubic_up(LR)||_1` | **yes** |
| `focal_frequency` | 0.0 | focal frequency loss (thin structures) | no |
| `gradient_variance` | 0.0 | gradient-variance loss (anti-blur) | no |
| `hessian` | 0.0 | Hessian structureness (curvilinear continuity) | no |

Extra parameters: `charbonnier_eps` (1e-3), `ssim_window` (11). Grid loss variants:
`base`, `dc`, `ffl`, `hess`, `gradvar`.

`flow_matching` ignores the loss weights: it trains on a Charbonnier loss over the
velocity target (reusing `charbonnier_eps`).

## 3. Training methods

| Dimension | Options | Default | Where it acts |
|---|---|---|---|
| objective | regression / flow-matching | from `model.name` | training loop + sampling |
| `training.optimizer` | `adam` / `sam` | adam | optimizer (`sam_rho`: 0.05) |
| `training.select_metric` | `val_psnr` / `val_ssim` / `val_f1` | val_psnr | best-checkpoint selection |
| `training.ema` | false / true | false | weight EMA (`ema_decay`: 0.999) |
| `eval.tta` | false / true | false | **inference-time** 8-view self-ensemble (`tta_mode`: median/mean) |
| `augmentation.morphology_ood` | false / true | false | **data-side** synthetic morphology domain |
| `training.flow_steps` / `flow_sigma` | int / float | 20 / 1.0 | flow sampling only |

### 3.1 `training.ema` — exponential moving average of weights

Keeps a shadow copy of the model (`src/utils/ema.py`) whose parameters follow
`ema = decay * ema + (1 - decay) * param`, with an early ramp
`effective_decay = min(decay, (1 + step) / (10 + step))`. It is updated after every
optimizer step; buffers are copied verbatim (never averaged).

When enabled, the **EMA weights are the evaluated model**: validation, periodic
visualization, and the saved `best.pt`/`last.pt` all use `ema.module` (and the
final test loads `best.pt`). Effect: smoother weights, less prediction noise, and
usually more stable generalization at almost no training cost (one extra weight
copy, no extra forward/backward). None of the current backbones use BatchNorm, so
there is no BN running-stat recomputation issue.

### 3.2 `eval.tta` — test-time geometric self-ensemble

At final evaluation (`evaluate_split`), the LR image is passed through the model
under the 8 dihedral views (4 rotations x horizontal flip); each SR prediction is
mapped back to the original orientation and the views are aggregated —
`median` (default, preferred for thin structures) or `mean` (blurs them).
See `src/utils/predict_tiled_tta`.

Effect: cancels orientation-dependent artifacts and reduces variance at ~8x
inference cost, with no retraining. It is **inference-only**: it does not affect
training or training-time validation / checkpoint selection. It is skipped for
`flow_matching` (flow samples via `sample()` instead). It helps only if the model
is approximately equivariant and the acquisition/downsampling is symmetric.

### 3.3 Other training methods

- `training.optimizer=sam` — Sharpness-Aware Minimization: a two-step update that
  seeks flat minima (better OOD robustness). See `src/utils/sam.py`.
- `training.select_metric=val_f1` — select the checkpoint by structural F1 rather
  than PSNR, aligning model selection with fidelity.
- `augmentation.morphology_ood=true` — label-free synthetic morphology domain
  (filaments/curves/rings/dots/Voronoi/fractal, `src/data/morphology.py`) run
  through the same forward model, added via `ConcatDataset` with balanced sampling.
- `flow_matching` objective — conditional flow matching: `x_t = (1-t)*noise +
  t*x1`, regress `x1 - noise`; sample by Euler integration of the ODE.

## 4. One-click grid (`scripts/run_grid.sh`)

Axes: **models=8** x **losses=5** x **trainings=5**, each over 4 LOSO folds
(CCPs / ER / Microtubules / F-actin).

| Mode | Combinations | Runs |
|---|---|---|
| `models` | 8 backbones x base loss x base training | 32 |
| `losses` | dfcan x 5 loss variants | 20 |
| `training` | dfcan x 5 training variants | 20 |
| `flow` | dfcan (regression) vs flow_matching | 8 |
| `full` | 8 x 5 x 5 = 200 combos | 800 |

## 5. Caveats

- **Parameters are not matched** (0.23M-15.27M; flow is also multi-step). Cross-
  backbone `models`/`full` comparisons mix architecture with capacity; report
  params and/or add a capacity-matched setting before drawing architecture
  conclusions.
- **`flow_matching` ignores the loss axis** (velocity loss instead) and the TTA
  flag; the grid still emits those combos as separate (redundant) runs.
- `data_consistency`/`residual` require the LR input (passed by `train.py`).
- `eval.tta` and `morphology_ood` do not change the training objective.
- Only `scale=2` is supported by all backbones.
