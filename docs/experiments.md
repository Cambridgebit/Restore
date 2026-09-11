# 实验说明：模型、损失、训练方式

本文档汇总当前框架的全部可选项。所有参数量均来自仓库自带配置，测量命令：

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

所有骨干共享同一契约：输入 `(N, 1, H, W)` float32 ∈ [0, 1]，输出
`(N, 1, 2H, 2W)`，只支持 `scale=2`，且默认启用残差预测
（`SR = U(LR) + R_theta(LR)`，`U` 为抗锯齿双三次上采样）。全部由 config 驱动，
通过 `src.models.build_model` 构建。

## 1. 模型库

| `model.name` | 配置 | 参数量 | 家族 / 核心机制 | 目标 |
|---|---|---|---|---|
| `bicubic` | `configs/bicubic.yaml` | 0 | 无可学习参数，双三次 ×2 | 参照下限 |
| `nafnet` | `configs/nafnet.yaml` | 0.23M | 门控卷积（SimpleGate + 简化通道注意力） | 回归 |
| `mambair` | `configs/mambair.yaml` | 0.68M | 状态空间，四方向 2D 选择性扫描（VSS） | 回归 |
| `swinir` | `configs/swinir.yaml` | 0.90M | 移位窗口自注意力 | 回归 |
| `restormer` | `configs/restormer.yaml` | 1.09M | U-Net + 通道转置注意力（MDTA）+ 门控 FFN | 回归 |
| `wavemixsr` | `configs/wavemixsr.yaml` | 1.65M | Haar 小波 token mixing | 回归 |
| `dfcan` | `configs/dfcan.yaml` | 3.20M | 傅里叶通道注意力（主骨干） | 回归 |
| `flow_matching` | `configs/flowmatching.yaml` | 7.58M | 条件 Flow Matching，时间条件 U-Net 速度场 | **生成式** |
| `rcan` | `configs/rcan.yaml` | 15.27M | 通道注意力（baseline） | 回归 |

`mambair` 使用纯 PyTorch 选择性扫描，**不依赖 `mamba-ssm`**；在服务器上安装
`mamba-ssm` / `causal-conv1d` 可启用融合 CUDA 内核，显著提速。

## 2. 损失分量（`src/losses`，`COMPONENT_DEFAULTS`）

损失是加权和，**只计算权重 > 0 的分量**。

| 分量 | 默认权重 | 作用 | 需要 `lr` |
|---|---|---|---|
| `charbonnier` | 1.0 | 鲁棒像素保真 | 否 |
| `ssim` | 0.1 | 局部结构（`1 - SSIM`） | 否 |
| `gradient` | 0.1 | 中心差分 L1（细丝 / 边缘） | 否 |
| `fourier` | 0.0 | 幅度谱 L1 | 否 |
| `data_consistency` | 0.0 | `||downsample(SR) - LR||_1`（抗幻觉） | **是** |
| `residual` | 0.0 | `||SR - bicubic_up(LR)||_1` | **是** |
| `focal_frequency` | 0.0 | 焦点频域损失（细结构） | 否 |
| `gradient_variance` | 0.0 | 梯度方差损失（抗发虚） | 否 |
| `hessian` | 0.0 | Hessian 结构度（曲率 / 细丝连续性） | 否 |

附加参数：`charbonnier_eps`（1e-3）、`ssim_window`（11）。网格可选损失变体：
`base`、`dc`、`ffl`、`hess`、`gradvar`。

`flow_matching` **不使用损失权重**：它在速度场上使用 Charbonnier 损失
（复用 `charbonnier_eps`）。

## 3. 训练方式

| 维度 | 可选值 | 默认 | 生效位置 |
|---|---|---|---|
| objective（目标） | 回归 / flow-matching | 由 `model.name` 决定 | 训练循环 + 采样 |
| `training.optimizer` | `adam` / `sam` | adam | 优化器（`sam_rho`：0.05） |
| `training.select_metric` | `val_psnr` / `val_ssim` / `val_f1` | val_psnr | 最佳 checkpoint 选择 |
| `training.ema` | false / true | false | 权重 EMA（`ema_decay`：0.999） |
| `eval.tta` | false / true | false | **推理时** 8 视角自集成（`tta_mode`：median/mean） |
| `augmentation.morphology_ood` | false / true | false | **数据侧** 合成形态域 |
| `training.flow_steps` / `flow_sigma` | int / float | 20 / 1.0 | 仅 flow 采样 |

### 3.1 `training.ema` —— 权重的指数滑动平均

保持一份模型的影子副本（`src/utils/ema.py`），参数按
`ema = decay * ema + (1 - decay) * param` 更新，前期带升温：
`effective_decay = min(decay, (1 + step) / (10 + step))`。每个优化器 step 后更新；
buffer 直接复制（不做平均）。

开启后，**评估用的就是 EMA 权重**：验证、周期性可视化、保存的
`best.pt` / `last.pt` 都使用 `ema.module`（最终测试加载 `best.pt`）。效果：权重更
平滑、预测噪声更小，通常泛化更稳；代价几乎为零（多一份权重副本，无额外前向/反向）。
当前骨干都不含 BatchNorm，因此没有 BN 统计量重构的问题。

### 3.2 `eval.tta` —— 推理时的几何自集成

在最终评估（`evaluate_split`）时，把 LR 在 8 个二面体视角（4 个旋转 × 水平翻转）下
分别过模型，每个 SR 结果映射回原方向后聚合——`median`（默认，对细结构更友好）或
`mean`（会糊细结构）。实现见 `src/utils/predict_tiled_tta`。

效果：抵消方向相关伪影、降低方差，推理约 8× 成本，**无需重训**。它是**纯推理**项：
不影响训练，也不影响训练期的验证 / checkpoint 选择；对 `flow_matching` 会跳过
（flow 走 `sample()`）。仅在模型近似等变、且采集/下采样近似对称时才有效。

### 3.3 其他训练方式

- `training.optimizer=sam` —— 锐度感知最小化（SAM）：两步更新寻找平坦极小值，改善
  OOD 鲁棒性。见 `src/utils/sam.py`。
- `training.select_metric=val_f1` —— 用结构 F1 而非 PSNR 选 checkpoint，使模型选择
  与"保真优先"对齐。
- `augmentation.morphology_ood=true` —— 无标签合成形态域（丝状/曲线/环/点/Voronoi/
  分形，`src/data/morphology.py`），经同一前向模型生成 LR，用 `ConcatDataset` 加入并按
  结构均衡采样。
- `flow_matching` 目标 —— 条件流匹配：`x_t = (1-t)*noise + t*x1`，回归 `x1 - noise`；
  推理用 Euler 积分求解 ODE。

## 4. 一键网格（`scripts/run_grid.sh`）

轴：**模型=8** × **损失=5** × **训练方式=5**，每个组合跑 4 个 LOSO 折
（CCPs / ER / Microtubules / F-actin）。

| 模式 | 组合 | 运行数 |
|---|---|---|
| `models` | 8 个骨干 × base 损失 × base 训练 | 32 |
| `losses` | dfcan × 5 个损失变体 | 20 |
| `training` | dfcan × 5 个训练变体 | 20 |
| `flow` | dfcan（回归）vs flow_matching | 8 |
| `full` | 8 × 5 × 5 = 200 个组合 | 800 |

## 5. 注意事项

- **参数量未对齐**（0.23M–15.27M；flow 还是多步采样）。`models` / `full` 的跨骨干
  对比会混入容量效应；下架构结论前应先报告参数量，或增加参数对齐档。
- **`flow_matching` 忽略损失轴**（改用速度损失）与 TTA 开关；网格仍会生成这些组合
  （属于重复运行）。
- `data_consistency` / `residual` 需要 LR 输入（由 `train.py` 传入）。
- `eval.tta` 与 `morphology_ood` 不改变训练目标。
- 所有骨干只支持 `scale=2`。
