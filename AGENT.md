# AGENTS.md

## 1. Project Goal

本项目研究：

> **Structure-Generalizable Microscopy Super-Resolution**

当前阶段只关注 **BioSR 数据集内部的跨细胞结构泛化**：

```text
Train on 3 structures
        ↓
Test on the unseen 4th structure
```

核心目标不是生成更锐利的图像，而是：

> **恢复训练中未见结构时，尽量减少 structural hallucination。**

优先级：

```text
Structural Fidelity
> Cross-Structure Generalization
> PSNR / Visual Sharpness
```

---

## 2. Current Research Scope

当前固定：

* Dataset: `BioSR`
* Task: Single-Image Super-Resolution
* Scale: `×2 Linear-SIM`
* Structures:

  * CCP
  * ER
  * Microtubules (MT)
  * F-actin
* Main backbone: `DFCAN`
* Baseline: `RCAN`
* Framework: `PyTorch`

除非明确要求，否则不要加入：

* GAN
* Diffusion
* perceptual loss
* text/structure conditioning
* ×3 nonlinear-SIM
* external datasets

保持问题单一：

> **只研究 morphology / structure shift。**

---

# 3. Experimental Protocol

必须使用：

## Leave-One-Structure-Out

四个实验：

```text
Fold 1
Train: ER + MT + F-actin
Test : CCP

Fold 2
Train: CCP + MT + F-actin
Test : ER

Fold 3
Train: CCP + ER + F-actin
Test : MT

Fold 4
Train: CCP + ER + MT
Test : F-actin
```

最终报告：

```text
每个 unseen structure 的结果
+
4-fold mean ± std
```

禁止将四种结构混合后随机切分作为主要泛化实验。

---

# 4. Data Leakage Rules

这是最高优先级规则。

### 必须

先按照：

```text
original FOV / source GT
```

划分数据，再生成 patch。

同一个 GT / FOV 对应的：

```text
different signal levels
different LR realizations
different patches
```

必须属于同一个 split。

### 禁止

```text
image
 ↓
crop patches
 ↓
random train/test split
```

这会造成严重数据泄漏。

被留出的 test structure：

> **不得以任何形式进入训练。**

包括：

* patch
* augmentation
* pretraining
* validation
* normalization statistics

Structure label 只允许用于：

```text
split
sampling
evaluation
logging
```

**禁止作为模型输入。**

---

# 5. Data Sampling

训练采用 structure-balanced sampling：

```text
sample structure
      ↓
sample FOV
      ↓
sample signal level
      ↓
random crop
```

确保训练的三类结构贡献大致相等：

$$
P(S_1)\approx P(S_2)\approx P(S_3)
$$

不要让 patch 数量最多的结构主导训练。

---

# 6. Backbone

## Main Model: DFCAN

使用 DFCAN 作为主要 backbone。

基本结构：

```text
LR
 ↓
Shallow Conv
 ↓
DFCAN Residual Groups
 ↓
Fourier Channel Attention
 ↓
Upsampling ×2
 ↓
SR Residual
```

推荐使用 residual prediction：

```text
LR ─────────→ Bicubic ×2 ───────┐
                                 +
LR → DFCAN → HR Residual ────────┘
                 ↓
                SR
```

即：

$$
SR=U(LR)+R_\theta(LR)
$$

网络重点学习缺失的高频信息，而不是重新生成整幅图像。

---

# 7. Baselines

至少保留：

```text
Bicubic
RCAN
DFCAN
```

不要一开始引入大量 backbone。

研究重点是：

> **generalization strategy**

而不是：

> backbone leaderboard。

后续需要时再加入：

```text
SwinIR / HAT
Wavelet-based SR
Physics-informed SR
```

---

# 8. Loss

默认：

$$
L=
L_{char}
+\lambda_sL_{SSIM}
+\lambda_gL_{grad}
$$

其中：

### Pixel Fidelity

```text
Charbonnier Loss
```

保证基本重建准确性。

### Structural Loss

```text
SSIM Loss
```

约束局部结构。

### Gradient Loss

$$
L_{grad}
=
\|\nabla_x SR-\nabla_x GT\|_1
+
\|\nabla_y SR-\nabla_y GT\|_1
$$

重点保护：

* filament continuity
* edges
* thin structures
* morphology

---

## Optional Fourier Loss

后续 ablation 可以加入：

$$
L_{FFT}
=
\|
|\mathcal F(SR)|
-
|\mathcal F(GT)|
\|_1
$$

使用小权重。

原则：

```text
Gradient constraint
>
Fourier constraint
```

因为频率正确不代表空间结构位置正确。

所有 loss 权重必须写入 config，不允许硬编码。

---

# 9. Data Augmentation

Baseline 仅使用：

```text
random crop
horizontal / vertical flip
90° rotation
intensity scaling
mild noise augmentation
```

首先建立可靠 baseline。

之后单独实验：

## Morphology-OOD Augmentation

可加入：

```text
random filament
random curves
random branches
random rings
random dots
Voronoi
fractal structures
```

目标不是模拟某一种细胞器，而是扩大：

$$
P(\text{possible morphology})
$$

降低模型对训练结构模板的依赖。

所有 morphology augmentation 必须能够单独关闭，用于 ablation。

---

# 10. Evaluation

基础指标：

```text
PSNR
SSIM / MS-SSIM
ZNCC
FRC / resolution metric
```

但本项目不能只报告 PSNR。

必须加入结构指标。

## Structural Precision

从 SR 与 GT 提取 edge / skeleton：

$$
Precision
=
\frac{|E_{SR}\cap E_{GT}|}
{|E_{SR}|}
$$

低 Precision 表示：

> 模型生成了 GT 中不存在的结构。

## Structural Recall

$$
Recall
=
\frac{|E_{SR}\cap E_{GT}|}
{|E_{GT}|}
$$

低 Recall 表示：

> 真实结构没有成功恢复。

至少报告：

```text
Structural Precision
Structural Recall
F1
```

---

# 11. Evaluation Breakdown

任何实验结果必须分别报告：

```text
per unseen structure
per signal level
overall mean
standard deviation
```

不要只给整体平均结果。

特别关注：

```text
High SNR
vs
Low SNR
```

因为结构 hallucination 通常在信息不足时更明显。

---

# 12. Required Ablations

实验优先按照以下顺序推进：

```text
A. RCAN
   basic loss

B. DFCAN
   basic loss

C. DFCAN
   + multi-structure balanced training

D. DFCAN
   + gradient loss

E. DFCAN
   + Fourier loss

F. DFCAN
   + morphology augmentation
```

每次只改变一个主要变量。

不要同时修改：

```text
backbone
loss
augmentation
sampling
```

否则无法解释性能提升来自哪里。

---

# 13. Code Organization

如果项目没有既有结构，使用：

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── dfcan.yaml
│   └── rcan.yaml
├── src/
│   ├── data/
│   │   ├── biosr.py
│   │   └── splits.py
│   ├── models/
│   │   ├── dfcan.py
│   │   └── rcan.py
│   ├── losses/
│   │   ├── pixel.py
│   │   ├── structural.py
│   │   └── frequency.py
│   ├── metrics/
│   │   ├── image.py
│   │   └── structure.py
│   └── utils/
├── train.py
├── evaluate.py
└── tests/
```

如果仓库已经存在合理结构：

> 遵循现有结构，不进行无必要重构。

---

# 14. Configuration

实验必须 config-driven。

Config 至少包含：

```yaml
dataset:
  scale: 2
  held_out_structure: ER
  signal_levels: all

model:
  name: dfcan

training:
  seed: 42
  batch_size:
  learning_rate:
  epochs:

loss:
  charbonnier:
  ssim:
  gradient:
  fourier:

augmentation:
  morphology_ood: false
```

不要把实验参数散落在 Python 文件中。

---

# 15. Reproducibility

必须：

* 固定 random seed
* 保存完整 config
* 保存 checkpoint
* 保存 train/val/test split
* 保存 git commit hash（若可用）
* 记录最佳模型选择标准

训练日志至少包括：

```text
train loss
validation PSNR
validation SSIM
learning rate
epoch
```

---

# 16. Tests

至少实现：

### Dataset test

验证：

```text
train/test structure 不重叠
FOV 不重叠
GT family 不跨 split
```

### Model test

验证：

```text
input shape
output shape
×2 scaling
forward/backward
```

### Loss test

验证：

```text
loss finite
gradient finite
```

### Evaluation smoke test

能够使用少量样本完成：

```text
load model
→ inference
→ calculate metrics
→ save results
```

---

# 17. Coding Rules

优先：

```text
simple
modular
reproducible
experiment-friendly
```

不要为了“高级”而引入复杂抽象。

要求：

* functions/classes 有明确职责
* tensor shape 清晰
* 不硬编码绝对路径
* 不复制重复训练逻辑
* dataset / model / loss / metric 解耦
* 新实验通过 config 控制
* 保留已有可工作的代码

---

# 18. Agent Workflow

执行任务前：

1. 阅读 `AGENTS.md`
2. 阅读现有 README/config
3. 检查已有代码，避免重复实现
4. 明确当前实验对应哪一个 ablation

修改代码后：

1. 运行相关 tests
2. 运行最小 smoke training
3. 检查 tensor shape
4. 检查 train/test leakage
5. 检查输出 metrics
6. 总结实际修改内容

不要声称实验有效，除非已有实际结果支持。

---

# 19. Scientific Guardrails

以下原则不可违反：

### Never optimize only for visual sharpness.

### Never use test structures during training.

### Never evaluate structural generalization using random patch splitting.

### Never treat hallucinated high-frequency detail as successful SR.

### Never introduce GAN / diffusion merely because images look sharper.

### Never change the scientific protocol silently.

本项目始终围绕：

$$
\boxed{
\text{Can the model reconstruct a morphology it has never seen?}
}
$$

最终目标：

$$
\boxed{
\text{Seen structures recover well}
+
\text{Unseen structures do not hallucinate}
}
$$

---

# 20. Definition of Done

一个实验只有同时满足以下条件才算完成：

* LOSO split 正确
* 无数据泄漏
* 模型成功训练
* 测试 unseen structure
* PSNR / SSIM 正常输出
* Structural Precision / Recall 正常输出
* 保存配置和 checkpoint
* 结果可以复现
* 与 baseline 在完全相同 split 下比较

**科研可信度优先于模型复杂度。**
