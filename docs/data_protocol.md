# BioSR 数据协议

BioSR 用于后续 SR（超分辨率）模型训练、验证和测试。协议只约定数据发现、样本配对和训练所需字段，不改变原始 TIFF 文件。

## 1. 数据位置

默认数据根目录：

```text
/home/user/Documents/Code_yjq/SuperRestore/Robust_Restore/datasets/BioSR
```

可用环境变量 `BIOSR_ROOT` 覆盖。

支持以下目录结构：

```text
# Flat：Microtubules、CCPs、F-actin、F-actin_Nonlinear
BioSR/<structure>/Cell_XXX/
  RawSIMData_level_XX.tif   # 低分辨率输入
  RawSIMData_gt.tif         # 可选：同分辨率参考
  SIM_gt*.tif               # 高分辨率 SR 目标

# ER
BioSR/ER/Cell_XXX/
  RawSIMData/RawSIMData_level_XX.tif
  RawGTSIMData/RawGTSIMData_level_XX.tif  # 可选
  GTSIM/GTSIM_level_XX.tif                # 高分辨率 SR 目标
```

`Myosin-IIA_MRC` 默认不纳入训练，因为缺少配对的高分辨率 GT。加载器应忽略检查点目录、`*_n2v2.tif` 和其他派生临时文件。

## 2. 统一样本

每个 `structure / cell / level` 组成一条样本：

```json
{
  "id": "Microtubules/Cell_001/level_05",
  "structure": "Microtubules",
  "cell": "Cell_001",
  "level": 5,
  "layout": "flat",
  "input": "Microtubules/Cell_001/RawSIMData_level_05.tif",
  "hr_gt": "Microtubules/Cell_001/SIM_gt.tif",
  "wf_gt": "Microtubules/Cell_001/RawSIMData_gt.tif",
  "scale_hr": 2.0,
  "split": "train"
}
```

字段约定：

- `input`：SR 模型输入，即指定 level 的低分辨率宽场图像。
- `hr_gt`：SR 训练目标，必须是与 `input` 配对的高分辨率 SIM 真值。
- `wf_gt`：可选的同分辨率参考，仅在去噪任务中使用。
- `scale_hr`：高分辨率目标相对输入的倍率；通常为 `2.0`，F-actin_Nonlinear 可为 `3.0`。
- `wf_gt` 或 `hr_gt` 不存在时写为 `null`；SR 训练样本必须有 `hr_gt`。

## 3. SR 训练约定

默认任务为：

```text
input  = RawSIMData_level_XX.tif
target = hr_gt
```

训练前统一转换为 `float32` 张量 `N×1×H×W`。默认将每张图像归一化到 `[0, 1]`；若实验使用其他归一化方式，必须写入实验配置。

推荐：

- 按 `cell` 划分 `train / val / test`，同一细胞不得跨集合。
- 一个 split 可以包含多个 level，但实验配置必须记录实际使用的 level。
- 训练、验证和测试使用同一套样本发现规则，避免手写不一致的文件匹配逻辑。

`BioSRSample.to_dict()` 的输出可直接写入 JSON 或 JSONL manifest。


