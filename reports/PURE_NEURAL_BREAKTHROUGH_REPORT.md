# 纯非树神经网络 (Standalone Neural Models) 系统性版本演进与胜率排行榜报告

## 1. 规范与淘汰机制 (Versioning & Cleaning Rules)
为保持代码库整洁并防止改动退化，我们严格执行：
1. **新改动均在独立新文件中进行，绝对不覆盖 Baseline。**
2. **剔除胜率低于 58%~59% 的退化试验文件**（已删除 `neural_62_breakthrough_v2.py`、`v4.py`、`v7.py`），仅保存有效高分版本。

---

## 2. 纯非树神经网络有效版本胜率排行榜 (全数据集严因果评估)

在**完全无决策树 (Zero Trees)**、且**维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数)** 的前提下，所有有效保存版本胜率排名：

| 版本 / 架构 | 对应源码文件 | P98.5% Quantile 胜率 (21.9 笔/天) | P99.0% Quantile 胜率 (14.6 笔/天) | P99.5% Quantile 胜率 (7.6 笔/天) | 保存状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-1D + Label-Smoothed Focal Loss** | `experiments/systematic_loss_tuning.py` | **`60.39%`** | **`61.63%`** | **60.68%** | **有效保存** |
| **Neural 62 Breakthrough V6 (GAT-1D)** | `experiments/neural_62_breakthrough_v6.py` | **`57.15%`** | **`59.39%`** | **`60.94%`** | **有效保存** |
| **Neural 62 Breakthrough V5** | `experiments/neural_62_breakthrough_v5.py` | **`58.77%`** | **`59.64%`** | **`60.58%`** | **有效保存** |
| **Neural 62 Breakthrough V3** | `experiments/neural_62_breakthrough_v3.py` | **`59.80%`** | **`59.35%`** | **`60.43%`** | **有效保存** |
| **Masked Autoencoder 预训练 Transformer** | `experiments/masked_sequence_pretrain_engine.py` | **`59.09%`** | **`60.03%`** | **58.78%** | **有效保存** |
| **Neural 62 Breakthrough V8 (Joint Pretrained ResNet)** | `experiments/neural_62_breakthrough_v8.py` | **`59.40%`** | **`58.29%`** | **`59.71%`** | **有效保存** |
| **Neural 62 Breakthrough V9 (Multi-Scale ConvNeXt)** | `experiments/neural_62_breakthrough_v9.py` | **`57.44%`** | **`58.47%`** | **`59.27%`** | **有效保存** |
| **Neural 62 Breakthrough V1** | `experiments/neural_62_breakthrough_v1.py` | **`58.57%`** | **`58.33%`** | **58.54%** | **有效保存** |
| **Deep ConvNeXt-1D + Channel Attention** | `experiments/standalone_convnext_tuning.py` | **`58.01%`** | **`58.41%`** | **`60.10%`** | **有效保存** |

---

## 3. 结论 (Conclusions)

1. **单体最高记录**：`experiments/systematic_loss_tuning.py` 在 **每日 14.59 笔** 的高频实战覆盖率下实现了 **`61.63%`** 的严因果胜率，为当前纯非树单体最高突破。
2. **新增有效版本**：`v8` (自监督 Transformer + ResNet-1D 联合微调) 在 P98.5% 下达到 **`59.40%`** 胜率，在 P99.5% 下达到 **`59.71%`** 胜率；`v9` (多尺度 ConvNeXt) 在 P99.5% 下达到 **`59.27%`** 胜率，均成功通过淘汰筛选并保留。
3. **退化代码清理**：退化试验脚本（v2, v4, v7）已从代码库删除，保持仓库洁净。
