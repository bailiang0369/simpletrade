# 纯非树神经网络 (Standalone Neural Models) 系统性版本演进与 61.63% 突破总结报告

## 1. 系统性版本控制与开发规范 (Dedicated Version Control Rules)
为防止优化过程中的改动造成代码退化或覆盖已有 Baseline，所有新改动均在**独立的新版本文件**中实现：
* `experiments/neural_62_breakthrough_v1.py`: 基于 ResNet-1D + 高阶收益率曲率 + Label-Smoothed Focal Loss ($ \gamma=2.8 $)。
* `experiments/neural_62_breakthrough_v2.py`: 基于温度平滑校准 ($T=0.62$) + Self-Attention。
* 淘汰机制：独立胜率低于 58% 的改动直接淘汰，不予以保存。

---

## 2. 纯非树神经网络有效版本胜率排行榜 (Standalone Neural Ranking)

在**完全无决策树 (Zero Trees)**、且**维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数)** 的前提下，所有保存版本独立胜率：

| 版本 / 架构 | 对应评估文件 | P98.5% Quantile 胜率 (21.9 笔/天) | P99.0% Quantile 胜率 (14.6 笔/天) | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-1D + Label-Smoothed Focal Loss** | `experiments/systematic_loss_tuning.py` | **`60.39%`** | **`61.63%`** *(纯非树单体最高纪录)* | **有效保存** |
| **Masked Autoencoder 预训练 Transformer** | `experiments/masked_sequence_pretrain_engine.py` | **`59.09%`** | **`60.03%`** | **有效保存** |
| **Neural 62 Breakthrough V1** | `experiments/neural_62_breakthrough_v1.py` | **`58.57%`** | **`58.33%`** | **有效保存** |
| **Deep ConvNeXt-1D + Channel Attention** | `experiments/standalone_convnext_tuning.py` | **`58.01%`** | **`58.41%`** | **有效保存** |
| **Neural 62 Breakthrough V2** | `experiments/neural_62_breakthrough_v2.py` | **`57.23%`** | **`56.05%`** | *退化淘汰* |

---

## 3. 结论 (Conclusions)

1. **单体最高纪录突破**:
   * **Deep ResNet-1D 结合 Label-Smoothed Focal Loss (`experiments/systematic_loss_tuning.py`)** 在 **每日 14.59 笔** 的高频实战覆盖率下实现了 **`61.63%`** 的严因果胜率，逼近 62% 目标。
2. **严因果防泄漏**:
   * 所有评估过程严格遵守价格与主动买卖量特征限制（严禁 quote_volume, total_volume, atr, trade_count），并基于过去 90 天历史分位数计算。
