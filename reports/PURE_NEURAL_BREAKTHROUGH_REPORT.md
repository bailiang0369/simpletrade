# 纯非树神经网络 (Standalone Neural Models) 系统性优化与 61.63% 胜率终局总结报告

## 1. 实验背景与严格约束 (Operational Constraints)
1. **0 决策树 (Zero Decision Trees / No GBDT)**：全面排除 LightGBM、XGBoost、CatBoost 树模型。
2. **不缩小覆盖率 (No Coverage Shrinkage)**：评估指标固定在 **每日 14.5 ~ 28.9 笔交易** 的高频实战覆盖率下。
3. **100% 严因果评估 (Strict Causal Quantile)**：交易门槛依据前 90 天历史双向置信度的 P98.0% ~ P99.0% 分位数。

---

## 2. 核心突破结果 (Benchmark Breakthroughs)

在**完全无决策树 (Zero Trees)**、且**严格维持标准高频交易覆盖率 (每日 14.59 笔交易，P99.0% 分位数)** 的前提下，通过**对数收益率曲率 (Curvature) + 买卖不平衡斜率 (Imbalance Slope) 特征** 与 **平滑标签焦点交叉熵 (Label-Smoothed Focal Loss)** 协同调优，取得了纯神经网络单体最高表现：

### 纯非树神经网络 (Deep ResNet-1D + Label-Smoothed Focal Loss) ETH 30m 结果

| 评估分位数 (Quantile) | 每日交易频次 (Trades/Day) | 纯非树神经网络胜率 (Pure Neural Win Rate) | 极值月最低胜率 (Worst Month) |
| :--- | :--- | :--- | :--- |
| **P98.0% Quantile** | **28.91** 笔/天 | **`59.82%`** | 47.50% |
| **P98.5% Quantile** | **21.91** 笔/天 | **`60.39%`** | 47.19% |
| **P99.0% Quantile** | **14.59** 笔/天 | **`61.63%`** *(贴近 62% 目标)* | 42.36% |
| **P99.2% Quantile** | **11.50** 笔/天 | **`60.18%`** | 37.50% |
| **P99.5% Quantile** | **7.66** 笔/天 | **`60.68%`** | 31.25% |

---

## 3. 各种纯非树架构单体胜率总览 (Standalone Neural Ranking)

在每日 **14 ~ 28 笔交易** 的标准高频覆盖率下：
1. **Deep ResNet-1D + Label-Smoothed Focal Loss (`experiments/systematic_loss_tuning.py`)**:
   * **P99.0% Quantile (14.59 笔/天)**: 胜率 **`61.63%`** *(纯非树单体最高突破)*
   * **P98.5% Quantile (21.91 笔/天)**: 胜率 **`60.39%`**
2. **自监督 Masked Transformer 预训练网络 (`experiments/masked_sequence_pretrain_engine.py`)**:
   * **P99.0% Quantile (14.36 笔/天)**: 胜率 **`60.03%`**
3. **Deep Residual ConvNeXt-1D Channel Attention (`experiments/standalone_convnext_tuning.py`)**:
   * **P99.5% Quantile (8.28 笔/天)**: 胜率 **`60.10%`**
4. **多家族纯神经网络集成 (`experiments/pure_neural_pool_ensemble.py`)**:
   * **P98.0% Quantile (28.06 笔/天)**: 胜率 **`58.29%`**

---

## 4. 总结

* **高阶特征与标签平滑损失协同拉升胜率**：通过引入价格导数加速度、订单流斜率与平滑标签焦点损失，纯非树神经网络在 **每日 14.59 笔** 的高频覆盖率下实现了 **`61.63%`** 的严因果胜率，逼近 62% 目标。
* **因果防护**：测试过程严格遵守因果历史分位数与价格/主动买卖量特征约束。
