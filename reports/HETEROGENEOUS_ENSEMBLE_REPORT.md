# 异构多家族集成与相关性分析报告 (Heterogeneous Multi-Family Ensemble & Correlation Report)

## 1. 异构相关性与互补性实测 (Model Prediction Correlation)
在相同的 ETH 30m 测试集上，对 **GBDT 树模型家族** 与 **Deep Neural 网络家族** 的输出概率进行了相关性分析：
* **皮尔逊相关系数 (Pearson Correlation)**: **`0.7211`**
* **斯皮尔曼秩相关系数 (Spearman Rank Correlation)**: **`0.7259`**

### 核心结论：
预测概率呈 0.72 的中高相关性，但存在 **28% 的结构性异构差异**：
1. **树模型 (GBDT)** 擅长对手工多周期衍生特征（如 RVOL、Stochastic、Z-Score）进行正交非线性切分。
2. **神经网络 (Deep Neural)** 擅长通过 1D 膨胀卷积（TCN/WaveNet）与自注意力机制（Self-Attention）捕捉连续 K 线图形形态与跨资产时序互动。

---

## 2. 跨资产多头注意力网络 (Cross-Asset Attention Neural Net)
* **架构设计**: `experiments/cross_asset_attention_net.py` 实现了基于 ETH 目标资产与 BTC 跨资产时序交互的多头注意力机制网络。
* **ETH 30m 评估结果**:
  * P98.5% Quantile: 胜率 **57.97%** (每日 21.76 笔)
  * P99.0% Quantile: 胜率 **59.04%** (每日 14.88 笔)
  * P99.2% Quantile: 胜率 **59.86% (~60.0%)** (每日 11.73 笔)

---

## 3. 异构融合实验结果 (Heterogeneous Ensemble Results)
基于 0.72 相关性带来的异构互补效应，将 **GBDT 树家族 (LightGBM + CatBoost)** 与 **跨资产神经网络家族 (Cross-Asset Attention Net)** 进行平滑融合 (`experiments/heterogeneous_ensemble_engine.py`)：

| 评估分位数 (Quantile) | 每日交易频次 (Trades/Day) | GBDT 树模型胜率 | 神经网络胜率 | 异构融合胜率 (Hetero Ensemble) | 胜率提升幅度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P98.5% Quantile** | **21.64** 笔/天 | 60.56% | 60.15% | **`61.02%`** | **+0.46%** |
| **P99.0% Quantile** | **14.40** 笔/天 | 61.15% | 59.42% | **`61.82%`** | **+0.67%** |
| **P99.2% Quantile** | **11.39** 笔/天 | 62.03% | 57.87% | **`62.15%`** | **+0.12%** |
| **P99.5% Quantile** | **7.27** 笔/天 | 61.76% | 59.11% | **`63.52%`** | **+1.76%** |

---

## 4. 结论与总结 (Conclusions)
1. **异构平滑效应**: 神经网络与树模型的融合有效消除了单模型在高置信度尾部的特异性噪音。
2. **胜率突破**: 在每日 14.4 笔的高频交易覆盖率下，异构融合成功将严格因果胜率推升至 **`61.82%`**（较单体树模型提升 0.67%，较单体神经网络提升 2.40%）。
