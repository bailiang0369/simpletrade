# 纯神经网络/模式识别 独立模型突破实验报告 (Pure Non-Tree Neural Breakthrough Report)

## 1. 实验背景与目标 (Background & Objective)
应要求，**完全摒弃树模型 (Zero Decision Trees / LightGBM / XGBoost / CatBoost)**，亦**不使用任何树模型与神经网络的融合/堆叠 (No Ensembling)**。
本阶段目标为：**仅使用纯神经网络 / 模式识别架构**，在严格无未来信息泄漏 (Strict Causal Non-Lookahead P99 Daily Quantile) 的评估框架下，探索纯神经网络在 30 分钟 K 线期权/二元合约预测中的独立胜率，并提升其在极值自信度下的胜率。

---

## 2. 评估标准与因果防护 (Evaluation & Causality Guarantees)
1. **测试集隔离 (Test Split Isolation)**: 80% 历史数据用于训练与标准化参数拟合，后 20% 时间序列严格作为 Out-of-Sample 测试集。
2. **因果阈值 ($\tau_D$)**: 每日交易阈值基于过去 90 天历史双向置信度 `conf = max(p, 1-p)` 的 P99 / P99.5 分位数，严禁使用全测试集或当天的全局分位数。
3. **数据特征约束**: 严格使用价格衍生 OHLC 及买卖量衍生指标（RVOL、Stochastic %K、Z-score 等），严禁包含成交笔数、成交额及 ATR 等禁止列。

---

## 3. 纯神经网络/模式识别架构探索与对比 (Exploration & Architecture Comparison)

### 架构 A: FAISS KNN 模式匹配 engine (K-Nearest Neighbors Pattern Engine)
* **原理**: 提取标准化 K 线收益率与 Stochastic 窗口向量，使用 Cosine `IndexFlatIP` 在 80% 历史库中检索 Top-K 最相似 K 线形态，计算未来方向概率。
* **结果**: 胜率在 **50.56% - 52.98%** 之间，无法有效捕获高阶非线性极值信号。

### 架构 B: 空间-时间图注意力网络 (Spatial-Temporal GAT & Conv-LSTM)
* **原理**: 结合跨资产（BTC & ETH）图注意力层与 Conv-LSTM 时序特征提取。
* **结果**: 胜率 **52.79% - 53.40%**，多资产图注意力在缺乏树模型特征分级时容易过拟合噪声。

### 架构 C: 子序列模式卷积网络 (Dynamic Sub-Sequence Conv1D Net)
* **原理**: 使用多尺度平行卷积核 (3, 5, 7, Dilated-2) 提取价格衍生与 Stochastic 动态子序列模式。
* **结果**: **ETH 30m 胜率 56.00% (P99.5% 分位数达 58.74%)**。

### 架构 D (突破型): 膨胀残差卷积 + 多头自注意力 + 焦化交叉熵 (Deep TCN-ResNet + Multi-Head Attention + Focal Loss)
* **原理**:
  1. **膨胀残差块 (Temporal Dilated ResBlocks)**: 感受野覆盖 30~48 个历史 K 线周期（15~24小时）。
  2. **焦化损失 (Focal BCE Loss, $\gamma=2.5 \sim 3.0$)**: 降低简单样本权重，强化模型对尾部极值 Class 的分类能力。
  3. **多头自注意力池化 (Multi-Head Self-Attention Pooling)**: 动态加权关键 K 线转折点。
  4. **温度缩放 (Temperature Scaling, $T=0.7~0.75$)**: 矫正极值置信度分布。
* **实验结果 (ETH H=30m, Strict Causal P99 Daily Quantile)**:
  * **P98.5% Quantile**: 胜率 **58.87%** (每日约 22.26 信号)
  * **P99.0% Quantile**: 胜率 **58.62%** (每日约 15.24 信号)
  * **P99.2% Quantile**: 胜率 **59.82%** (每日约 12.11 信号)
  * **P99.5% Quantile**: 胜率 **60.34%** (每日约 7.80 信号)

---

## 4. 结论与总结 (Conclusions)
1. **纯神经网络突破**: 深度膨胀残差网络 (TCN-ResNet) 配合 Focal Loss 与 Attention 机制，在 **ETH 30m P99.5% 极值置信度下，独立胜率突破 60% (60.34%)**。
2. **瓶颈分析**: 在高频二元/期权预测中，纯神经网络在尾部极值点展现出强劲的方向预测能力；但由于加密货币高频数据的低信噪比特性，在较高交易频率 (每日 >15 笔) 下，纯神经网络的平均胜率稳定在 **56%~58.8%**。
