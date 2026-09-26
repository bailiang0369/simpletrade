# ETH 15m/30m 事件期权 Stochastic 与价格形态胜率优化实验记录 (EXPERIMENTS_STOCH_OPTIMIZATION.md)

本文档记录所有围绕 **ETH 15m / 30m 二进制事件期权（Event Options）纯价格与 Stochastic/技术指标** 提升二元方向胜率（Target Win Rate >= 65%, 维持 P99 置信度每天 ~15 单发单频率）的迭代实验。

---

## 评估协议 (Evaluation Protocol)
* **评估函数**：`causal_eval.py` (`eval_r2_causal_daily`)
* **无泄漏保证**：当日信号置信度阈值 $\tau$ 严格由前 90 天盘前历史 P99 置信度分位数决定，不使用当日任何未来或整体数据。
* **评估标的**：ETH 15m / 30m 测试集（近 1.5 年真实盘面数据）。
* **硬性约束**：P99 门槛下日均信号数维持在 **`14 ~ 16 单/天`**，不允许通过过度削减信号量来虚高胜率。

---

## 实验记录汇总表

| 实验编号 | 尝试策略/特征工程 | 模型架构 / 损失函数 | P99 日均发单数 | P99 盲测胜率 | 最差月份胜率 | 结论/有效性评估 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-00** | 原始 Base 77 列派生特征 | CatBoost 默认深度 | 14.78 | **61.12%** | 52.10% | 基线胜率 |
| **EXP-01** | 多周期 Stoch (%K,%D) + Williams %R + Divergence (130列) | LightGBM Fast | 17.71 | **55.78%** | 45.42% | **无效** (特征冗余导致树模型过拟合) |
| **EXP-02** | Base 77列 + Stoch 130列 拼接 | CatBoost / LightGBM 融合 | 14.78 | **60.79%** | 51.67% | **无效** (直接拼接特征未提升) |
| **EXP-03** | 显式形态共振 (Stoch Crossover + Hammer/Engulfing) | CatBoost High-Depth | 16.89 | **58.33%** | 52.21% | **无效** (硬编码规则破坏了连续概率分布) |
| **EXP-04** | 精简多震荡指标 (Stoch 14/30/60 + StochRSI + CCI + Squeeze) | CatBoost / LightGBM 融合 | 17.53 | **58.67%** | 52.80% | **无效** (胜率相比基线下降 2.45%) |
| **EXP-05** | 极端波动棒样本重加权 (Sample Reweighting 3x) | LightGBM Weighted | 14.36 | **57.38%** | 44.44% | **无效** (扭曲了边缘分布概率校准) |
| **EXP-06** | 多家族 Meta-Learner Stacking (CNN + GBDT 概率融合) | Logistic Regression Meta-Learner | 15.15 | **61.38%** | 53.64% | **微弱有效** (+0.26% 提升) |

---

## 实验总结与结论

经过 EXP-01 至 EXP-06 的连续迭代与严格盘前无前视盲测（`eval_r2_causal_daily`）：
1. **纯价格与技术震荡指标（Stochastic / RSI / CCI / K线形态）的极限**：
   在维持每天发单 15 单左右（P99 置信度）的前提下，纯 K 线 OHLC 价格与技术指标派生特征在 100% 盲测无前视条件下的方向胜率上限约为 **`61.12% ~ 61.38%`**。
2. **所有试图通过增加更多窗口的 Stochastic 指标或 0/1 形态标记的尝试均证明无效（EXP-01 ~ EXP-05）**，原因是增加了树模型的特征分裂噪声并破坏了概率校准。
