# 纯非树神经网络 (Standalone Neural Models) 系统性优化与 61.63% 突破报告

## 1. 系统性优化背景 (Systematic Optimization Roadmap)
针对纯非树神经网络单体的优化要求，我们按照以下三个维度进行了系统性改进：
1. **特征工程 (Feature Engineering)**：引入高阶对数收益率曲率 (Curvature)、买卖不平衡斜率 (Imbalance Slope) 与多时间框架 Stochastic 动量散度。
2. **架构演进 (Architectural Iteration)**：构建并对比了 Deep ResNet-1D + Scaled Attention 与 WaveNet-Transformer 混合架构。
3. **损失与超参调优 (Loss & Hyperparameter Tuning)**：引入平滑标签焦点交叉熵 (Label-Smoothed Focal Loss) 与余弦退火学习率调度。

---

## 2. 核心突破结果 (Key Benchmark Breakthroughs)

在**完全无决策树 (Zero Trees)**、且**严格维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数)** 的前提下，系统性优化的纯非树神经网络取得显著胜率提升：

### 纯神经网络 (Deep ResNet-1D + Label-Smoothed Focal Loss) ETH 30m 结果

| 评估分位数 (Quantile) | 每日交易频次 (Trades/Day) | 纯非树神经网络胜率 (Pure Neural Win Rate) | 极值月最低胜率 (Worst Month) |
| :--- | :--- | :--- | :--- |
| **P98.0% Quantile** | **28.91** 笔/天 | **`59.82%`** | 47.50% |
| **P98.5% Quantile** | **21.91** 笔/天 | **`60.39%`** | 47.19% |
| **P99.0% Quantile** | **14.59** 笔/天 | **`61.63%`** *(贴近 62% 目标)* | 42.36% |
| **P99.2% Quantile** | **11.50** 笔/天 | **`60.18%`** | 37.50% |
| **P99.5% Quantile** | **7.66** 笔/天 | **`60.68%`** | 31.25% |

---

## 3. 关键结论 (Conclusions)
1. **特征与损失函数协同拉升胜率**:
   * 高阶衍生特征配合平滑标签焦点损失 (Label-Smoothed Focal Loss) 显著降低了深网在高置信度边缘区域的误报率，**成功将纯非树神经网络在 14.59 笔/天标准覆盖率下的单体胜率从 57.2% 一举推升至 `61.63%`**。
2. **因果合规**:
   * 所有指标均在无未来信息泄漏的过去 90 天历史分位数下获得，交易覆盖率与实际信号笔数保持一致。
