# 纯神经网络与时序/图模型独立预测连续优化实验报告 (CONTINUOUS_NEURAL_OPTIMIZATION.md)

本报告记录**脱离决策树（CatBoost/LightGBM）与 Stacking 集成**，单独针对 **神经网络单模型（Deep Spatial-Temporal ResNet / WaveNet / Transformer / GNN）** 在高维全功能白名单特征（含多尺度波动率 `rvol`、移动平均线斜率 `ema_slope`、偏离 Z-Score `z_30/60/120` 与时间周期编码）上的独立预测能力提升实验。

评估统一采用 `causal_eval.py` (`eval_r2_causal_daily`)，在 **Top 1% 置信度（P99 Quantile，日均 ~14-15 单）** 下进行 100% 盘前历史 90 天无前视盲测。

---

## 一、 独立神经网络连续优化实验记录汇总

| 实验编号 | 神经网络架构 | 维度与标准化方式 | 标的 / 周期 | P99 日均发单量 | P99 盲测胜率 | 最差单月胜率 | 坏月份数 (<55%) | 归因与结论分析 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-N01** | Standalone PatternResNet | 100% 局域 Rolling Normalization (60m) | ETH H15m | 14.29 单/天 | **`50.73%`** | 6.25% | 11 个坏月 | 丢失全局价格水位，退化至抛硬币水准 |
| **EXP-N02** | Standalone PatternResNet | 13 列裸 K 几何通道 ($W=60\text{m}$) | ETH H15m | 14.98 单/天 | **`57.21%`** | 46.05% | 4 个坏月 | 纯裸 K 相对通道上限（缺乏支撑阻力位） |
| **EXP-N03** | Standalone PatternResNet | 59 列全白名单 Train-Set Standard Scaling | ETH H15m | 14.33 单/天 | **`56.99%`** | 42.79% | 7 个坏月 | 恢复全局 Z-Score 水位后的神经网络独立上限 |
| **EXP-N04** | Deep Spatial-Temporal ResNet | 59 列全白名单 + GroupNorm + Cosine Annealing | ETH H30m | 13.97 单/天 | **`57.55%`** | 43.49% | 4 个坏月 | 深层残差块收敛，盲测胜率提升至 57.55% |
| **EXP-N05** | Deep Spatial-Temporal ResNet | 59 列全白名单 + GroupNorm + Cosine Annealing | BTC H15m | 14.82 单/天 | **`52.30%`** | 42.50% | 8 个坏月 | BTC 高频噪声对神经网络梯度干扰大 |

---

## 二、 神经网络单模型胜率在 57.5% 遇到瓶颈的底层数学原因

1. **金融 K 线低信噪比与不连续硬边界（Axis-Aligned Splits vs Smooth Approximations）**：
   * 金融时间序列具有极强的阶梯硬边界（例如当偏离 Z-Score $Z > +2.0$ 时抛售压力剧增，或 RSI < 20 时引发快速均值回归）。**GBDT 树模型**通过正交直角切割（Axis-aligned Decision Splits）可以极其敏锐地锁定并切割这些硬边界，从而实现 **`61.8% ~ 62.58%`** 的高盲测胜率；
   * **神经网络（ResNet/Transformer/GNN）** 依靠连续矩阵乘法与 Sigmoid 激活函数逼近，在面对低信噪比（Low SNR）高频 K 线数据时，连续拟合极易将微观伪突破噪声误判为高置信度方向，导致概率极化与选单抖动。

2. **结论**：
   * 单神经网络模型在 100% 无前视/无树模型辅助的前提下，最佳盲测胜率为 **`57.55%`**（深层 Spatial-Temporal ResNet，ETH 30m）；
   * 要想实现 60%~62.5% 以上的实盘胜率，建议使用 GBDT 树模型对高维 Z-Score 边界进行正交切割。
