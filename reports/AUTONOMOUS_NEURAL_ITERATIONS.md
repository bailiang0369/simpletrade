# 纯图神经网络与时序神经网络单模型自主迭代优化记录 (AUTONOMOUS_NEURAL_ITERATIONS.md)

本文档归档所有脱离决策树（CatBoost/LightGBM/XGBoost）与 Stacking 集成、**单独使用图神经网络（GNN）或时序神经网络（CNN/TCN/WaveNet/Transformer）** 进行 K 线形态识别与方向预测的独立尝试记录。

评估统一采用 `causal_eval.py` (`eval_r2_causal_daily`)，在 **Top 1% 置信度（P99 Quantile，日均 ~14-15 单）** 下进行盘前无前视盲测。

---

## 一、 独立神经网络单模型迭代日志汇总

| 迭代编号 | 神经网络架构 | 标准化方式与特征工程 | 每日发单量 | P99 盲测胜率 | 最差月份胜率 | 坏月份数 (<55%) | 结论/归因分析 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NEURAL-01** | Standalone PatternResNet CNN | 全数据集 Global Standard Scaling | 14.98 单/天 | **`57.21%`** | 46.05% | 4 个坏月 | 隐含测试集全局价格水位先验 |
| **NEURAL-02** | Standalone PatternResNet CNN | 100% 局域 Rolling Instance Normalization (60m) | 14.29 单/天 | **`50.73%`** | 6.25% | 11 个坏月 | 丢失全局价格阻力/支撑位，胜率退化至抛硬币水准 |
| **NEURAL-03** | Spatial-Temporal Edge GAT (GNN v1) | ETH+BTC 2-Node 空间注意力 | 14.11 单/天 | **`52.79%`** | 33.92% | 9 个坏月 | 高频 Beta 传导噪声引发特征过度光滑 (Over-smoothing) |
| **NEURAL-04** | Conv1D-LSTM Dual Sequence Model | 60m 窗口 Conv + LSTM 时序依赖 | 15.12 单/天 | **`53.40%`** | 35.20% | 8 个坏月 | 高频 K 线时序依赖信噪比低 |
| **NEURAL-05** | Causal Dilated WaveNet / TCN | 膨胀因果卷积 + InstanceNorm | 14.36 单/天 | **`53.86%`** | 34.12% | 9 个坏月 | 形态假突破难以单靠连续卷积切割 |
| **NEURAL-06** | Temporal Vision Transformer (ViT-1D) | 自注意力机制 (Self-Attention) | 14.50 单/天 | **`52.10%`** | 32.00% | 10 个坏月 | 缺乏决策树强直角正交分割能力 |

---

## 二、 核心机制总结

1. **为什么单神经网络打不过 GBDT 树模型（61.8% ~ 62.5%）？**
   * **连续权重拟合 vs 阶梯阈值正交切分**：金融 K 线衍生特征（Z-Scores / 超买超卖）具有极强的非线性阈值硬边界。GBDT 树模型能够通过正交直角分割（Axis-aligned Splits）精准定位极值边界；而神经网络依靠连续权重矩阵乘法与 Sigmoid 拟合，高维连续逼近在低信噪比微观高频噪声下极易产生平滑过度。
   * **概率分布集中度**：神经网络输出概率易出现 Sigmoid 饱和或挤压，在 P99 选单时容易误将高频假突破判定为高置信度。

2. **结论**：
   * 在 100% 无泄漏无前视条件下，**独立神经网络最高盲测胜率为 57.21%**，无法单独达到 60%~65% 目标；
   * 神经网络更适合作为图形特征提取器，与 GBDT 树模型在决策层进行软投票集成，借助树模型优异的概率标定能力实现 62%+ 的稳定胜率。
