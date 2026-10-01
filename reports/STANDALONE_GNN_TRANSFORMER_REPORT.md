# 独立图神经网络 (GNN) 与时序神经网络 (Neural Models) 单模型胜率与统计机制归因报告 (STANDALONE_GNN_TRANSFORMER_REPORT.md)

本报告脱离任何决策树模型（CatBoost/LightGBM/XGBoost）与 Stacking 集成，针对 **“单独使用图神经网络（GNN）或时序神经网络（CNN/Transformer/LSTM），能否在纯盲测无前视下突破 60% 胜率”** 这一核心疑问，进行了严格的独立基准测试与数学统计归因分析。

评估统一采用 `causal_eval.py` (`eval_r2_causal_daily`)，在 **Top 1% 置信度（P99 Quantile）** 下进行盘前无前视盲测。

---

## 一、 独立神经网络单模型盲测结果汇总 (ETH 15m, P99 Quantile, 日均 ~14-15 单)

| 神经网络模型架构 | 模型网络机制 | 每日发单量 (单/天) | 独立盲测胜率 | 最差单月胜率 | 坏月份数 (<55%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **独立 GNN v1 (Spatial-Temporal GAT)** | 图注意力网络 (Cross-Asset Spatial GAT + Temporal Conv) | **14.11 单/天** | **`52.79%`** | `33.92%` | 9 个坏月 |
| **独立 PatternResNet CNN (全局 Scaling)** | 1D 残差卷积网络 (60m 图像/向量窗口) | **14.98 单/天** | **`57.21%`** | `46.05%` | 4 个坏月 |
| **独立 PatternResNet CNN (Rolling Z-Score)** | 100% 局域 Rolling Standardization 残差网络 | **14.29 单/天** | **`50.73%`** | `6.25%` | 11 个坏月 |
| **独立 Conv1D-LSTM** | 卷积时序长短期记忆网络 | **15.12 单/天** | **`53.40%`** | `35.20%` | 8 个坏月 |

---

## 二、 核心结论与数学统计机制解答

### 结论：单独使用图神经网络或时序神经网络，在完全无泄漏/无树模型辅助下，**无法在 Top 1% 高频发单下稳定突破 60% 胜率**（独立胜率封顶在 52% ~ 57%）。

之所以神经网络单模型打不过 GBDT 树模型（61%~62.5%），核心原因有以下三点：

1. **正交直角分割（Axis-Aligned Boundary Splits）与连续权重拟合的差异**：
   * K 线与量价指标具有极强的不连续“阈值效应”（如 $Z > +2.0$ 强阻力、RSI < 20 极度超卖）。**GBDT 树模型**通过正交直角切分能够精准定位并锁定极值点的硬边界；
   * **神经网络（GNN/CNN/Transformer）** 依靠连续权重矩阵乘法与 Sigmoid/Softmax 激活函数逼近，高维连续拟合在面对金融 K 线的低信噪比（Low Signal-to-Noise Ratio）微观噪声时，极易产生平滑过度或拟合抖动。

2. **神经网络预测概率的 Sigmoid 饱和与概率标定（Calibration）失效**：
   * 神经网络在 Softmax/Sigmoid 输出端的概率集中度较差（存在大量概率极化或在中部 0.48~0.52 挤压），导致在 P99 置信度尾部选单时，选出的并不是真正有确定性方向的样本，而是概率分布被噪声拉偏的样本；
   * GBDT 树模型通过多树累加与 Leaf-wise 限制，输出概率具有极强的单调性与尾部分辨率。

3. **GNN 图注意力在双币种上的过度光滑（Over-smoothing）**：
   * 当 GAT（图注意力网络）在 ETH 和 BTC 节点之间传递空间注意力边（Spatial Edges）时，加密货币高频行情中的 Beta 传导噪声会导致节点特征发生过度光滑（Over-smoothing），稀释了 ETH 单币种微观突破的特异性。

---

## 三、 总结

* **单神经网络模型独立上限**：在 100% 严格因果无前视下，独立神经网络最高胜率为 **`57.21%`**（PatternResNet CNN）；
* **只有将 PatternResNet CNN 的形态特征与 GBDT 树模型的代数分割进行软投票集成（Dual/Three-Family Ensemble）**，借助树模型优异的概率标定能力，系统胜率才能突破 **`61.8% ~ 62.58%`**！
