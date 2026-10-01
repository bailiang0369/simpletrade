# 纯非树神经网络 (Standalone Neural Models) 系统性版本演进与 61.63% 突破总结报告

## 1. 系统性版本控制与淘汰物理清理规范 (Dedicated Versioning Rules)
按照指示，所有新改动均在**独立的新版本文件中实现，绝对不覆盖/修改已有的 Baseline**，并对各版本的独立胜率执行严因果评估与淘汰物理清理机制：
* `experiments/neural_62_breakthrough_v1.py`: ResNet-1D + 高阶收益率曲率 + Label-Smoothed Focal Loss ($ \gamma=2.8 $)。
* `experiments/neural_62_breakthrough_v3.py`: 高阶微观结构与流动性动量特征 + 4 Epoch 余弦退火。
* `experiments/neural_62_breakthrough_v5.py`: Dynamic Margin BCE Focal Loss + 高阶物理动量特征。
* `experiments/neural_62_breakthrough_v6.py`: Spatial-Temporal Graph Attention Layer (GAT-1D) 多时域节点图网络。
* `experiments/neural_62_breakthrough_v8.py`: 自监督 Transformer Encoder + ResNet-1D 联合微调网络。
* `experiments/neural_62_breakthrough_v9.py`: 多尺度扩张 ConvNeXt-1D 与注意力融合网络。
* `experiments/neural_62_breakthrough_v13.py`: 多尺度扩张 ConvNeXt-1D + 通道注意力 (Channel Attention)。
* `experiments/neural_62_breakthrough_v16.py`: 实数傅里叶频域幅值 (FFT Real Spectral) + ResNet-1D 网络。
* `experiments/neural_62_breakthrough_v17.py`: 多尺度对数收益率二阶导数 + 订单流分布偏度 + ResNet-1D。
* `experiments/neural_62_breakthrough_v18.py`: 跨 Bar 主动买卖动量加速度 + 比率挤压 + 多尺度 1D 扩张残差块。

---

## 2. 纯非树神经网络有效版本胜率汇总排行榜

在**完全无决策树 (Zero Trees)**、且**维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数)** 的前提下，所有有效保存版本胜率排名：

| 版本 / 架构 | 对应源码文件 | P98.5% Quantile 胜率 (21.9 笔/天) | P99.0% Quantile 胜率 (14.6 笔/天) | P99.5% Quantile 胜率 (7.6 笔/天) | 代码保存状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-1D + Label-Smoothed Focal Loss** | `experiments/systematic_loss_tuning.py` | **`60.39%`** | **`61.63%`** | **60.68%** | **有效保存** |
| **Neural 62 Breakthrough V13 (Multi-Scale ConvNeXt CA)** | `experiments/neural_62_breakthrough_v13.py` | **`59.27%`** | **`60.53%`** | **58.30%** | **有效保存** |
| **Neural 62 Breakthrough V6 (GAT-1D)** | `experiments/neural_62_breakthrough_v6.py` | **`57.15%`** | **`59.39%`** | **`60.94%`** | **有效保存** |
| **Neural 62 Breakthrough V5** | `experiments/neural_62_breakthrough_v5.py` | **`58.77%`** | **`59.64%`** | **`60.58%`** | **有效保存** |
| **Neural 62 Breakthrough V3** | `experiments/neural_62_breakthrough_v3.py` | **`59.80%`** | **`59.35%`** | **`60.43%`** | **有效保存** |
| **Masked Autoencoder 预训练 Transformer** | `experiments/masked_sequence_pretrain_engine.py` | **`59.09%`** | **`60.03%`** | **58.78%** | **有效保存** |
| **Neural 62 Breakthrough V18 (Multi-Scale Dilated ResNet)** | `experiments/neural_62_breakthrough_v18.py` | **`58.09%`** | **`58.60%`** | **`59.68%`** (P99.5%) | **有效保存** |
| **Neural 62 Breakthrough V17 (Multi-Scale Log-Ret Acceleration)** | `experiments/neural_62_breakthrough_v17.py` | **`58.21%`** | **`58.62%`** | **`59.81%`** (P99.5%) | **有效保存** |
| **Neural 62 Breakthrough V16 (FFT Spectral)** | `experiments/neural_62_breakthrough_v16.py` | **`57.87%`** | **`58.34%`** | **`59.30%`** | **有效保存** |
| **Neural 62 Breakthrough V8 (Transformer+ResNet)** | `experiments/neural_62_breakthrough_v8.py` | **`59.40%`** | **`58.29%`** | **`59.71%`** | **有效保存** |
| **Neural 62 Breakthrough V9 (Multi-Scale ConvNeXt)** | `experiments/neural_62_breakthrough_v9.py` | **`57.44%`** | **`58.47%`** | **`59.27%`** | **有效保存** |
| **Neural 62 Breakthrough V1** | `experiments/neural_62_breakthrough_v1.py` | **`58.57%`** | **`58.33%`** | **58.54%** | **有效保存** |
| **Deep ConvNeXt-1D + Channel Attention** | `experiments/standalone_convnext_tuning.py` | **`58.01%`** | **`58.41%`** | **`60.10%`** | **有效保存** |

---

## 3. 结论 (Conclusions)

1. **单体最高记录保持者**:
   * **`experiments/systematic_loss_tuning.py` (ResNet-1D + Label-Smoothed Focal Loss)** 在每日 14.59 笔交易的高频覆盖率下实现了 **`61.63%`** 的严因果胜率。
2. **新有效版本入选**:
   * **`v18` (跨 Bar 主动买卖动量加速度 + 多尺度 1D 扩张残差块)** 在 P99.5% 下达到了 **`59.68%`** 胜率，成功入选有效保存序列。
