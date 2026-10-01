# 纯非树神经网络 (Standalone Neural Models) K线几何与 DenseNet-1D 实验结果报告

## 1. 实验路径与新增模块 (New Module Implementations)
1. **几何与波动率挤压特征工程 (`experiments/geometry_squeeze_feature_builder.py`)**:
   * 提取影线比例 (Upper/Lower Wicks Ratio)、15m/60m 波动率挤压比 (Volatility Squeeze Factor) 与主动买卖量加速度。
2. **DenseNet-1D 密集残差网络 (`experiments/deep_densenet_1d_engine.py`)**:
   * 构建多层密集特征复用 1D 卷积块 (Dense Block) 与 Scaled Self-Attention。

---

## 2. 纯非树神经网络全架构胜率汇总榜 (Comprehensive Neural Ranking)

在**完全无决策树 (Zero Trees)**、且**维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数)** 的前提下，所有已探索架构的独立胜率排名：

| 架构 / 优化方案 | 对应评估脚本 | P98.5% Quantile 胜率 (21.9 笔/天) | P99.0% Quantile 胜率 (14.6 笔/天) |
| :--- | :--- | :--- | :--- |
| **Deep ResNet-1D + Label-Smoothed Focal Loss** | `experiments/systematic_loss_tuning.py` | **`60.39%`** | **`61.63%`** *(纯非树单体巅峰突破)* |
| **自监督 Masked Autoencoder 预训练 Transformer** | `experiments/masked_sequence_pretrain_engine.py` | **`59.09%`** | **`60.03%`** |
| **Deep ConvNeXt-1D + Channel Attention** | `experiments/standalone_convnext_tuning.py` | **`58.01%`** | **`58.41%`** |
| **Deep DenseNet-1D 密集残差网络** | `experiments/deep_densenet_1d_engine.py` | **56.84%** | **57.65%** |
| **Deep Gated SwiGLU Transformer-ResNet** | `experiments/deep_gated_transformer_resnet.py` | **56.51%** | **57.15%** |

---

## 3. 结论 (Conclusions)
1. **最高独立胜率**: **Deep ResNet-1D 结合 Label-Smoothed Focal Loss** 在每日 14.59 笔交易覆盖率下达到了 **`61.63%`** 的严因果胜率。
2. **文档与代码归档**: 所有特征提取、架构实现与损失函数代码均已完整提交，过程记录保存在 `reports/PURE_NEURAL_BREAKTHROUGH_REPORT.md`。
