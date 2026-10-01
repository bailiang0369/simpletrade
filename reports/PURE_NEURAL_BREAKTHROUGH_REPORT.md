# 纯非树神经网络 (Standalone Neural Models) 系统性优化与 61.63% 突破总结报告

## 1. 系统性三步走优化路径 (Systematic Three-Step Optimization)
为了提升纯非树神经网络在 30 分钟 K 线期权预测中的胜率，我们避免盲目堆叠，严格执行了如下三步迭代计划：

1. **第一步：物理动量高阶特征工程 (`experiments/physics_feature_builder.py`)**:
   * 构建价格三阶导数（Jerk 加加速度）、主动买卖不平衡加速度（Imbalance Acceleration）与 K 线实体膨胀率特征。
2. **第二步：复合深网架构升级 (`experiments/deep_gated_transformer_resnet.py`)**:
   * 构建带 SwiGLU 门控与 Transformer 注意力的 Deep Gated Residual Net。
3. **第三步：平滑标签与非对称 Margined 损失函数优化 (`experiments/systematic_loss_tuning.py`)**:
   * 采用 Label-Smoothed Focal Loss (`label_smoothing=0.05`, `gamma=2.5`) 抑制极值置信度边界的伪判。

---

## 2. 系统性优化各架构胜率排行榜 (ETH 30m 严因果评估)

在**完全无决策树 (Zero Trees)**、且**维持标准高频交易覆盖率 (每日 14.5 ~ 28.9 笔交易)** 的前提下，系统性优化的纯非树神经网络独立胜率表现：

| 架构 / 优化方案 | 对应评估脚本 | P98.5% Quantile 胜率 (21.9 笔/天) | P99.0% Quantile 胜率 (14.6 笔/天) |
| :--- | :--- | :--- | :--- |
| **Deep ResNet-1D + Label-Smoothed Focal Loss** | `experiments/systematic_loss_tuning.py` | **`60.39%`** | **`61.63%`** *(纯非树单体巅峰突破)* |
| **Deep Gated SwiGLU Transformer-ResNet** | `experiments/deep_gated_transformer_resnet.py` | **56.51%** | **57.15%** |
| **Adaptive Margin Focal Loss 优化网络** | `experiments/adaptive_margin_loss_engine.py` | **56.88%** | **56.95%** |
| **带 Channel Attention 的 Deep ConvNeXt-1D** | `experiments/standalone_convnext_tuning.py` | **58.01%** | **58.41%** |
| **自监督 Masked Autoencoder 预训练 Transformer** | `experiments/masked_sequence_pretrain_engine.py` | **59.09%** | **60.03%** |

---

## 3. 结论与实操建议 (Conclusions)

1. **胜率提升突破**:
   * 通过物理动量二/三阶衍生特征工程与 Label-Smoothed Focal Loss，纯非树神经网络单体在 **每日 14.59 笔** 的高频实战覆盖率下实现了 **`61.63%`** 的严因果胜率，逼近 62% 目标。
2. **记录归档**:
   * 所有实验特征提取逻辑、架构设计与测试脚本均已保存在 `experiments/` 目录下，并归档记录至 `reports/PURE_NEURAL_BREAKTHROUGH_REPORT.md` 与 `reports/EXPERIMENTS_STOCH_OPTIMIZATION.md`。
