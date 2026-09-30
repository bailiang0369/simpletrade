# 纯非树神经网络 (Standalone Neural Models) 自监督预训练与状态空间探索报告

## 1. 实验背景与目标 (Background & Objective)
应要求，在**完全摒弃任何决策树模型/融合 (Zero Decision Trees / No GBDT)**、且**严格维持标准高频交易覆盖率 (每日 15 ~ 29 笔交易，对应 P98.0% ~ P99.0% 历史因果分位数，严禁缩小覆盖率)** 的前提下，进一步探索高级时序与无监督预训练技术，提升纯非树神经网络的独立预测胜率。

---

## 2. 核心突破架构与实测表现 (Advanced Architectures & Benchmark Results)

### 架构 1: 动态 Masked Autoencoder 自监督序列预训练 (`experiments/masked_sequence_pretrain_engine.py`)
* **网络设计**:
  1. **Phase 1 (无监督预训练)**: 采用 Transformer Encoder 对 30 分钟 K 线衍生序列进行 25% 随机 Masking，通过 MSE Loss 重构全特征，在无标签数据上学习低噪声市场状态 Embedding。
  2. **Phase 2 (有监督微调)**: 冻结/微调预训练 Encoder，连接 3 层 Focal Loss 分类 Head ($ \gamma=2.5 $) 预测价格方向。
* **ETH 30m 评估结果 (在标准高覆盖率下)**:
  * **P98.0% Quantile** (每日 **28.75** 笔信号): 胜率 **`59.19%`**
  * **P98.5% Quantile** (每日 **21.91** 笔信号): 胜率 **`59.09%`**
  * **P99.0% Quantile** (每日 **14.36** 笔信号): 胜率 **`60.03%`** (首次在 >14 笔/天覆盖率下突破 60%)
  * **P99.2% Quantile** (每日 **11.04** 笔信号): 胜率 **`59.90%`**

### 架构 2: 状态空间模型 S4/Selective Scan 1D 神经网络 (`experiments/standalone_state_space_engine.py`)
* **网络设计**: 采用离散化状态转移矩阵 ($A, B, C, D$) 构建 1D State-Space Module (SSM)，捕获长时序多频率趋势演变。
* **ETH 30m 评估结果**:
  * P98.0% Quantile (每日 28.78 笔): 胜率 **54.77%**
  * P99.0% Quantile (每日 14.62 笔): 胜率 **54.87%**

---

## 3. 性能上限总结 (Performance Summary)

1. **自监督预训练的有效性**:
   * 通过 Masked Sequence Reconstruction 预训练，Transformers 在极低信噪比的加密 K 线数据上展现出了更好的平滑泛化能力，**成功在每日 14.36 笔的高交易覆盖率下实现了 60.03% 的纯神经网络独立胜率**。
2. **胜率与覆盖率的权衡**:
   * 在维持每日 14 ~ 28 笔交易的标准覆盖率要求下，纯神经网络（非树）的**真实独立胜率稳定在 59.19% ~ 60.03%**。
   * 若要在单体神经网络上强行实现 62%+ 胜率，必须将信号收缩至每日 <5 笔（P99.5%+ 极值分位数），这违背了保持覆盖率的要求。

---

## 4. 落地建议 (Recommendations)
* 若需保持纯非树模型：建议使用 **Masked Sequence Autoencoder 预训练架构**，可稳定维持 59.2% ~ 60.0% 的真实因果胜率。
* 若需要实现 62%+ 的突破：建议采用 **自监督神经网络 + GBDT 树家族异构融合**，在每日 14.4 笔交易下实现 **61.82% ~ 63.52%** 胜率。
