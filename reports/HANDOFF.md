# SimpleTrade 项目架构与全量盲测交接文档 (HANDOFF.md)

本文档整理并归档 SimpleTrade 项目的核心设计架构、特征工程限制、防泄漏机制、模型训练与全币种全预测周期（15m / 30m / 60m）的最新盲测结果。

---

## 1. 核心设计原则与铁律 (Core Principles & Strict Rules)

1. **绝对无前视泄漏 (Zero Lookahead / Leakage Guarantee)**:
   * 必须使用 `causal_eval.py` (`eval_r2_causal_daily`) 进行评估；
   * 当日信号选择门槛 $\tau_D$ 必须严格依据盘前历史 90 天置信度分位数决定，禁止使用当天整体排序；
   * 特征计算严格使用 Polars 递推滑动窗口 `rolling_*`，禁止使用全局分桶（如全局 VPIN 分桶已被严禁）。
2. **特征限制 (Allowed Feature Inputs)**:
   * 仅允许使用 OHLC（开高低收）价格派生特征与主动买/卖成交量（`buy_vol`, `sell_vol`）；
   * 严禁使用总成交量、计价货币成交量、ATR 以及成交笔数。
3. **高置信度发单约束 (High Confidence P99 Quantile)**:
   * 事件期权策略统一锁定 Top 1% 置信度（P99 分位数），维持日均 **~15 单/天** 发单频率，严禁通过过度缩减信号量来虚高胜率。

---

## 2. 核心架构与模型族 (Model Families Architecture)

系统采用 **三大正交模型族（Three Families Stacking）**：
1. **Family 1 (Pattern GBDT)**: 由 CatBoost 与 LightGBM 构成的对称树与 Leaf-wise 几何树模型族；
2. **Family 2 (PatternResNet CNN)**: 包含残差块（ResBlock）与 LayerNorm 的 1D 卷积神经网络，使用 60 分钟 1min 滚动窗口，捕捉 K 线微观形态突破；
3. **Family 3 (JOINT Cross-Asset Pool)**: 包含 BTC 与 ETH 跨币种联动特征的 Deep XGBoost 跨资产模型族；
4. **决策层 (Rank Uniformization Voting Meta-Learner)**: 对三族预测概率进行 Percentile Rank 归一化后 Soft-Voting 融合。

---

## 3. 全币种与全预测周期严格无前视盲测统计矩阵 (Test Matrix)

评估基于近 1.5 年真实盘面测试集，统一门槛为 P99 Quantile（日均 ~15 单）：

| 标的币种 | 预测周期 | 总体盲测胜率 | 日均发单量 (单/天) | 坏月份数 (<55%) | 最差单月胜率 | 坏月详细信息 (月份 / 胜率 / 该坏月总发单量) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BTC** | **H = 15m** | **`61.82%`** | **14.71 单/天** | ★ **0 个** | `55.88%` | **无坏月**（所有 16 个测试月份胜率均 $\ge 55.88\%$） |
| **BTC** | **H = 30m** | **`62.06%`** | **15.39 单/天** | ★ **0 个** | `55.12%` | **无坏月**（所有 16 个测试月份胜率均 $\ge 55.12\%$） |
| **ETH** | **H = 30m** | **`62.13%`** | **15.02 单/天** | **1 个** | `51.97%` | 1) `2025-09`: **51.97%** (发单量: **458 单**) |
| **ETH** | **H = 15m** | **`60.86%`** | **14.67 单/天** | **2 个** | `52.02%` | 1) `2025-09`: **54.01%** (发单量: **474 单**)<br>2) `2026-02`: **52.02%** (发单量: **446 单**) |
| **ETH** | **H = 60m** | **`61.10%`** | **14.91 单/天** | **2 个** | `50.26%` | 1) `2025-06`: **53.18%** (发单量: **393 单**)<br>2) `2026-01`: **50.26%** (发单量: **388 单**) |
| **BTC** | **H = 60m** | **`61.34%`** | **14.96 单/天** | **5 个** | `50.42%` | 1) `2025-06`: **54.55%** (462 单)<br>2) `2025-08`: **51.14%** (659 单)<br>3) `2026-01`: **50.42%** (474 单)<br>4) `2026-07`: **53.85%** (468 单)<br>5) `2026-08`: **52.84%** (299 单) |

---

## 4. 关键项目文件说明 (Key Repositories)

* `causal_eval.py`: 严格无前视盘前阈值评估模块 (`eval_r2_causal_daily`)；
* `audit_all_leaks.py`: 数据泄漏 5 维审计脚本；
* `features.py`: C(t) 递推精简特征生成器；
* `experiments/restore_high_winrate_model.py`: 恢复的三家族 PatternResNet + Pattern GBDT + JOINT 架构；
* `experiments/eval_single_runner.py`: 单币种单周期快速无前视盲测运行器；
* `reports/ALL_SYMBOLS_HORIZONS_REPORT.md`: 全币种全周期统计报告；
* `reports/EXPERIMENTS_STOCH_OPTIMIZATION.md`: Stochastic 与特征优化实验记录（EXP-00 ~ EXP-06）。
