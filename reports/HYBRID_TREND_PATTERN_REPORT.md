# 因果趋势上下文 + 几何形态 + 跨资产相对动量混合引擎报告 (HYBRID_TREND_PATTERN_REPORT.md)

本报告记录将 **微观几何形态特征（Scale-Invariant Wicks/Body/Range Ratios）** 与 **大周期因果趋势 Z-Score 上下文（15m, 60m, 240m, 1440m 均线距离）** 及 **BTC-ETH 跨资产相对动量 Z-Score** 进行融合后的独立测试结果。

评估统一采用 `causal_eval.py` (`eval_r2_causal_daily`)，在 **Top 1% 置信度（P99 Quantile）** 下进行盘前无前视盲测。

---

## 一、 混合特征引擎（Hybrid Feature Engine）设计

1. **局部 K 线几何形态 (Local Geometry)**：
   * 实体方向比例 (`geom_body_ratio`)
   * 上影线比例 (`geom_upper_wick`)
   * 下影线比例 (`geom_lower_wick`)
   * 振幅比例 (`geom_range_pct`)
2. **大周期因果趋势上下文 (Causal Trend Context)**：
   * 距离 15m, 60m, 240m, 1440m 移动平均线的局部因果 Z-Score (`causal_z_15`, `causal_z_60`, `causal_z_240`, `causal_z_1440`)
   * 对数收益率动量 (`causal_lr_15`, `causal_lr_60`, `causal_lr_240`, `causal_lr_1440`)
3. **跨资产相对动量 (Cross-Asset Relative Momentum)**：
   * BTC vs ETH 相对动量 Spread (`cross_relative_lr_15`, `cross_relative_lr_60`, `cross_relative_lr_240`)

---

## 二、 全标的盲测结果汇总 (P99 Quantile, 日均 ~15 单)

| 标的币种 | 预测周期 | 总体盲测胜率 | 日均发单量 (单/天) | 坏月份数 (<55%) | 最差单月胜率 | 特点说明 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ETH** | **H = 15m** | **`59.64%`** | **14.77 单/天** | ★ **0 个** | **55.08%** | **实现 0 坏月硬约束！** |
| **ETH** | **H = 30m** | **`57.60%`** | **14.81 单/天** | **4 个** | `48.21%` | 30m 形态噪声增加 |
| **BTC** | **H = 15m** | **`56.75%`** | **14.97 单/天** | **4 个** | `50.32%` | - |
| **BTC** | **H = 30m** | **`56.58%`** | **15.11 单/天** | **7 个** | `47.19%` | - |

---

## 三、 深度总结与对比

1. **ETH 15m 混合引擎成功达成 0 坏月硬约束**：
   * 当在微观 K 线形态的基础上补充大周期因果 Z-Score 趋势水位与跨资产动量后，**ETH 15m 实现了 100% 0 坏月**（最差单月胜率为 `55.08%`），整体胜率为 **`59.64%`**（日均 14.77 单）。

2. **多特征族全特征矩阵（59列基线特征）依然保持最高的综合胜率（61.8% ~ 62.1%）**：
   * 将多尺度波动率（rvol）、均线间距斜率（ema_slope）、周期编码（hour_sin/cos）以及全套 Stochastic 摆动指标整合后，系统总体胜率最高可稳定在 **`61.8% ~ 62.1%`**（如之前 `ALL_SYMBOLS_HORIZONS_REPORT.md` 所示）。

3. **结论**：
   * 大周期趋势上下文（Trend Context）对于消除微观 K 线形态假突破的坏月至关重要。
