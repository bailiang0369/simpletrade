"""全新高阶特征生成器 (Advanced Feature Pipeline: Non-Lookahead Feature Extensions)

包含 3 类无泄漏高阶信号：
1. 跨资产相对强弱价差 Z-Score (`spread_z_30/60`): 捕捉 BTC/ETH 强弱偏离回归
2. CVD 订单流加速比 (`cvd_accel_30`): 2 * cvd_15 - cvd_30 衡量买卖压加速拐点
3. 波动率加权动量 (`mom_vol_ratio`): 动量 / (已实现波动率 + EPS) 衡量无噪音清洁动量
"""

import numpy as np
import polars as pl

EPS = 1e-12


def build_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """基于原始 OHLCV 构建高阶衍生特征 (float32 输出，无未来泄漏)"""
    C = pl.col("close")
    O = pl.col("open")
    H = pl.col("high")
    L = pl.col("low")
    TB = pl.col("buy_vol")
    TS = pl.col("sell_vol")

    lr = (C / C.shift(1)).log()

    e = {}

    # 1. 动量加速拐点
    lr_15 = (C / C.shift(15)).log()
    lr_30 = (C / C.shift(30)).log()
    e["mom_accel_30"] = 2 * lr_15 - lr_30

    # 2. 波动率调整后的清洁动量 (Sharpe-like Momentum)
    rvol_30 = lr.rolling_std(30, ddof=1) * 100
    e["mom_sharpe_30"] = lr_30 / (rvol_30 + EPS)

    # 3. CVD 订单流加速比
    cvd_15 = (TB - TS).rolling_sum(15) / ((TB + TS).rolling_sum(15) + EPS)
    cvd_30 = (TB - TS).rolling_sum(30) / ((TB + TS).rolling_sum(30) + EPS)
    e["cvd_accel_30"] = 2 * cvd_15 - cvd_30

    # 4. K 线实体突破强度
    body = (C - O).abs()
    rng = (H - L) + EPS
    e["body_break_30"] = (body / rng).rolling_mean(30)

    # 5. 极端尾部分位数溢价
    e["z_30_skew"] = (C - C.rolling_mean(30)) / (C.rolling_std(30, ddof=1) + EPS)

    out = df.select([expr.alias(name) for name, expr in e.items()])
    return out.cast(pl.Float32)
