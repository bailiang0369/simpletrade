"""High-Order Physics Momentum & Imbalance Acceleration Feature Suite.
Strictly respects prohibitions: NO total_volume, NO quote_volume, NO atr, NO trade_count.
Calculates:
1. Log-return Jerk (3rd derivative of log close price).
2. Taker Volume Imbalance Acceleration (2nd derivative of taker buy-sell ratio).
3. Relative Spread / K-line Body Expansion Ratio.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_physics_momentum_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    close = df_raw['close']
    open_p = df_raw['open']
    high = df_raw['high']
    low = df_raw['low']
    buy_vol = df_raw['buy_vol']
    sell_vol = df_raw['sell_vol']

    # Log Returns & Derivatives
    log_ret_1m = (close / close.shift(1)).log()
    log_ret_5m = (close / close.shift(5)).log()
    log_ret_15m = (close / close.shift(15)).log()

    # Acceleration (2nd derivative) & Jerk (3rd derivative)
    ret_acc_5m = log_ret_5m - log_ret_5m.shift(5)
    ret_jerk_5m = ret_acc_5m - ret_acc_5m.shift(5)

    # Imbalance & Acceleration
    vol_tot = buy_vol + sell_vol + 1e-6
    imbalance = (buy_vol - sell_vol) / vol_tot
    imb_vel_15m = imbalance.rolling_mean(window_size=15) - imbalance.rolling_mean(window_size=15).shift(15)
    imb_acc_15m = imb_vel_15m - imb_vel_15m.shift(15)

    # K-line Geometry Expansion
    body = (close - open_p).abs()
    range_hl = (high - low) + 1e-6
    body_ratio = body / range_hl
    body_expansion = body_ratio / (body_ratio.rolling_mean(window_size=30) + 1e-6)

    df_feat = pl.DataFrame({
        'ts': df_raw['ts'],
        'log_ret_1m': log_ret_1m,
        'log_ret_5m': log_ret_5m,
        'log_ret_15m': log_ret_15m,
        'ret_acc_5m': ret_acc_5m,
        'ret_jerk_5m': ret_jerk_5m,
        'imbalance': imbalance,
        'imb_vel_15m': imb_vel_15m,
        'imb_acc_15m': imb_acc_15m,
        'body_ratio': body_ratio,
        'body_expansion': body_expansion
    }).fill_nan(0.0).fill_null(0.0)

    return df_feat

if __name__ == "__main__":
    df_raw = pl.read_parquet("data/datasets/raw_ETH.parquet")
    df_p = build_physics_momentum_features(df_raw)
    print(f"Engineered High-Order Physics Momentum Features: {df_p.shape}")
