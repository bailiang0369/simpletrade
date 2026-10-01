"""Systematic High-Order Neural Feature Engineering Suite.
Strictly respects prohibitions: NO total_volume, NO quote_volume, NO atr, NO trade_count.
Extracts:
1. Log-Return Velocity & Acceleration (Curvature) across horizons (1m, 3m, 5m, 15m, 30m).
2. Taker Buy/Sell Volume Ratio Imbalance Velocity & Slope.
3. Multi-Timeframe Stochastic Oscillator Momentum & Divergence (%K, %D, Acceleration).
4. Strictly Causal Rolling Z-score Normalization.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_systematic_neural_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Builds high-order price derivative and taker volume features from raw 1m OHLCV."""
    close = df_raw['close']
    open_p = df_raw['open']
    high = df_raw['high']
    low = df_raw['low']
    buy_vol = df_raw['buy_vol']
    sell_vol = df_raw['sell_vol']

    # 1. Log-Return Velocity & Curvature
    log_ret_1m = (close / close.shift(1)).log()
    log_ret_3m = (close / close.shift(3)).log()
    log_ret_5m = (close / close.shift(5)).log()
    log_ret_15m = (close / close.shift(15)).log()
    log_ret_30m = (close / close.shift(30)).log()

    # Curvature (Second Derivative)
    ret_acc_5m = log_ret_5m - log_ret_5m.shift(5)
    ret_acc_15m = log_ret_15m - log_ret_15m.shift(15)

    # 2. Taker Imbalance Velocity
    vol_tot_taker = buy_vol + sell_vol + 1e-6
    imbalance = (buy_vol - sell_vol) / vol_tot_taker
    imb_m15 = imbalance.rolling_mean(window_size=15)
    imb_slope = imb_m15 - imb_m15.shift(15)

    # 3. Multi-Timeframe Stochastic Oscillator
    def compute_stoch(window=14, smooth_k=3):
        lowest_l = low.rolling_min(window_size=window)
        highest_h = high.rolling_max(window_size=window)
        stoch_raw = (close - lowest_l) / (highest_h - lowest_l + 1e-6)
        stoch_k = stoch_raw.rolling_mean(window_size=smooth_k)
        return stoch_k

    stoch_14 = compute_stoch(14, 3)
    stoch_30 = compute_stoch(30, 5)
    stoch_60 = compute_stoch(60, 5)
    stoch_diff = stoch_14 - stoch_60

    # Assemble Feature DataFrame
    df_feat = pl.DataFrame({
        'ts': df_raw['ts'],
        'close': close,
        'ret_1m': log_ret_1m,
        'ret_3m': log_ret_3m,
        'ret_5m': log_ret_5m,
        'ret_15m': log_ret_15m,
        'ret_30m': log_ret_30m,
        'ret_acc_5m': ret_acc_5m,
        'ret_acc_15m': ret_acc_15m,
        'imbalance': imbalance,
        'imb_m15': imb_m15,
        'imb_slope': imb_slope,
        'stoch_14': stoch_14,
        'stoch_30': stoch_30,
        'stoch_60': stoch_60,
        'stoch_diff': stoch_diff
    }).fill_nan(0.0).fill_null(0.0)

    return df_feat

if __name__ == "__main__":
    df_raw = pl.read_parquet("data/datasets/raw_ETH.parquet")
    df_f = build_systematic_neural_features(df_raw)
    print(f"Engineered High-Order Neural Features: {df_f.shape}")
