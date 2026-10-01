"""Chart Geometry & Volatility Squeeze Feature Engineering Suite for Standalone Neural Model.
Strictly respects prohibitions: NO total_volume, NO quote_volume, NO atr, NO trade_count.
Extracts:
1. Upper/Lower Wick Ratios (K-line geometry).
2. Volatility Squeeze Factor (Standard Deviation Squeeze Ratio over 15m & 60m).
3. Taker Buy/Sell Skewness & Momentum Acceleration.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_geometry_squeeze_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    close = df_raw['close']
    open_p = df_raw['open']
    high = df_raw['high']
    low = df_raw['low']
    buy_vol = df_raw['buy_vol']
    sell_vol = df_raw['sell_vol']

    # K-line Wicks Geometry
    body_top = pl.max_horizontal(close, open_p)
    body_bottom = pl.min_horizontal(close, open_p)
    range_hl = (high - low) + 1e-6

    upper_wick = (high - body_top) / range_hl
    lower_wick = (body_bottom - low) / range_hl

    # Volatility Squeeze Factor
    ret_1m = (close / close.shift(1)).log()
    std_15m = ret_1m.rolling_std(window_size=15)
    std_60m = ret_1m.rolling_std(window_size=60) + 1e-6
    vol_squeeze = std_15m / std_60m

    # Taker Imbalance Momentum Acceleration
    vol_tot = buy_vol + sell_vol + 1e-6
    imbalance = (buy_vol - sell_vol) / vol_tot
    imb_vel_10m = imbalance.rolling_mean(window_size=10) - imbalance.rolling_mean(window_size=10).shift(10)
    imb_acc_10m = imb_vel_10m - imb_vel_10m.shift(10)

    df_feat = df_raw.select([
        pl.col('ts'),
        upper_wick.alias('upper_wick'),
        lower_wick.alias('lower_wick'),
        vol_squeeze.alias('vol_squeeze'),
        imb_vel_10m.alias('imb_vel_10m'),
        imb_acc_10m.alias('imb_acc_10m')
    ]).fill_nan(0.0).fill_null(0.0)

    return df_feat

if __name__ == "__main__":
    df_raw = pl.read_parquet("data/datasets/raw_ETH.parquet")
    df_g = build_geometry_squeeze_features(df_raw)
    print(f"Engineered Geometry & Squeeze Features Shape: {df_g.shape}")
