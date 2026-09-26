"""ETH Stochastic Oscillator & Technical Confluence Engine

Builds explicit technical pattern setups:
1. Stochastic %K/%D Crossovers in Extreme Overbought/Oversold Zones (< 20, > 80) across 5m, 15m, 30m, 60m.
2. Candlestick Reversals: Pinbar / Hammer / Inverted Hammer / Engulfing / Long Wick Reversals.
3. Multi-Timeframe Alignment: HTF Trend (SMA 60m/240m slope) + LTF Stochastic Momentum Reversal.
"""

import os, sys, time
import numpy as np
import polars as pl

def build_eth_stoch_confluence_features(df_raw: pl.DataFrame, horizon_min: int = 15) -> pl.DataFrame:
    df = df_raw.sort("ts")
    h_steps = horizon_min

    # Base price action calculations
    open_p = pl.col("open")
    high_p = pl.col("high")
    low_p = pl.col("low")
    close_p = pl.col("close")

    body_len = (close_p - open_p).abs()
    candle_len = (high_p - low_p) + 1e-8
    upper_wick = high_p - pl.max_horizontal(close_p, open_p)
    lower_wick = pl.min_horizontal(close_p, open_p) - low_p

    exprs = [
        (body_len / candle_len).alias("body_ratio"),
        (upper_wick / candle_len).alias("upper_wick_ratio"),
        (lower_wick / candle_len).alias("lower_wick_ratio"),

        # Bullish Pinbar / Hammer: long lower wick (> 60%), small body (< 30%)
        ((lower_wick / candle_len > 0.6) & (body_len / candle_len < 0.3)).cast(pl.Float32).alias("bull_hammer"),
        # Bearish Shooting Star: long upper wick (> 60%), small body (< 30%)
        ((upper_wick / candle_len > 0.6) & (body_len / candle_len < 0.3)).cast(pl.Float32).alias("bear_shooting_star"),

        # Bullish Engulfing
        ((close_p > open_p) & (close_p.shift(1) < open_p.shift(1)) & (close_p > open_p.shift(1)) & (open_p < close_p.shift(1))).cast(pl.Float32).alias("bull_engulfing"),
        # Bearish Engulfing
        ((close_p < open_p) & (close_p.shift(1) > open_p.shift(1)) & (close_p < open_p.shift(1)) & (open_p > close_p.shift(1))).cast(pl.Float32).alias("bear_engulfing"),
    ]

    # Calculate Multi-Timeframe Stochastic %K, %D (3-period) over 5m, 14m, 30m, 60m, 120m
    windows = [5, 14, 30, 60, 120]
    for w in windows:
        high_max = high_p.rolling_max(window_size=w)
        low_min = low_p.rolling_min(window_size=w)

        stoch_k = (close_p - low_min) / (high_max - low_min + 1e-8) * 100.0
        stoch_d = stoch_k.rolling_mean(window_size=3)
        stoch_d_prev = stoch_d.shift(1)
        stoch_k_prev = stoch_k.shift(1)

        exprs.extend([
            stoch_k.alias(f"stoch_k_{w}"),
            stoch_d.alias(f"stoch_d_{w}"),

            # Stochastic Crossovers in Overbought/Oversold zones
            # Bullish Cross (%K crosses above %D while below 25)
            ((stoch_k_prev <= stoch_d_prev) & (stoch_k > stoch_d) & (stoch_d <= 25.0)).cast(pl.Float32).alias(f"stoch_bull_cross_os_{w}"),
            # Bearish Cross (%K crosses below %D while above 75)
            ((stoch_k_prev >= stoch_d_prev) & (stoch_k < stoch_d) & (stoch_d >= 75.0)).cast(pl.Float32).alias(f"stoch_bear_cross_ob_{w}"),

            # Momentum slope
            (stoch_k - stoch_k_prev).alias(f"stoch_k_slope_{w}"),
        ])

        # Moving Averages & HTF Trends
        sma_w = close_p.rolling_mean(window_size=w)
        exprs.append(((close_p - sma_w) / (sma_w + 1e-8)).alias(f"dist_sma_{w}"))

    df_feat = df.with_columns(exprs)

    # Confluence Signals: Stoch Bull Cross + Bullish Candle Reversal
    confluence_exprs = []
    for w in [5, 14, 30]:
        bull_confluence = ((pl.col(f"stoch_bull_cross_os_{w}") > 0) & ((pl.col("bull_hammer") > 0) | (pl.col("bull_engulfing") > 0))).cast(pl.Float32).alias(f"bull_confluence_{w}")
        bear_confluence = ((pl.col(f"stoch_bear_cross_ob_{w}") > 0) & ((pl.col("bear_shooting_star") > 0) | (pl.col("bear_engulfing") > 0))).cast(pl.Float32).alias(f"bear_confluence_{w}")
        confluence_exprs.extend([bull_confluence, bear_confluence])

    df_feat = df_feat.with_columns(confluence_exprs)

    # Target calculation
    ret_future = ((df_feat["close"].shift(-h_steps) - df_feat["close"]) / df_feat["close"]).alias("ret_future")
    label = (ret_future > 0).cast(pl.Int8).alias("label")

    df_feat = df_feat.with_columns([ret_future, label])

    # Drop early null rows
    df_feat = df_feat.slice(1440, len(df_feat) - 1440 - h_steps)
    return df_feat

if __name__ == "__main__":
    raw_path = "data/datasets/raw_ETH.parquet"
    if os.path.exists(raw_path):
        df_raw = pl.read_parquet(raw_path)
        print("Building ETH Technical Confluence Dataset for H15...")
        df_h15 = build_eth_stoch_confluence_features(df_raw, horizon_min=15)
        df_h15.write_parquet("data/datasets/ds_ETH_confluence_h15.parquet")
        print("Dataset H15 saved successfully. Shape:", df_h15.shape)

        print("Building ETH Technical Confluence Dataset for H30...")
        df_h30 = build_eth_stoch_confluence_features(df_raw, horizon_min=30)
        df_h30.write_parquet("data/datasets/ds_ETH_confluence_h30.parquet")
        print("Dataset H30 saved successfully. Shape:", df_h30.shape)
