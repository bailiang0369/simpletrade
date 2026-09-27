"""Causal Trend-Context + Local K-Line Geometry + Cross-Asset Hybrid Feature Engine.

Strictly causal:
1. Local K-line geometry (scale-invariant wicks, bodies, candle ranges)
2. Causal HTF Trend-Context Z-Scores (distance to 60m, 240m, 1440m MAs, rolling 240m volatility)
3. Causal Cross-Asset Relative Momentum Z-Scores (BTC vs ETH relative strength)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl

def build_causal_hybrid_trend_geometry_features(symbol: str = "ETH", horizon_min: int = 15) -> pl.DataFrame:
    raw_path = f"data/datasets/raw_{symbol}.parquet"
    btc_raw_path = "data/datasets/raw_BTC.parquet"

    df_raw = pl.read_parquet(raw_path).sort("ts")
    df_btc = pl.read_parquet(btc_raw_path).sort("ts")

    # Local K-line Geometry (Scale-invariant)
    open_p = df_raw["open"]
    high_p = df_raw["high"]
    low_p = df_raw["low"]
    close_p = df_raw["close"]

    candle_len = (high_p - low_p) + 1e-8
    body_len = (close_p - open_p).abs()
    max_body = pl.max_horizontal(close_p, open_p)
    min_body = pl.min_horizontal(close_p, open_p)

    exprs = [
        ((close_p - open_p) / candle_len).alias("geom_body_ratio"),
        ((high_p - max_body) / candle_len).alias("geom_upper_wick"),
        ((min_body - low_p) / candle_len).alias("geom_lower_wick"),
        (candle_len / close_p).alias("geom_range_pct"),
    ]

    # Causal HTF Trend-Context Z-Scores (60m, 240m, 1440m)
    for w in [15, 60, 240, 1440]:
        sma_w = close_p.rolling_mean(window_size=w)
        std_w = close_p.rolling_std(window_size=w) + 1e-8
        z_w = (close_p - sma_w) / std_w
        lr_w = (close_p / close_p.shift(w)).log()

        exprs.extend([
            z_w.alias(f"causal_z_{w}"),
            lr_w.alias(f"causal_lr_{w}")
        ])

    df_feats = df_raw.with_columns(exprs)

    # Causal Cross-Asset BTC Relative Strength
    btc_close = df_btc["close"]
    btc_lr_15 = (btc_close / btc_close.shift(15)).log()
    btc_lr_60 = (btc_close / btc_close.shift(60)).log()
    btc_lr_240 = (btc_close / btc_close.shift(240)).log()

    df_btc_feats = df_btc.select([
        pl.col("ts"),
        btc_lr_15.alias("btc_causal_lr_15"),
        btc_lr_60.alias("btc_causal_lr_60"),
        btc_lr_240.alias("btc_causal_lr_240")
    ])

    df_joint = df_feats.join(df_btc_feats, on="ts")

    # Relative Spread / Spread Momentum
    df_joint = df_joint.with_columns([
        (pl.col("causal_lr_15") - pl.col("btc_causal_lr_15")).alias("cross_relative_lr_15"),
        (pl.col("causal_lr_60") - pl.col("btc_causal_lr_60")).alias("cross_relative_lr_60"),
        (pl.col("causal_lr_240") - pl.col("btc_causal_lr_240")).alias("cross_relative_lr_240"),
    ])

    # Target calculation
    ret_future = ((df_joint["close"].shift(-horizon_min) - df_joint["close"]) / df_joint["close"]).alias("ret_future")
    label = (ret_future > 0).cast(pl.Int8).alias("label")

    df_joint = df_joint.with_columns([ret_future, label])

    # Slice early and late null rows
    df_valid = df_joint.slice(1440, len(df_joint) - 1440 - horizon_min)
    return df_valid

if __name__ == "__main__":
    for sym in ["ETH", "BTC"]:
        for h in [15, 30]:
            print(f"Building Causal Trend-Geometry Hybrid Dataset for {sym} H={h}m...", flush=True)
            df_hybrid = build_causal_hybrid_trend_geometry_features(symbol=sym, horizon_min=h)
            out_path = f"data/datasets/ds_{sym}_hybrid_h{h}.parquet"
            df_hybrid.write_parquet(out_path)
            print(f"Saved {out_path}, shape: {df_hybrid.shape}", flush=True)
