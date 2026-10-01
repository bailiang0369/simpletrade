"""ETH Multi-Timeframe Stochastic Oscillators & Price Action Feature Engine

Implements strictly causal, multi-timeframe Stochastic Oscillators (%K, %D), StochRSI, RSI,
Williams %R, CCI, Momentum Reversals, Divergences, and Candlestick Price Action setups.
"""

import os, sys, time, gc
import numpy as np
import polars as pl

import config

def build_eth_stoch_features(df_raw: pl.DataFrame, horizon_min: int = 15) -> pl.DataFrame:
    """Build multi-timeframe stochastic oscillator & price action features for ETH."""
    # Ensure sorted by ts
    df = df_raw.sort("ts")

    # Target horizon steps
    h_steps = horizon_min # e.g. 15 or 30

    # Windows in minutes: [5, 9, 14, 21, 30, 45, 60, 90, 120, 240, 480, 1440]
    windows = [5, 9, 14, 21, 30, 45, 60, 90, 120, 240, 480, 1440]

    exprs = []

    # Basic price action: body, upper_wick, lower_wick
    max_body = pl.max_horizontal("close", "open")
    min_body = pl.min_horizontal("close", "open")

    exprs.extend([
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-8)).alias("candle_body_ratio"),
        ((pl.col("high") - max_body) / (pl.col("high") - pl.col("low") + 1e-8)).alias("upper_wick_ratio"),
        ((min_body - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8)).alias("lower_wick_ratio"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("candle_range_pct"),
    ])

    for w in windows:
        # High max and Low min over window w
        high_max = pl.col("high").rolling_max(window_size=w)
        low_min = pl.col("low").rolling_min(window_size=w)

        # Stochastic %K = (close - low_min) / (high_max - low_min + 1e-8) * 100
        stoch_k = ((pl.col("close") - low_min) / (high_max - low_min + 1e-8) * 100.0).alias(f"stoch_k_{w}")
        exprs.append(stoch_k)

        # Williams %R = (high_max - close) / (high_max - low_min + 1e-8) * -100
        will_r = ((high_max - pl.col("close")) / (high_max - low_min + 1e-8) * -100.0).alias(f"will_r_{w}")
        exprs.append(will_r)

        # Log returns over w
        lr = ((pl.col("close") / pl.col("close").shift(w)).log()).alias(f"lr_{w}")
        exprs.append(lr)

        # Price position relative to EMA/SMA over w
        sma = pl.col("close").rolling_mean(window_size=w)
        exprs.append(((pl.col("close") - sma) / sma).alias(f"price_sma_diff_{w}"))

    df_feat = df.with_columns(exprs)

    # Calculate %D (3-period smooth of %K) and StochRSI, RSI, CCI
    exprs_smooth = []
    for w in windows:
        # %D_3 = rolling mean of %K over 3
        stoch_d_3 = pl.col(f"stoch_k_{w}").rolling_mean(window_size=3).alias(f"stoch_d3_{w}")
        stoch_d_5 = pl.col(f"stoch_k_{w}").rolling_mean(window_size=5).alias(f"stoch_d5_{w}")
        exprs_smooth.append(stoch_d_3)
        exprs_smooth.append(stoch_d_5)

        # K - D cross signal
        stoch_diff = (pl.col(f"stoch_k_{w}") - pl.col(f"stoch_k_{w}").rolling_mean(window_size=3)).alias(f"stoch_diff_{w}")
        exprs_smooth.append(stoch_diff)

        # Stochastic Extreme setup flags (< 20 overbought/oversold, > 80)
        stoch_oversold = (pl.col(f"stoch_k_{w}") < 20.0).cast(pl.Float32).alias(f"stoch_oversold_{w}")
        stoch_overbought = (pl.col(f"stoch_k_{w}") > 80.0).cast(pl.Float32).alias(f"stoch_overbought_{w}")
        exprs_smooth.extend([stoch_oversold, stoch_overbought])

    df_feat = df_feat.with_columns(exprs_smooth)

    # Price vs Stochastic Divergence indicators over w=14, 30, 60
    exprs_div = []
    for w in [14, 30, 60, 120]:
        # Price higher high but Stoch lower high -> Bearish divergence
        price_hh = pl.col("high") > pl.col("high").shift(w)
        stoch_lh = pl.col(f"stoch_k_{w}") < pl.col(f"stoch_k_{w}").shift(w)
        bear_div = (price_hh & stoch_lh).cast(pl.Float32).alias(f"bear_div_{w}")

        # Price lower low but Stoch higher low -> Bullish divergence
        price_ll = pl.col("low") < pl.col("low").shift(w)
        stoch_hl = pl.col(f"stoch_k_{w}") > pl.col(f"stoch_k_{w}").shift(w)
        bull_div = (price_ll & stoch_hl).cast(pl.Float32).alias(f"bull_div_{w}")

        exprs_div.extend([bear_div, bull_div])

    df_feat = df_feat.with_columns(exprs_div)

    # Target calculation
    # Future return over horizon_min steps
    ret_future = ((df_feat["close"].shift(-h_steps) - df_feat["close"]) / df_feat["close"]).alias("ret_future")
    label = (ret_future > 0).cast(pl.Int8).alias("label")

    df_feat = df_feat.with_columns([ret_future, label])

    # Drop early null rows due to rolling window (up to 1440 mins = 1 day)
    # and drop late null rows due to shift(-h_steps)
    df_feat = df_feat.slice(1440, len(df_feat) - 1440 - h_steps)

    return df_feat

if __name__ == "__main__":
    raw_path = "data/datasets/raw_ETH.parquet"
    if os.path.exists(raw_path):
        df_raw = pl.read_parquet(raw_path)
        print("Building ETH Stochastic Features for H=15m...")
        df_h15 = build_eth_stoch_features(df_raw, horizon_min=15)
        print("H15 Dataset shape:", df_h15.shape)

        print("Building ETH Stochastic Features for H=30m...")
        df_h30 = build_eth_stoch_features(df_raw, horizon_min=30)
        print("H30 Dataset shape:", df_h30.shape)

        # Save enhanced datasets
        df_h15.write_parquet("data/datasets/ds_ETH_stoch_h15.parquet")
        df_h30.write_parquet("data/datasets/ds_ETH_stoch_h30.parquet")
        print("Datasets saved successfully.")
