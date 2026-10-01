"""EXP-04: Concise High-Information Oscillator Feature Engine

Implements non-redundant Stochastic Oscillators (14, 30), StochRSI, RSI(14), CCI(20), Williams %R(14),
ROC(15, 30, 60), Volatility Squeeze (Bollinger Band / Keltner Ratio), and Multi-Scale Stochastic Momentum Alignment.
"""

import os, sys, time
import numpy as np
import polars as pl

def build_concise_oscillator_features(df_raw: pl.DataFrame, horizon_min: int = 15) -> pl.DataFrame:
    df = df_raw.sort("ts")
    h_steps = horizon_min

    open_p = pl.col("open")
    high_p = pl.col("high")
    low_p = pl.col("low")
    close_p = pl.col("close")

    exprs = []

    # 1. Stochastic Oscillators for core standard windows [14, 30, 60]
    for w in [14, 30, 60]:
        high_max = high_p.rolling_max(window_size=w)
        low_min = low_p.rolling_min(window_size=w)

        stoch_k = (close_p - low_min) / (high_max - low_min + 1e-8) * 100.0
        stoch_d = stoch_k.rolling_mean(window_size=3)

        exprs.extend([
            stoch_k.alias(f"stoch_k_{w}"),
            stoch_d.alias(f"stoch_d_{w}"),
            (stoch_k - stoch_d).alias(f"stoch_kd_diff_{w}"),
            (stoch_k - stoch_k.shift(3)).alias(f"stoch_k_mom_{w}"),
            # Williams %R
            ((high_max - close_p) / (high_max - low_min + 1e-8) * -100.0).alias(f"will_r_{w}")
        ])

    # 2. RSI(14) and StochRSI(14)
    # Price changes
    change = close_p - close_p.shift(1)
    gain = pl.when(change > 0).then(change).otherwise(0.0).rolling_mean(window_size=14)
    loss = pl.when(change < 0).then(-change).otherwise(0.0).rolling_mean(window_size=14)
    rs = gain / (loss + 1e-8)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    rsi_min_14 = rsi_14.rolling_min(window_size=14)
    rsi_max_14 = rsi_14.rolling_max(window_size=14)
    stoch_rsi = (rsi_14 - rsi_min_14) / (rsi_max_14 - rsi_min_14 + 1e-8) * 100.0

    exprs.extend([
        rsi_14.alias("rsi_14"),
        stoch_rsi.alias("stoch_rsi_14"),
        (stoch_rsi - stoch_rsi.rolling_mean(window_size=3)).alias("stoch_rsi_diff_14")
    ])

    # 3. Commodity Channel Index (CCI 20)
    tp = (high_p + low_p + close_p) / 3.0
    tp_sma = tp.rolling_mean(window_size=20)
    mad = (tp - tp_sma).abs().rolling_mean(window_size=20)
    cci_20 = (tp - tp_sma) / (0.015 * mad + 1e-8)
    exprs.append(cci_20.alias("cci_20"))

    # 4. Volatility Squeeze (Bollinger Band Width / ATR)
    bb_mid = close_p.rolling_mean(window_size=20)
    bb_std = close_p.rolling_std(window_size=20)
    bb_width = (2.0 * bb_std) / (bb_mid + 1e-8)
    exprs.append(bb_width.alias("bb_width_20"))

    # 5. Candlestick Anatomy
    body_len = (close_p - open_p).abs()
    candle_len = (high_p - low_p) + 1e-8
    exprs.extend([
        (body_len / candle_len).alias("body_ratio"),
        ((high_p - pl.max_horizontal(close_p, open_p)) / candle_len).alias("upper_wick_ratio"),
        ((pl.min_horizontal(close_p, open_p) - low_p) / candle_len).alias("lower_wick_ratio"),
    ])

    df_feat = df.with_columns(exprs)

    # Multi-Oscillator Momentum Reversal Alignment Score
    # Score +1 when Stoch < 20 AND StochRSI < 20 AND CCI < -100
    # Score -1 when Stoch > 80 AND StochRSI > 80 AND CCI > 100
    score_bull = ((df_feat["stoch_k_14"] < 20.0) & (df_feat["stoch_rsi_14"] < 20.0) & (df_feat["cci_20"] < -100.0)).cast(pl.Float32)
    score_bear = ((df_feat["stoch_k_14"] > 80.0) & (df_feat["stoch_rsi_14"] > 80.0) & (df_feat["cci_20"] > 100.0)).cast(pl.Float32)
    alignment_score = (score_bull - score_bear).alias("oscillator_alignment_score")

    df_feat = df_feat.with_columns([alignment_score])

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
        print("Building EXP-04 Concise Oscillator Features for ETH H15...")
        df_h15 = build_concise_oscillator_features(df_raw, horizon_min=15)
        df_h15.write_parquet("data/datasets/ds_ETH_exp04_h15.parquet")
        print("EXP-04 H15 saved successfully. Shape:", df_h15.shape)

        print("Building EXP-04 Concise Oscillator Features for ETH H30...")
        df_h30 = build_concise_oscillator_features(df_raw, horizon_min=30)
        df_h30.write_parquet("data/datasets/ds_ETH_exp04_h30.parquet")
        print("EXP-04 H30 saved successfully. Shape:", df_h30.shape)
