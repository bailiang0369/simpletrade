"""Pure Manual Trader Chart Pattern & Stochastic Setup Detector.

Encodes manual trader setup rules using ONLY Close/High/Low/Open price action and Stochastic %K/%D:
1. Stochastic %K/%D Extreme Zone Crosses (below 20 for Longs, above 80 for Shorts).
2. Pure Candlestick Pattern Reversals: Pinbar / Hammer (lower wick > 65%, body < 25%), Shooting Star, Bullish/Bearish Engulfing.
3. Key Support/Resistance Band Touches (reversal occurring at 20-period High/Low or Bollinger Band extremes).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

def detect_manual_pattern_setups(symbol: str = "ETH", horizon_min: int = 15):
    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    df_raw = pl.read_parquet(raw_path).sort("ts")
    df_ds = pl.read_parquet(ds_path).sort("ts")

    df = df_raw.join(df_ds, on="ts", suffix="_ds")

    open_p = pl.col("open")
    high_p = pl.col("high")
    low_p = pl.col("low")
    close_p = pl.col("close")

    candle_len = (high_p - low_p) + 1e-8
    body_len = (close_p - open_p).abs()
    max_body = pl.max_horizontal(close_p, open_p)
    min_body = pl.min_horizontal(close_p, open_p)

    upper_wick = high_p - max_body
    lower_wick = min_body - low_p

    # Stochastic %K and %D (14-period)
    high_max_14 = high_p.rolling_max(window_size=14)
    low_min_14 = low_p.rolling_min(window_size=14)

    stoch_k = (close_p - low_min_14) / (high_max_14 - low_min_14 + 1e-8) * 100.0
    stoch_d = stoch_k.rolling_mean(window_size=3)

    stoch_k_prev = stoch_k.shift(1)
    stoch_d_prev = stoch_d.shift(1)

    # 1. Stochastic Extreme Zone Crosses
    bull_stoch_cross = (stoch_k_prev <= stoch_d_prev) & (stoch_k > stoch_d) & (stoch_d <= 25.0)
    bear_stoch_cross = (stoch_k_prev >= stoch_d_prev) & (stoch_k < stoch_d) & (stoch_d >= 75.0)

    # 2. Pure Candlestick Pattern Reversals
    bull_pinbar = (lower_wick / candle_len > 0.60) & (body_len / candle_len < 0.30)
    bear_pinbar = (upper_wick / candle_len > 0.60) & (body_len / candle_len < 0.30)

    bull_engulfing = (close_p > open_p) & (close_p.shift(1) < open_p.shift(1)) & (close_p >= open_p.shift(1))
    bear_engulfing = (close_p < open_p) & (close_p.shift(1) > open_p.shift(1)) & (close_p <= open_p.shift(1))

    # 3. Support / Resistance Band Touch (20-period Low / High)
    touch_support = (low_p <= low_p.rolling_min(window_size=20).shift(1))
    touch_resistance = (high_p >= high_p.rolling_max(window_size=20).shift(1))

    # Manual Trader Setup Score: Sum of Setup Confirmations
    long_setup_score = (
        bull_stoch_cross.cast(pl.Float32) * 2.0 +
        bull_pinbar.cast(pl.Float32) * 1.5 +
        bull_engulfing.cast(pl.Float32) * 1.0 +
        touch_support.cast(pl.Float32) * 1.0
    )

    short_setup_score = (
        bear_stoch_cross.cast(pl.Float32) * 2.0 +
        bear_pinbar.cast(pl.Float32) * 1.5 +
        bear_engulfing.cast(pl.Float32) * 1.0 +
        touch_resistance.cast(pl.Float32) * 1.0
    )

    # Net Directional Probability Score
    net_score = long_setup_score - short_setup_score
    p_manual = 1.0 / (1.0 + np.exp(-net_score)) # Sigmoid transformation

    df_scored = df.with_columns([
        p_manual.alias("p_manual")
    ])

    return df_scored

def evaluate_manual_pattern_setup(symbol: str = "ETH", horizon_min: int = 15):
    print(f"\n=======================================================", flush=True)
    print(f"Manual Trader Chart Pattern & Stochastic Setup ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    df_scored = detect_manual_pattern_setups(symbol=symbol, horizon_min=horizon_min)

    n = len(df_scored)
    train_idx = int(n * 0.8)

    df_te = df_scored.slice(train_idx)

    p_manual = df_te["p_manual"].to_numpy().astype(np.float32)
    y_te = df_te["label"].to_numpy().astype(np.float32)
    ts_te = df_te["ts"].to_numpy().astype(np.int64)

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL MANUAL PATTERN SETUP EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_manual, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Manual Setup Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

    acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_manual, y_te, ts_te, p_quantile=99.0)
    return {
        'symbol': symbol,
        'horizon_min': horizon_min,
        'overall_acc': acc_p99,
        'daily_trades': tpd,
        'bad_m_count': bad_m,
        'worst_month_acc': min_a,
        'acc_m': acc_m
    }

if __name__ == "__main__":
    evaluate_manual_pattern_setup(symbol="ETH", horizon_min=15)
    evaluate_manual_pattern_setup(symbol="ETH", horizon_min=30)
    evaluate_manual_pattern_setup(symbol="BTC", horizon_min=15)
    evaluate_manual_pattern_setup(symbol="BTC", horizon_min=30)
