"""Optimized FAISS Cosine KNN Search with High-Dimension Feature Embeddings.

Combines L2-normalized relative OHLC pattern vectors with Z-score trend context features into FAISS Index Flat IP.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import faiss

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

def run_optimized_faiss_knn(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 30, top_k: int = 10):
    print(f"\n=======================================================", flush=True)
    print(f"Optimized FAISS Cosine KNN Engine ({symbol} H={horizon_min}m, K={top_k})", flush=True)
    print(f"=======================================================", flush=True)

    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    df_raw = pl.read_parquet(raw_path).sort("ts")
    df_ds = pl.read_parquet(ds_path).sort("ts")

    df = df_raw.join(df_ds, on="ts", suffix="_ds")

    close_p = df["close"].to_numpy().astype(np.float32)
    open_p = df["open"].to_numpy().astype(np.float32)
    high_p = df["high"].to_numpy().astype(np.float32)
    low_p = df["low"].to_numpy().astype(np.float32)

    y = df["label"].to_numpy().astype(np.float32)
    ts = df["ts"].to_numpy().astype(np.int64)

    from numpy.lib.stride_tricks import sliding_window_view

    close_win = sliding_window_view(close_p, window_shape=seq_len)
    open_win = sliding_window_view(open_p, window_shape=seq_len)
    high_win = sliding_window_view(high_p, window_shape=seq_len)
    low_win = sliding_window_view(low_p, window_shape=seq_len)

    c_t = close_win[:, -1:] + 1e-8

    o_rel = (open_win - c_t) / c_t
    h_rel = (high_win - c_t) / c_t
    l_rel = (low_win - c_t) / c_t
    c_rel = (close_win - c_t) / c_t

    # Vector concatenate
    vectors = np.concatenate([o_rel, h_rel, l_rel, c_rel], axis=1).astype(np.float32)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    n = len(vectors_norm)
    train_idx = int(n * 0.8)

    tr_stride = 8
    tr_indices = np.arange(0, train_idx, tr_stride)

    te_stride = 16
    te_indices = np.arange(0, len(vectors_norm) - train_idx, te_stride)

    vectors_tr = vectors_norm[tr_indices]
    vectors_te = vectors_norm[train_idx + te_indices]

    y_tr = y[seq_len - 1 : train_idx + seq_len - 1][tr_indices]
    y_te_all = y[train_idx + seq_len - 1:]
    ts_te_all = ts[train_idx + seq_len - 1:]

    dim = vectors_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_tr)

    print(f"Querying FAISS IndexFlatIP (Cosine) with {index.ntotal:,} historical patterns...", flush=True)
    t0 = time.time()
    distances, indices = index.search(vectors_te, top_k)
    weights = np.maximum(distances, 1e-5)
    neighbor_labels = y_tr[indices]

    weighted_p_sub = np.sum(neighbor_labels * weights, axis=1) / np.sum(weights, axis=1)
    print(f"Search completed in {time.time()-t0:.1f}s", flush=True)

    p_full = np.full(len(ts_te_all), 0.5, dtype=np.float32)
    p_full[te_indices] = weighted_p_sub

    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"FAISS COSINE KNN EVALUATION ({symbol} H={horizon_min}m, K={top_k})", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te_all, ts_te_all, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | FAISS Cosine Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

    acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te_all, ts_te_all, p_quantile=99.0)
    return {
        'symbol': symbol,
        'horizon_min': horizon_min,
        'top_k': top_k,
        'overall_acc': acc_p99,
        'daily_trades': tpd,
        'bad_m_count': bad_m,
        'worst_month_acc': min_a,
        'acc_m': acc_m
    }

if __name__ == "__main__":
    run_optimized_faiss_knn(symbol="ETH", horizon_min=15, seq_len=30, top_k=10)
    run_optimized_faiss_knn(symbol="ETH", horizon_min=30, seq_len=30, top_k=10)
    run_optimized_faiss_knn(symbol="BTC", horizon_min=15, seq_len=30, top_k=10)
    run_optimized_faiss_knn(symbol="BTC", horizon_min=30, seq_len=30, top_k=10)
