"""Fast FAISS KNN Colab Protocol Alignment & Discrepancy Diagnostic Script.
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

def debug_faiss_discrepancy_fast(symbol: str = "ETH", horizon_min: int = 15):
    print("=========================================================================")
    print(f"  FAISS KNN COLAB PROTOCOL DIAGNOSTIC ENGINE ({symbol} H={horizon_min}m)  ")
    print("=========================================================================")

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

    seq_len = 20
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

    lr_win = np.diff(np.log(close_win + 1e-8), axis=1)

    vectors_ohlc = np.concatenate([o_rel, h_rel, l_rel, c_rel], axis=1).astype(np.float32)
    vectors_lr = lr_win.astype(np.float32)

    vectors_ohlc_norm = vectors_ohlc / (np.linalg.norm(vectors_ohlc, axis=1, keepdims=True) + 1e-8)
    vectors_lr_norm = vectors_lr / (np.linalg.norm(vectors_lr, axis=1, keepdims=True) + 1e-8)

    n = len(vectors_ohlc)
    train_idx = int(n * 0.8)

    tr_stride = 8
    tr_indices = np.arange(0, train_idx, tr_stride)

    te_stride = 16
    te_indices = np.arange(0, len(vectors_ohlc) - train_idx, te_stride)

    y_tr = y[seq_len - 1 : train_idx + seq_len - 1][tr_indices]
    y_te = y[train_idx + seq_len - 1:]
    ts_te = ts[train_idx + seq_len - 1:]

    experiments = [
        ("OHLC Relative (IndexFlatL2)", vectors_ohlc, "L2"),
        ("OHLC Cosine Normalized (IndexFlatIP)", vectors_ohlc_norm, "IP"),
        ("Log Returns Cosine Normalized (IndexFlatIP)", vectors_lr_norm, "IP"),
    ]

    for exp_name, vecs, metric in experiments:
        vecs_tr = vecs[tr_indices]
        vecs_te = vecs[train_idx + te_indices]

        dim = vecs.shape[1]
        if metric == "IP":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(vecs_tr)

        for top_k in [1, 5, 10, 25]:
            distances, indices = index.search(vecs_te, top_k)

            if metric == "IP":
                weights = np.maximum(distances, 1e-5)
            else:
                weights = 1.0 / (distances + 1e-5)

            neighbor_labels = y_tr[indices]
            p_sub = np.sum(neighbor_labels * weights, axis=1) / np.sum(weights, axis=1)

            p_full = np.full(len(ts_te), 0.5, dtype=np.float32)
            p_full[te_indices] = p_sub
            df_p = pd.Series(p_full)
            df_p[df_p == 0.5] = np.nan
            p_full = df_p.ffill().bfill().to_numpy()

            raw_acc = ((p_sub >= 0.5) == y_te[te_indices]).mean()
            acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=99.0)

            print(f"[{exp_name:42s} | K={top_k:2d}] Raw Acc: {raw_acc*100:5.2f}% | P99 Win Rate: {acc_p99*100:5.2f}% | Signals/Day: {tpd:5.2f}")

if __name__ == "__main__":
    debug_faiss_discrepancy_fast(symbol="ETH", horizon_min=15)
