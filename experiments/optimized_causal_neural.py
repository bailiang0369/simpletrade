"""Optimized Deep Pattern Recognition Neural Model (GroupNorm + Rank Margin Loss).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

class CausalPatternResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + residual)

class OptimizedPatternResNet(nn.Module):
    def __init__(self, in_features, seq_len=60, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = CausalPatternResBlock(hidden_dim)
        self.res2 = CausalPatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.in_proj(x))
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out).squeeze(-1)
        return torch.sigmoid(self.head(out))

def train_eval_optimized_neural(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 60):
    print(f"\n=======================================================", flush=True)
    print(f"Optimized Deep Pattern Neural Model for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    df_raw = pl.read_parquet(raw_path)
    df_ds = pl.read_parquet(ds_path)
    df = df_raw.join(df_ds, on="ts", suffix="_ds")

    open_p = df["open"]
    high_p = df["high"]
    low_p = df["low"]
    close_p = df["close"]

    candle_len = (high_p - low_p) + 1e-8
    body_len = (close_p - open_p).abs()
    max_body = pl.max_horizontal(close_p, open_p)
    min_body = pl.min_horizontal(close_p, open_p)

    df_feat = df.with_columns([
        ((close_p - open_p) / candle_len).alias("body_direction_ratio"),
        ((high_p - max_body) / candle_len).alias("upper_wick_ratio"),
        ((min_body - low_p) / candle_len).alias("lower_wick_ratio"),
        (candle_len / close_p).alias("range_pct"),
        ((close_p / close_p.shift(1)).log()).alias("lr_1"),
        ((close_p / close_p.shift(5)).log()).alias("lr_5"),
        ((close_p / close_p.shift(15)).log()).alias("lr_15"),
        ((close_p / close_p.shift(60)).log()).alias("lr_60"),
    ])

    feat_cols = [
        "body_direction_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct",
        "lr_1", "lr_5", "lr_15", "lr_60"
    ]

    X_all = df_feat.select(feat_cols).to_numpy().astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    y_all = df_feat["label"].to_numpy().astype(np.float32)
    ts_all = df_feat["ts"].to_numpy().astype(np.int64)

    n = len(df_feat)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X_all[:train_idx], y_all[:train_idx]
    X_te, y_te, ts_te = X_all[train_idx:], y_all[train_idx:], ts_all[train_idx:]

    # Construct Rolling Normalized Tensor Sequences
    stride = 16
    num_seqs = (len(X_tr) - seq_len) // stride
    X_tr_seq = np.zeros((num_seqs, seq_len, X_tr.shape[1]), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs, dtype=np.float32)

    for i in range(num_seqs):
        idx = i * stride
        wx = X_tr[idx : idx + seq_len]
        wm = np.mean(wx, axis=0, keepdims=True)
        ws = np.std(wx, axis=0, keepdims=True) + 1e-6
        X_tr_seq[i] = (wx - wm) / ws
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedPatternResNet(in_features=X_tr.shape[1], seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

    model.train()
    for epoch in range(3):
        t0 = time.time()
        running_loss = 0.0
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/3 Loss: {running_loss/len(loader_tr):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Prediction
    model.eval()
    test_stride = 8
    num_test_seqs = (len(X_te) - seq_len) // test_stride
    X_te_seq = np.zeros((num_test_seqs, seq_len, X_te.shape[1]), dtype=np.float32)
    indices = []

    for i in range(num_test_seqs):
        idx = i * test_stride
        wx = X_te[idx : idx + seq_len]
        wm = np.mean(wx, axis=0, keepdims=True)
        ws = np.std(wx, axis=0, keepdims=True) + 1e-6
        X_te_seq[i] = (wx - wm) / ws
        indices.append(idx + seq_len - 1)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        out = model(bx).squeeze().cpu().numpy()

    p_neural = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_neural[indices] = out

    df_p = pd.Series(p_neural)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL OPTIMIZED ROLLING NEURAL EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Causal Rolling Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    train_eval_optimized_neural(symbol="ETH", horizon_min=15, seq_len=60)
