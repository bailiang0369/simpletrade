"""Fast Dynamic Sub-sequence Pattern Neural Network Engine.
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

class MultiScaleSubsequenceConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.gn = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h3 = self.conv3(x)
        h5 = self.conv5(x)
        h7 = self.conv7(x)
        hd = self.conv_dilated(x)
        out = torch.cat([h3, h5, h7, hd], dim=1)
        return self.relu(self.gn(out))

class SubsequenceTemporalAttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        scores = self.attn_fc(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class DynamicSubsequencePatternNet(nn.Module):
    def __init__(self, in_features=2, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.block1 = MultiScaleSubsequenceConvBlock(hidden_dim, hidden_dim)
        self.block2 = MultiScaleSubsequenceConvBlock(hidden_dim, hidden_dim)
        self.attn_pool = SubsequenceTemporalAttentionPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        h0 = torch.relu(self.in_proj(x_trans))
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        h_seq = h2.transpose(1, 2)
        pooled = self.attn_pool(h_seq)
        return torch.sigmoid(self.head(pooled))

def build_subsequence_inputs(symbol: str = "ETH", horizon_min: int = 15):
    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    df_raw = pl.read_parquet(raw_path).sort("ts")
    df_ds = pl.read_parquet(ds_path).sort("ts")

    high_max_29 = df_raw["high"].rolling_max(window_size=29)
    low_min_29 = df_raw["low"].rolling_min(window_size=29)
    stoch_k29 = ((df_raw["close"] - low_min_29) / (high_max_29 - low_min_29 + 1e-8) * 100.0)

    df_raw = df_raw.with_columns([
        stoch_k29.alias("stoch_k29"),
        ((df_raw["close"] / df_raw["close"].shift(1)).log()).alias("lr_1")
    ])

    df = df_raw.join(df_ds, on="ts", suffix="_ds")

    lr_1 = df["lr_1"].to_numpy().astype(np.float32)
    stoch_k = df["stoch_k29"].to_numpy().astype(np.float32) / 100.0

    X = np.column_stack([lr_1, stoch_k]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.5, posinf=0.5, neginf=0.5)

    y = df["label"].to_numpy().astype(np.float32)
    ts = df["ts"].to_numpy().astype(np.int64)

    return X, y, ts

def train_eval_subsequence_fast(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 30):
    print(f"\n=======================================================", flush=True)
    print(f"Dynamic Sub-sequence Pattern Neural Net for {symbol} H={horizon_min}m (SeqLen={seq_len})", flush=True)
    print(f"=======================================================", flush=True)

    X, y, ts = build_subsequence_inputs(symbol=symbol, horizon_min=horizon_min)

    n = len(X)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    stride = 32
    num_seqs_tr = (len(X_tr) - seq_len) // stride
    X_tr_seq = np.zeros((num_seqs_tr, seq_len, X_tr.shape[1]), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs_tr, dtype=np.float32)

    for i in range(num_seqs_tr):
        idx = i * stride
        X_tr_seq[i] = X_tr[idx : idx + seq_len]
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicSubsequencePatternNet(in_features=2, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

    print(f"Training Dynamic Sub-sequence Neural Model on device: {device}...", flush=True)
    model.train()
    for epoch in range(3):
        t0 = time.time()
        running_loss = 0.0
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/3 | Train Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    test_stride = 16
    num_seqs_te = (len(X_te) - seq_len) // test_stride
    X_te_seq = np.zeros((num_seqs_te, seq_len, X_te.shape[1]), dtype=np.float32)
    indices_te = []

    for i in range(num_seqs_te):
        idx = i * test_stride
        X_te_seq[i] = X_te[idx : idx + seq_len]
        indices_te.append(idx + seq_len - 1)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        out = model(bx).squeeze(-1).cpu().numpy()

    p_neural = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_neural[indices_te] = out

    df_p = pd.Series(p_neural)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"DYNAMIC SUB-SEQUENCE NEURAL CAUSAL EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Dynamic Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    train_eval_subsequence_fast(symbol="ETH", horizon_min=15, seq_len=30)
