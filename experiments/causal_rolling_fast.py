"""Fast Causal Rolling Window Neural Engine.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

class RollingCausalPatternDataset(Dataset):
    def __init__(self, raw_path, ds_path, seq_len=60, stride=16, train_ratio=0.8, is_train=True):
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
        train_idx = int(n * train_ratio)

        if is_train:
            self.X = X_all[:train_idx]
            self.y = y_all[:train_idx]
            self.ts = ts_all[:train_idx]
        else:
            self.X = X_all[train_idx:]
            self.y = y_all[train_idx:]
            self.ts = ts_all[train_idx:]

        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (len(self.X) - seq_len) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len

        window_x = self.X[start:end].copy()

        # STRICT Causal Window-Level Rolling Standardization
        w_mean = np.mean(window_x, axis=0, keepdims=True)
        w_std = np.std(window_x, axis=0, keepdims=True) + 1e-6
        window_norm = (window_x - w_mean) / w_std

        return torch.tensor(window_norm, dtype=torch.float32), torch.tensor(self.y[end - 1], dtype=torch.float32)

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

class CausalPatternResNet(nn.Module):
    def __init__(self, in_features, seq_len=60, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = CausalPatternResBlock(hidden_dim)
        self.res2 = CausalPatternResBlock(hidden_dim)
        self.res3 = CausalPatternResBlock(hidden_dim)
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
        out = self.res3(out)
        out = self.pool(out).squeeze(-1)
        return torch.sigmoid(self.head(out))

def train_eval_causal_rolling_neural_fast(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 60):
    print(f"\n=======================================================", flush=True)
    print(f"Fast Causal Rolling Window Neural Model for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    ds_tr = RollingCausalPatternDataset(raw_path, ds_path, seq_len=seq_len, stride=16, is_train=True)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalPatternResNet(in_features=8, seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"Training Causal Rolling PatternResNet on {device}...", flush=True)
    model.train()
    for epoch in range(2):
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
        print(f"  Epoch {epoch+1}/2 Loss: {running_loss/len(loader_tr):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Fast Evaluate Test Set with Stride 8
    test_stride = 8
    ds_te = RollingCausalPatternDataset(raw_path, ds_path, seq_len=seq_len, stride=test_stride, is_train=False)
    loader_te = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=2)

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for bx, by in loader_te:
            bx = bx.to(device)
            out = model(bx).squeeze().cpu().numpy()
            test_preds.extend(out)
            test_targets.extend(by.numpy())

    preds_sub = np.array(test_preds)
    y_te_sub = np.array(test_targets)

    # Linear forward fill for test set array to align with test ts
    df_ds = pl.read_parquet(ds_path)
    n = len(df_ds)
    train_idx = int(n * 0.8)
    ts_te = df_ds["ts"].to_numpy()[train_idx:]
    y_te = df_ds["label"].to_numpy()[train_idx:]

    p_neural = np.full(len(ts_te), 0.5, dtype=np.float32)
    test_indices = np.arange(seq_len - 1, len(ts_te) - 1, test_stride)[:len(preds_sub)]
    p_neural[test_indices] = preds_sub

    df_p = pd.Series(p_neural)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL ROLLING WINDOW NEURAL EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Causal Rolling Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    train_eval_causal_rolling_neural_fast(symbol="ETH", horizon_min=15, seq_len=60)
