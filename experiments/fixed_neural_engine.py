"""Fast High-Performance Pure Neural Model Engine.
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

class PatternResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class FixedPatternResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = PatternResBlock(hidden_dim)
        self.res2 = PatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.in_proj(x))
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out).squeeze(-1)
        return self.head(out)

def train_eval_fixed_neural_fast(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 30):
    print(f"\n=======================================================", flush=True)
    print(f"Fixed High-Performance Pure Neural ResNet for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    dataset_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(dataset_path):
        dataset_path = f"data/datasets/ds_{symbol}.parquet"

    df = pl.read_parquet(dataset_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df['label'].to_numpy().astype(np.float32)
    ts = df['ts'].to_numpy().astype(np.int64)

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Train-Set Global Feature Standardization
    feat_mean = np.nanmean(X_tr, axis=0, keepdims=True)
    feat_std = np.nanstd(X_tr, axis=0, keepdims=True) + 1e-6

    # Construct Train Sequences with Stride 32
    stride = 32
    num_seqs_tr = (len(X_tr) - seq_len) // stride
    X_tr_seq = np.zeros((num_seqs_tr, seq_len, X_tr.shape[1]), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs_tr, dtype=np.float32)

    for i in range(num_seqs_tr):
        idx = i * stride
        wx = X_tr[idx : idx + seq_len]
        X_tr_seq[i] = (wx - feat_mean) / feat_std
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    X_tr_seq = np.nan_to_num(X_tr_seq, nan=0.0, posinf=0.0, neginf=0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FixedPatternResNet(in_features=X_tr.shape[1], hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

    print(f"Training Fixed Neural ResNet on device: {device}...", flush=True)
    model.train()
    for epoch in range(4):
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
        print(f"  Epoch {epoch+1}/4 | Train Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Evaluate Test Set
    model.eval()
    test_stride = 16
    num_seqs_te = (len(X_te) - seq_len) // test_stride
    X_te_seq = np.zeros((num_seqs_te, seq_len, X_te.shape[1]), dtype=np.float32)
    indices_te = []

    for i in range(num_seqs_te):
        idx = i * test_stride
        wx = X_te[idx : idx + seq_len]
        X_te_seq[i] = (wx - feat_mean) / feat_std
        indices_te.append(idx + seq_len - 1)

    X_te_seq = np.nan_to_num(X_te_seq, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        logits = model(bx).squeeze(-1)
        p_out = torch.sigmoid(logits).cpu().numpy()

    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_full[indices_te] = p_out
    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL FIXED NEURAL RESNET EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Fixed Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    train_eval_fixed_neural_fast(symbol="ETH", horizon_min=15, seq_len=30)
