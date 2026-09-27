"""Fast Pure Pattern Recognition Neural Models.
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

class PurePatternResNet(nn.Module):
    def __init__(self, in_features, seq_len=30, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = PatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.in_proj(x))
        out = self.res1(out)
        out = self.pool(out).squeeze(-1)
        return torch.sigmoid(self.head(out))

def train_eval_pure_pattern_fast(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 30):
    print(f"\n=======================================================", flush=True)
    print(f"Fast Pure Chart Pattern Neural Model (ResNet) for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    base_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(base_path):
        base_path = f"data/datasets/ds_{symbol}.parquet"

    df = pl.read_parquet(base_path)
    pattern_cols = [c for c in df.columns if c in [
        'open', 'high', 'low', 'close', 'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'candle_range_pct', 'lr_15', 'lr_120', 'z_30', 'z_60', 'rvol_30', 'stoch_k14', 'stoch_d14'
    ] or c.startswith('lr_') or c.startswith('z_') or c.startswith('pos_')]

    print(f"Dataset rows: {len(df):,}, Pattern Features count: {len(pattern_cols)}", flush=True)

    X = df.select(pattern_cols).to_numpy().astype(np.float32)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + 1e-6
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    stride = 16
    num_seqs = (len(X_tr) - seq_len) // stride
    X_tr_seq = np.zeros((num_seqs, seq_len, X_tr.shape[1]), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs, dtype=np.float32)
    for i in range(num_seqs):
        idx = i * stride
        X_tr_seq[i] = X_tr[idx : idx + seq_len]
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PurePatternResNet(in_features=X_tr.shape[1], seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

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

    model.eval()
    preds = np.full(len(X_te), 0.5, dtype=np.float32)
    test_stride = 8
    num_test_seqs = (len(X_te) - seq_len) // test_stride
    X_te_seq = np.zeros((num_test_seqs, seq_len, X_te.shape[1]), dtype=np.float32)
    indices = []
    for i in range(num_test_seqs):
        idx = i * test_stride
        X_te_seq[i] = X_te[idx : idx + seq_len]
        indices.append(idx + seq_len - 1)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        out = model(bx).squeeze().cpu().numpy()
        preds[indices] = out

    df_p = pd.Series(preds)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"PURE PATTERN RESNET CAUSAL EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pure ResNet Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    train_eval_pure_pattern_fast(symbol="ETH", horizon_min=15, seq_len=30)
