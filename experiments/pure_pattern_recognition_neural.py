"""Standalone Pure Pattern Recognition Neural Models (PatternResNet CNN & Conv1D-LSTM/Transformer).

Focuses strictly on pure chart pattern recognition (OHLC 2D price window geometry, wicks, bodies, channel breakouts)
without decision trees, ensembles, or stacking.
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

# 1. Pure Chart Pattern ResNet (1D/2D Channel Convolution for Geometry & Breakouts)
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
    def __init__(self, in_features, seq_len=60, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = PatternResBlock(hidden_dim)
        self.res2 = PatternResBlock(hidden_dim)
        self.res3 = PatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
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

# 2. Pure Conv1D-LSTM Chart Pattern Network (Temporal Chart Progression)
class PureConvLSTM(nn.Module):
    def __init__(self, in_features, seq_len=60, hidden_dim=64, lstm_hidden=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        out_conv = self.relu(self.bn1(self.conv1(x_trans))).transpose(1, 2)
        out_lstm, _ = self.lstm(out_conv)
        last_hidden = out_lstm[:, -1, :]
        return torch.sigmoid(self.head(last_hidden))

# Lazy Sequence Dataset for PyTorch Memory Efficiency
class PurePatternDataset(Dataset):
    def __init__(self, X_arr, y_arr, seq_len=60, stride=5):
        self.X = X_arr
        self.y = y_arr
        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (len(X_arr) - seq_len) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        return torch.tensor(self.X[start:end], dtype=torch.float32), torch.tensor(self.y[end - 1], dtype=torch.float32)

def train_eval_pure_pattern_model(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 60, model_type: str = "resnet"):
    print(f"\n=======================================================", flush=True)
    print(f"Pure Chart Pattern Neural Model ({model_type.upper()}) for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    base_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(base_path):
        base_path = f"data/datasets/ds_{symbol}.parquet"

    df = pl.read_parquet(base_path)
    pattern_cols = [c for c in df.columns if c in [
        'open', 'high', 'low', 'close', 'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'candle_range_pct', 'lr_15', 'lr_120', 'z_30', 'z_60', 'rvol_30', 'stoch_k14', 'stoch_d14'
    ] or c.startswith('lr_') or c.startswith('z_') or c.startswith('pos_')]

    if len(pattern_cols) < 5:
        ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts']
        pattern_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Dataset rows: {len(df):,}, Pure Pattern Features count: {len(pattern_cols)}", flush=True)

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

    dataset_tr = PurePatternDataset(X_tr, y_tr, seq_len=seq_len, stride=8)
    loader_tr = DataLoader(dataset_tr, batch_size=256, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "resnet":
        model = PurePatternResNet(in_features=X_tr.shape[1], seq_len=seq_len).to(device)
    else:
        model = PureConvLSTM(in_features=X_tr.shape[1], seq_len=seq_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"Training {model_type.upper()} neural model on device: {device}...", flush=True)
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

    # Fast Predict on Test Set
    model.eval()
    print("Evaluating test set predictions...", flush=True)
    preds = np.full(len(X_te), 0.5, dtype=np.float32)
    batch_seqs = []
    indices = []

    for i in range(seq_len, len(X_te), 4):
        batch_seqs.append(X_te[i - seq_len : i])
        indices.append(i)
        if len(batch_seqs) >= 1024 or i >= len(X_te) - 4:
            with torch.no_grad():
                bx = torch.tensor(np.array(batch_seqs), dtype=torch.float32).to(device)
                out = model(bx).squeeze().cpu().numpy()
                preds[indices] = out
            batch_seqs = []
            indices = []

    df_p = pd.Series(preds)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"PURE PATTERN {model_type.upper()} CAUSAL EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pure Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

    acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=99.0)
    return {
        'symbol': symbol,
        'horizon_min': horizon_min,
        'model_type': model_type,
        'seq_len': seq_len,
        'overall_acc': acc_p99,
        'daily_trades': tpd,
        'bad_m_count': bad_m,
        'worst_month_acc': min_a,
        'acc_m': acc_m
    }

if __name__ == "__main__":
    run_resnet_15 = train_eval_pure_pattern_model(symbol="ETH", horizon_min=15, seq_len=60, model_type="resnet")
    run_convlstm_15 = train_eval_pure_pattern_model(symbol="ETH", horizon_min=15, seq_len=60, model_type="convlstm")
