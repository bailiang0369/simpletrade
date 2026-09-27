"""Autonomous Neural Architecture Optimizer for Pure Chart Pattern & Temporal Models.
"""

import os, sys, time, warnings, json
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
        self.gn1 = nn.GroupNorm(4, channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + residual)

class StandalonePatternResNet(nn.Module):
    def __init__(self, in_features, seq_len=60, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = PatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = torch.relu(self.in_proj(x))
        h = self.res1(h)
        h = self.pool(h).squeeze(-1)
        return torch.sigmoid(self.head(h))

def train_eval_standalone_neural_iteration(model_name: str, model, X_tr, y_tr, X_te, y_te, ts_te, seq_len=60, stride=32, epochs=2):
    print(f"\n>>> Running Neural Iteration: {model_name} (Window={seq_len}m) <<<", flush=True)

    num_seqs_tr = (len(X_tr) - seq_len) // stride
    X_tr_seq = np.zeros((num_seqs_tr, seq_len, X_tr.shape[1]), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs_tr, dtype=np.float32)

    for i in range(num_seqs_tr):
        idx = i * stride
        wx = X_tr[idx : idx + seq_len]
        wm = np.mean(wx, axis=0, keepdims=True)
        ws = np.std(wx, axis=0, keepdims=True) + 1e-6
        X_tr_seq[i] = (wx - wm) / ws
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    # Clean NaNs
    X_tr_seq = np.nan_to_num(X_tr_seq, nan=0.0, posinf=0.0, neginf=0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

    model.train()
    for epoch in range(epochs):
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
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(loader_tr):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Prediction
    model.eval()
    test_stride = 16
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

    X_te_seq = np.nan_to_num(X_te_seq, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        out = model(bx).squeeze(-1).cpu().numpy()

    p_neural = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_neural[indices] = out

    df_p = pd.Series(p_neural)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=99.0)
    print(f"  Result -> [{model_name}] P99 Win Rate: {acc_p99*100:.2f}%, Daily Trades: {tpd:.2f}, Bad Months: {bad_m}", flush=True)

    return {
        'model_name': model_name,
        'win_rate_p99': acc_p99,
        'daily_trades': tpd,
        'bad_months': bad_m,
        'worst_month_acc': min_a,
        'monthly_acc': acc_m
    }

if __name__ == "__main__":
    print("Autonomous Neural Optimizer Engine Ready.", flush=True)
