"""Lightweight Systematic Audit & Diagnostic Script for Neural Model Training Dynamics.
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

class SimplePatternResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv2(h1)))
        out = self.pool(h2).squeeze(-1)
        return torch.sigmoid(self.head(out))

def diagnose_neural_pipeline_fast():
    print("=========================================================")
    print("  SYSTEMATIC NEURAL PIPELINE DIAGNOSTIC & AUDIT ENGINE  ")
    print("=========================================================")

    dataset_path = "data/datasets/ds_ETH_h15.parquet"
    df = pl.read_parquet(dataset_path)

    feat_cols = [c for c in df.columns if c not in ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']]

    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df['label'].to_numpy().astype(np.float32)
    ts = df['ts'].to_numpy().astype(np.int64)

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    print(f"Train samples: {len(X_tr):,}, Test samples: {len(X_te):,}, Features count: {len(feat_cols)}")

    # 1. Feature Standardization using Train Statistics
    feat_mean = np.nanmean(X_tr, axis=0, keepdims=True)
    feat_std = np.nanstd(X_tr, axis=0, keepdims=True) + 1e-6

    seq_len = 30
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
    model = SimplePatternResNet(in_features=X_tr.shape[1], hidden_dim=32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset_tr = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)

    print("\n--- Diagnostic 1: Training Loss Convergence Curve ---")
    model.train()
    for epoch in range(5):
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

        avg_loss = running_loss / len(loader_tr)
        print(f"  Epoch {epoch+1}/5 | Average Train BCE Loss: {avg_loss:.5f} ({time.time()-t0:.1f}s)")

    # Diagnostic 2: Test Set Prediction Distribution
    print("\n--- Diagnostic 2: Test Set Probability Distribution Audit ---")
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
        p_out = model(bx).squeeze(-1).cpu().numpy()

    print(f"  Test Predictions Mean: {np.mean(p_out):.4f}, Std: {np.std(p_out):.4f}")
    print(f"  Test Predictions Min:  {np.min(p_out):.4f}, Max: {np.max(p_out):.4f}")
    print(f"  Test AUC (ROC AUC):    {roc_auc_score(y_te[indices_te], p_out):.4f}")

    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_full[indices_te] = p_out
    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    acc_p99, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=99.0)
    print(f"\n--- Diagnostic 3: Causal Backtest Result ---")
    print(f"  P99 Quantile Win Rate: {acc_p99*100:.2f}% | Daily Signals: {tpd:.2f} | Bad Months: {bad_m}")

if __name__ == "__main__":
    diagnose_neural_pipeline_fast()
