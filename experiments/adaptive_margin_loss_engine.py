"""Margin-Aware Adaptive Focal Loss with Asymmetric Calibration for Deep Neural Nets.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily
from experiments.systematic_architecture_benchmark import FastSeqDataset, DeepResNet1D

class AdaptiveMarginFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=3.5, margin=0.08, label_smoothing=0.04):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.eps) + 0.5 * self.eps
        probs = torch.sigmoid(logits)

        p_pos = torch.clamp(probs - self.margin, min=1e-6, max=1.0)
        p_neg = torch.clamp(1.0 - probs - self.margin, min=1e-6, max=1.0)

        bce_pos = -targets_smooth * torch.log(p_pos)
        bce_neg = -(1.0 - targets_smooth) * torch.log(p_neg)

        focal_pos = ((1.0 - p_pos) ** self.gamma_pos) * bce_pos
        focal_neg = ((1.0 - p_neg) ** self.gamma_neg) * bce_neg

        return torch.mean(focal_pos + focal_neg)

def run_adaptive_margin_loss_experiment(symbol="ETH", horizon_min=30, seq_len=30, epochs=3, temperature=0.65):
    print(f"\n=======================================================", flush=True)
    print(f"ADAPTIVE MARGIN FOCAL LOSS TUNING: {symbol} H={horizon_min}m", flush=True)
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

    ds_tr = FastSeqDataset(X_tr, y_tr, seq_len=seq_len, stride=48)
    ds_te = FastSeqDataset(X_te, y_te, seq_len=seq_len, stride=16, mean=ds_tr.mean, std=ds_tr.std)

    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepResNet1D(in_features=X.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AdaptiveMarginFocalLoss(gamma_pos=2.0, gamma_neg=3.5, margin=0.08, label_smoothing=0.04)

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for bx, _, end_indices in loader_te:
            bx = bx.to(device)
            logits = model(bx).squeeze(-1)
            probs = torch.sigmoid(logits / temperature).cpu().numpy()
            p_full[end_indices.numpy()] = probs

    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"--- RESULTS: Adaptive Margin Focal Loss ---", flush=True)
    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pure Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_adaptive_margin_loss_experiment(symbol="ETH", horizon_min=30, seq_len=30, epochs=3)
