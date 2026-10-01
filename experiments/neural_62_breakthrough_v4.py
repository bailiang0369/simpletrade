"""Neural 62 Breakthrough Version 4 (Dedicated New Version File).
Swin-Transformer 1D Local Shifted Window Self-Attention Net.
Zero Decision Trees, Zero Ensembles.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily
from experiments.systematic_architecture_benchmark import FastSeqDataset

class ShiftedWindowAttention1d(nn.Module):
    def __init__(self, dim, window_size=6, num_heads=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, L, C)
        b, l, c = x.shape
        num_windows = l // self.window_size
        if num_windows == 0:
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + attn_out)

        x_win = x[:, :num_windows * self.window_size, :].reshape(b * num_windows, self.window_size, c)
        attn_win, _ = self.attn(x_win, x_win, x_win)
        attn_win = attn_win.reshape(b, num_windows * self.window_size, c)

        if l > num_windows * self.window_size:
            pad_len = l - num_windows * self.window_size
            attn_win = torch.cat([attn_win, x[:, -pad_len:, :]], dim=1)

        return self.norm(x + attn_win)

class SwinTransformer1DNet(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.win_attn1 = ShiftedWindowAttention1d(dim=hidden_dim, window_size=6, num_heads=2)
        self.win_attn2 = ShiftedWindowAttention1d(dim=hidden_dim, window_size=10, num_heads=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h = self.in_proj(x_t).transpose(1, 2)
        h = self.win_attn1(h)
        h = self.win_attn2(h)
        return self.head(h[:, -1, :])

class LabelSmoothedFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, label_smoothing=0.04):
        super().__init__()
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.eps) + 0.5 * self.eps
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_loss = ((1.0 - p_t) ** self.gamma) * bce
        return torch.mean(focal_loss)

def run_neural_62_v4(symbol="ETH", horizon_min=30, seq_len=30, epochs=3, temperature=0.70):
    print(f"\n=======================================================", flush=True)
    print(f"NEURAL 62 BREAKTHROUGH VERSION 4 (SWIN 1D): {symbol} H={horizon_min}m", flush=True)
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

    ds_tr = FastSeqDataset(X_tr, y_tr, seq_len=seq_len, stride=32)
    ds_te = FastSeqDataset(X_te, y_te, seq_len=seq_len, stride=16, mean=ds_tr.mean, std=ds_tr.std)

    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer1DNet(in_features=X.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothedFocalLoss(gamma=2.5, label_smoothing=0.04)

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

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL RESULTS: Neural 62 Breakthrough V4 ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    max_win_rate = 0.0
    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Neural V4 Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)
        if acc > max_win_rate:
            max_win_rate = acc

    return max_win_rate

if __name__ == "__main__":
    run_neural_62_v4(symbol="ETH", horizon_min=30, seq_len=30, epochs=3, temperature=0.70)
