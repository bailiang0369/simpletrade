"""Cross-Asset Multi-Head Attention Non-Tree Neural Architecture.
Integrates joint ETH & BTC cross-market temporal interactions with Dilated TCN & Self-Attention.
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

class CrossAssetDataset(Dataset):
    def __init__(self, X_target, X_cross, y, seq_len=30, stride=24, mean_target=None, std_target=None, mean_cross=None, std_cross=None):
        self.X_target = X_target
        self.X_cross = X_cross
        self.y = y
        self.seq_len = seq_len
        self.stride = stride

        self.mean_t = mean_target if mean_target is not None else np.nanmean(X_target, axis=0, keepdims=True)
        self.std_t = std_target if std_target is not None else (np.nanstd(X_target, axis=0, keepdims=True) + 1e-6)

        self.mean_c = mean_cross if mean_cross is not None else np.nanmean(X_cross, axis=0, keepdims=True)
        self.std_c = std_cross if std_cross is not None else (np.nanstd(X_cross, axis=0, keepdims=True) + 1e-6)

        self.valid_indices = np.arange(seq_len - 1, min(len(X_target), len(X_cross)), stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1

        wx_t = (self.X_target[start_idx : end_idx + 1] - self.mean_t) / self.std_t
        wx_c = (self.X_cross[start_idx : end_idx + 1] - self.mean_c) / self.std_c

        wx_t = np.nan_to_num(wx_t, nan=0.0, posinf=0.0, neginf=0.0)
        wx_c = np.nan_to_num(wx_c, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(wx_t, dtype=torch.float32), torch.tensor(wx_c, dtype=torch.float32), torch.tensor(self.y[end_idx], dtype=torch.float32), end_idx

class DilatedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        padding = (3 - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.gn1 = nn.GroupNorm(4, out_ch)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.gn2 = nn.GroupNorm(4, out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + res)

class CrossAssetAttentionNet(nn.Module):
    def __init__(self, target_dim, cross_dim, hidden_dim=64):
        super().__init__()
        self.t_proj = nn.Conv1d(target_dim, hidden_dim, kernel_size=1)
        self.c_proj = nn.Conv1d(cross_dim, hidden_dim, kernel_size=1)

        self.t_res = nn.Sequential(DilatedResBlock(hidden_dim, hidden_dim, 1), DilatedResBlock(hidden_dim, hidden_dim, 2))
        self.c_res = nn.Sequential(DilatedResBlock(hidden_dim, hidden_dim, 1), DilatedResBlock(hidden_dim, hidden_dim, 2))

        # Cross-Attention: Target queries Cross Asset
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_t, x_c):
        # x_t: target asset, x_c: cross asset
        ht = self.t_res(self.t_proj(x_t.transpose(1, 2))).transpose(1, 2)
        hc = self.c_res(self.c_proj(x_c.transpose(1, 2))).transpose(1, 2)

        attn_out, _ = self.cross_attn(query=ht, key=hc, value=hc)
        h_fused = self.norm(ht + attn_out)

        return self.head(h_fused[:, -1, :])

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = ((1 - p_t) ** self.gamma) * bce
        return torch.mean(focal_loss)

def run_cross_asset_experiment(target_symbol="ETH", cross_symbol="BTC", horizon_min=30, seq_len=30, epochs=3, gamma=2.5, temperature=0.75):
    print(f"\n=======================================================", flush=True)
    print(f"Cross-Asset Attention Neural Net: Target={target_symbol}, Cross={cross_symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    df_t = pl.read_parquet(f"data/datasets/ds_{target_symbol}_h{horizon_min}.parquet")
    df_c = pl.read_parquet(f"data/datasets/ds_{cross_symbol}_h{horizon_min}.parquet")

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols_t = [c for c in df_t.columns if c not in ignore_cols]
    feat_cols_c = [c for c in df_c.columns if c not in ignore_cols]

    # Align by timestamp length
    min_len = min(len(df_t), len(df_c))
    df_t = df_t.slice(len(df_t) - min_len, min_len)
    df_c = df_c.slice(len(df_c) - min_len, min_len)

    X_t = df_t.select(feat_cols_t).to_numpy().astype(np.float32)
    X_c = df_c.select(feat_cols_c).to_numpy().astype(np.float32)
    y = df_t['label'].to_numpy().astype(np.float32)
    ts = df_t['ts'].to_numpy().astype(np.int64)

    train_idx = int(min_len * 0.8)

    X_t_tr, X_c_tr, y_tr = X_t[:train_idx], X_c[:train_idx], y[:train_idx]
    X_t_te, X_c_te, y_te, ts_te = X_t[train_idx:], X_c[train_idx:], y[train_idx:], ts[train_idx:]

    ds_tr = CrossAssetDataset(X_t_tr, X_c_tr, y_tr, seq_len=seq_len, stride=24)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossAssetAttentionNet(target_dim=X_t.shape[1], cross_dim=X_c.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = FocalBCELoss(gamma=gamma)

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        for b_xt, b_xc, by, _ in loader_tr:
            b_xt, b_xc, by = b_xt.to(device), b_xc.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(b_xt, b_xc).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Focal Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Evaluation
    ds_te = CrossAssetDataset(X_t_te, X_c_te, y_te, seq_len=seq_len, stride=12,
                              mean_target=ds_tr.mean_t, std_target=ds_tr.std_t,
                              mean_cross=ds_tr.mean_c, std_cross=ds_tr.std_c)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    model.eval()
    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for b_xt, b_xc, _, end_indices in loader_te:
            b_xt, b_xc = b_xt.to(device), b_xc.to(device)
            logits = model(b_xt, b_xc).squeeze(-1)
            probs = torch.sigmoid(logits / temperature).cpu().numpy()
            p_full[end_indices.numpy()] = probs

    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"CROSS-ASSET ATTENTION RESULTS ({target_symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Cross-Asset Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_cross_asset_experiment(target_symbol="ETH", cross_symbol="BTC", horizon_min=30, seq_len=30, epochs=3)
