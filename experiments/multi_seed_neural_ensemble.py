"""Ultra-Fast Multi-Seed Ensembled Standalone Deep Neural Engine for ETH.
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

class FastSeqDataset(Dataset):
    def __init__(self, X, y, seq_len=30, stride=32, mean=None, std=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.mean = mean if mean is not None else np.nanmean(X, axis=0, keepdims=True)
        self.std = std if std is not None else (np.nanstd(X, axis=0, keepdims=True) + 1e-6)
        self.valid_indices = np.arange(seq_len - 1, len(X), stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        wx = self.X[start_idx : end_idx + 1]
        wx_norm = (wx - self.mean) / self.std
        wx_norm = np.nan_to_num(wx_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(wx_norm, dtype=torch.float32), torch.tensor(self.y[end_idx], dtype=torch.float32), end_idx

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

class DeepTCNResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.b1 = DilatedResBlock(hidden_dim, hidden_dim, dilation=1)
        self.b2 = DilatedResBlock(hidden_dim, hidden_dim, dilation=2)
        self.b3 = DilatedResBlock(hidden_dim, hidden_dim, dilation=4)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h = self.in_proj(x_t)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h).transpose(1, 2)
        attn_out, _ = self.attn(h, h, h)
        h_attn = self.norm(h + attn_out)
        return self.head(h_attn[:, -1, :])

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

def run_seed_ensemble_neural(symbol="ETH", horizon_min=30, n_seeds=3, seq_len=30, epochs=3, temperature=0.75):
    print(f"\n=======================================================", flush=True)
    print(f"FAST MULTI-SEED PURE NEURAL ENSEMBLE ({n_seeds} Seeds): {symbol} H={horizon_min}m", flush=True)
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
    seed_preds = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + 42)
        np.random.seed(seed * 100 + 42)
        print(f"  Training Pure Neural Seed {seed+1}/{n_seeds}...", flush=True)

        model = DeepTCNResNet(in_features=X.shape[1], hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = FocalBCELoss(gamma=2.5)

        model.train()
        for epoch in range(epochs):
            for bx, by, _ in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx).squeeze(-1)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

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
        seed_preds.append(p_full)

    p_ensemble_neural = np.mean(seed_preds, axis=0)

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL MULTI-SEED PURE NEURAL RESULTS ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_ensemble_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Multi-Seed Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_seed_ensemble_neural(symbol="ETH", horizon_min=30, n_seeds=3, seq_len=30, epochs=3)
