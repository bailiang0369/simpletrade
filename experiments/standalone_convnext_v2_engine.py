"""Memory-Efficient Polars Causal EMA & ConvNeXt-V2 GRN Engine.
Uses Polars `ewm_mean` for zero-copy memory-safe execution on 8GB RAM.
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

class FastCausalEmaDataset(Dataset):
    def __init__(self, X_norm, y, seq_len=30, stride=32):
        self.X_norm = X_norm
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.valid_indices = np.arange(seq_len - 1, len(X_norm), stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        wx = self.X_norm[start_idx : end_idx + 1]
        return torch.tensor(wx, dtype=torch.float32), torch.tensor(self.y[end_idx], dtype=torch.float32), end_idx

def compute_polars_causal_ema_norm(df_feats: pl.DataFrame, alpha=0.01) -> np.ndarray:
    exprs = []
    for col in df_feats.columns:
        m = pl.col(col).ewm_mean(alpha=alpha, adjust=False)
        v = ((pl.col(col) - m) ** 2).ewm_mean(alpha=alpha, adjust=False)
        s = v.sqrt() + 1e-5
        z = (pl.col(col) - m) / s
        exprs.append(z.fill_nan(0.0).fill_null(0.0).alias(col))

    df_norm = df_feats.select(exprs)
    return df_norm.to_numpy().astype(np.float32)

class GlobalResponseNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm1d(4 * dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return res + x

class AsymmetricMarginFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, margin=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_pos = torch.clamp(probs - self.margin, min=1e-6, max=1.0)
        p_neg = torch.clamp(1.0 - probs - self.margin, min=1e-6, max=1.0)
        loss_pos = -targets * ((1.0 - p_pos) ** self.gamma_pos) * torch.log(p_pos)
        loss_neg = -(1.0 - targets) * ((1.0 - p_neg) ** self.gamma_neg) * torch.log(p_neg)
        return torch.mean(loss_pos + loss_neg)

class StandaloneConvNeXtV2Engine(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.b1 = ConvNeXtV2Block(hidden_dim)
        self.b2 = ConvNeXtV2Block(hidden_dim)
        self.b3 = ConvNeXtV2Block(hidden_dim)
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

def run_standalone_neural_v2(symbol="ETH", horizon_min=30, seq_len=30, epochs=3, lr=1e-3, temperature=0.7):
    print(f"\n=======================================================", flush=True)
    print(f"FAST STANDALONE NON-TREE CONVNEXT-V2: {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    dataset_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(dataset_path):
        dataset_path = f"data/datasets/ds_{symbol}.parquet"

    df = pl.read_parquet(dataset_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    print("Computing Polars zero-copy causal EMA normalization...", flush=True)
    X_norm = compute_polars_causal_ema_norm(df.select(feat_cols), alpha=0.01)
    y = df['label'].to_numpy().astype(np.float32)
    ts = df['ts'].to_numpy().astype(np.int64)

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr_norm, y_tr = X_norm[:train_idx], y[:train_idx]
    X_te_norm, y_te, ts_te = X_norm[train_idx:], y[train_idx:], ts[train_idx:]

    ds_tr = FastCausalEmaDataset(X_tr_norm, y_tr, seq_len=seq_len, stride=32)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StandaloneConvNeXtV2Engine(in_features=len(feat_cols), hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = AsymmetricMarginFocalLoss(gamma_pos=2.0, gamma_neg=4.0, margin=0.05)

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
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Inference
    ds_te = FastCausalEmaDataset(X_te_norm, y_te, seq_len=seq_len, stride=16)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

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
    print(f"STRICT CAUSAL STANDALONE CONVNEXT-V2 RESULTS ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Standalone Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    for sym in ["ETH", "BTC"]:
        run_standalone_neural_v2(symbol=sym, horizon_min=30, seq_len=30, epochs=3, lr=1e-3, temperature=0.7)
