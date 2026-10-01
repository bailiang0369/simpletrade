"""Fast Optimized WaveNet-BiGRU Non-Tree Experiment Script.
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
    def __init__(self, X, y, seq_len=30, stride=24, mean=None, std=None):
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

class WaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn = nn.GroupNorm(4, channels)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        h = f * g
        h = self.gn(h)
        return x + self.proj_out(h)

class FastWaveNetBiGRU(nn.Module):
    def __init__(self, in_features, hidden_dim=64, gru_dim=32):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.w1 = WaveNetBlock(hidden_dim, dilation=1)
        self.w2 = WaveNetBlock(hidden_dim, dilation=2)
        self.w3 = WaveNetBlock(hidden_dim, dilation=4)

        self.gru = nn.GRU(hidden_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=gru_dim * 2, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(gru_dim * 2)

        self.head = nn.Sequential(
            nn.Linear(gru_dim * 2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h = self.in_proj(x_t)
        h = self.w1(h)
        h = self.w2(h)
        h = self.w3(h).transpose(1, 2)

        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        h_attn = self.norm(gru_out + attn_out)

        last_step = h_attn[:, -1, :]
        return self.head(last_step)

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

def run_wavenet_experiment(symbol="ETH", horizon_min=30, seq_len=30, epochs=3, gamma=2.5, temperature=0.75):
    print(f"\n=======================================================", flush=True)
    print(f"Fast WaveNet-BiGRU: {symbol} H={horizon_min}m (SeqLen={seq_len}, Gamma={gamma}, Temp={temperature})", flush=True)
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

    ds_tr = FastSeqDataset(X_tr, y_tr, seq_len=seq_len, stride=24)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastWaveNetBiGRU(in_features=X_tr.shape[1], hidden_dim=64, gru_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = FocalBCELoss(gamma=gamma)

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
        print(f"  Epoch {epoch+1}/{epochs} | Focal Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Set Evaluation
    ds_te = FastSeqDataset(X_te, y_te, seq_len=seq_len, stride=12, mean=ds_tr.mean, std=ds_tr.std)
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
    print(f"EVALUATION: {symbol} H={horizon_min}m (Fast WaveNet-BiGRU)", flush=True)
    print(f"=======================================================", flush=True)

    res = {}
    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pure Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)
        res[q] = (acc, tpd, min_a, bad_m)

    return res

if __name__ == "__main__":
    for sym in ["ETH", "BTC"]:
        run_wavenet_experiment(symbol=sym, horizon_min=30, seq_len=30, epochs=3, gamma=2.5, temperature=0.75)
