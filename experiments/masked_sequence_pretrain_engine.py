"""Super Fast Self-Supervised Masked Sequence Engine using PyTorch Transformer & 1D Conv.
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

class FastUnsupervisedDataset(Dataset):
    def __init__(self, X, seq_len=30, stride=64, mean=None, std=None):
        self.X = X
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
        return torch.tensor(wx_norm, dtype=torch.float32), end_idx

class MaskedSeqAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_dim=32, mask_ratio=0.25):
        super().__init__()
        self.in_features = in_features
        self.mask_ratio = mask_ratio
        self.encoder_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, x):
        b, l, c = x.shape
        mask = torch.rand(b, l, device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        h = self.encoder_proj(x_masked.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(h)
        reconstructed = self.decoder(encoded)
        return reconstructed, mask

class SupervisedFineTuner(nn.Module):
    def __init__(self, pretrained_encoder_proj, pretrained_encoder, in_features, hidden_dim=32):
        super().__init__()
        self.encoder_proj = pretrained_encoder_proj
        self.encoder = pretrained_encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.encoder_proj(x.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(h)
        last_step = encoded[:, -1, :]
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

def run_masked_pretraining_experiment(symbol="ETH", horizon_min=30, seq_len=30, pretrain_epochs=1, finetune_epochs=2, temperature=0.7):
    print(f"\n=======================================================", flush=True)
    print(f"SUPER FAST MASKED AUTOENCODER NEURAL NET: {symbol} H={horizon_min}m", flush=True)
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

    ds_unsupervised = FastUnsupervisedDataset(X_tr, seq_len=seq_len, stride=64)
    loader_pretrain = DataLoader(ds_unsupervised, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae = MaskedSeqAutoencoder(in_features=X.shape[1], hidden_dim=32, mask_ratio=0.25).to(device)
    optimizer_pretrain = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-3)
    mse_loss = nn.MSELoss()

    print("Phase 1: Unsupervised Masked Pretraining on OHLC Sequences...", flush=True)
    mae.train()
    for epoch in range(pretrain_epochs):
        t0 = time.time()
        running_loss = 0.0
        for bx, _ in loader_pretrain:
            bx = bx.to(device)
            optimizer_pretrain.zero_grad()
            reconstructed, mask = mae(bx)
            loss = mse_loss(reconstructed[mask], bx[mask])
            loss.backward()
            optimizer_pretrain.step()
            running_loss += loss.item()
        print(f"  Pretrain Epoch {epoch+1}/{pretrain_epochs} | Recon MSE: {running_loss/len(loader_pretrain):.5f} ({time.time()-t0:.1f}s)", flush=True)

    print("Phase 2: Supervised Fine-Tuning on Directional Labels...", flush=True)
    fine_tuner = SupervisedFineTuner(mae.encoder_proj, mae.encoder, in_features=X.shape[1], hidden_dim=32).to(device)
    optimizer_finetune = torch.optim.AdamW(fine_tuner.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = FocalBCELoss(gamma=2.5)

    class FastSupervisedDataset(Dataset):
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

    ds_tr = FastSupervisedDataset(X_tr, y_tr, seq_len=seq_len, stride=32)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    fine_tuner.train()
    for epoch in range(finetune_epochs):
        t0 = time.time()
        running_loss = 0.0
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer_finetune.zero_grad()
            logits = fine_tuner(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer_finetune.step()
            running_loss += loss.item()
        print(f"  Finetune Epoch {epoch+1}/{finetune_epochs} | Focal Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Inference Test Set
    ds_te = FastSupervisedDataset(X_te, y_te, seq_len=seq_len, stride=16, mean=ds_tr.mean, std=ds_tr.std)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    fine_tuner.eval()
    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for bx, _, end_indices in loader_te:
            bx = bx.to(device)
            logits = fine_tuner(bx).squeeze(-1)
            probs = torch.sigmoid(logits / temperature).cpu().numpy()
            p_full[end_indices.numpy()] = probs

    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL MASKED PRETRAINING RESULTS ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Standalone Pretrained Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_masked_pretraining_experiment(symbol="ETH", horizon_min=30, seq_len=30, pretrain_epochs=1, finetune_epochs=2)
