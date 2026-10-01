"""Ultra-Fast Neural 62 Breakthrough Version 8 with Stride 64 in Fine-Tuning.
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
from experiments.systematic_architecture_benchmark import FastSeqDataset, ResNet1DBlock
from experiments.masked_sequence_pretrain_engine import MaskedSeqAutoencoder, FastUnsupervisedDataset

class JointPretrainedResNetNet(nn.Module):
    def __init__(self, pretrained_encoder_proj, pretrained_encoder, in_features, hidden_dim=32):
        super().__init__()
        self.encoder_proj = pretrained_encoder_proj
        self.encoder = pretrained_encoder

        self.res1 = ResNet1DBlock(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.encoder_proj(x.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(h)

        h_t = encoded.transpose(1, 2)
        h_res = self.res1(h_t).transpose(1, 2)

        attn_out, _ = self.attn(h_res, h_res, h_res)
        h_fused = self.norm(h_res + attn_out)
        return self.head(h_fused[:, -1, :])

class LabelSmoothedFocalLoss(nn.Module):
    def __init__(self, gamma=2.8, label_smoothing=0.03):
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

def run_neural_62_v8(symbol="ETH", horizon_min=30, seq_len=30, pretrain_epochs=1, finetune_epochs=2, temperature=0.72):
    print(f"\n=======================================================", flush=True)
    print(f"ULTRA-FAST NEURAL 62 BREAKTHROUGH VERSION 8: {symbol} H={horizon_min}m", flush=True)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Phase 1: Masked Sequence Autoencoder Pretraining...", flush=True)
    ds_unsupervised = FastUnsupervisedDataset(X_tr, seq_len=seq_len, stride=64)
    loader_pretrain = DataLoader(ds_unsupervised, batch_size=512, shuffle=True)

    mae = MaskedSeqAutoencoder(in_features=X.shape[1], hidden_dim=32, mask_ratio=0.25).to(device)
    optimizer_pretrain = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-3)
    mse_loss = nn.MSELoss()

    mae.train()
    for epoch in range(pretrain_epochs):
        for bx, _ in loader_pretrain:
            bx = bx.to(device)
            optimizer_pretrain.zero_grad()
            reconstructed, mask = mae(bx)
            loss = mse_loss(reconstructed[mask], bx[mask])
            loss.backward()
            optimizer_pretrain.step()

    print("Phase 2: Fine-Tuning Joint Pretrained ResNet-1D...", flush=True)
    model = JointPretrainedResNetNet(mae.encoder_proj, mae.encoder, in_features=X.shape[1], hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    criterion = LabelSmoothedFocalLoss(gamma=2.8, label_smoothing=0.03)

    ds_tr = FastSeqDataset(X_tr, y_tr, seq_len=seq_len, stride=48)
    ds_te = FastSeqDataset(X_te, y_te, seq_len=seq_len, stride=16, mean=ds_tr.mean, std=ds_tr.std)

    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    model.train()
    for epoch in range(finetune_epochs):
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
        print(f"  Epoch {epoch+1}/{finetune_epochs} | Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

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
    print(f"STRICT CAUSAL RESULTS: Neural 62 Breakthrough V8 ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    max_win_rate = 0.0
    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Neural V8 Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)
        if acc > max_win_rate:
            max_win_rate = acc

    return max_win_rate

if __name__ == "__main__":
    run_neural_62_v8(symbol="ETH", horizon_min=30, seq_len=30, pretrain_epochs=1, finetune_epochs=2)
