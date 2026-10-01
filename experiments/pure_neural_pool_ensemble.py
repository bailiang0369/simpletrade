"""Ultra-Fast Pure Neural Pool Ensemble Engine with Stride 48.
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
    def __init__(self, X, y, seq_len=30, stride=48, mean=None, std=None):
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

# Architecture 1: Masked Transformer
from experiments.masked_sequence_pretrain_engine import MaskedSeqAutoencoder, SupervisedFineTuner, FastUnsupervisedDataset

# Architecture 2: Deep TCN-ResNet
from experiments.full_non_tree_benchmark import DeepTCNResNet

# Architecture 3: WaveNet-BiGRU
from experiments.deep_wavenet_bigru import FastWaveNetBiGRU

# Architecture 4: ConvNeXt-V2 GRN
from experiments.standalone_convnext_v2_engine import StandaloneConvNeXtV2Engine

def run_pure_neural_pool_ensemble(symbol="ETH", horizon_min=30, seq_len=30):
    print(f"\n=======================================================", flush=True)
    print(f"ULTRA-FAST PURE NEURAL POOL ENSEMBLE: {symbol} H={horizon_min}m", flush=True)
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
    criterion = FocalBCELoss(gamma=2.5)

    # Model 1: Masked Transformer
    print("1. Training Family 1: Masked Pretrained Transformer...", flush=True)
    ds_unsupervised = FastUnsupervisedDataset(X_tr, seq_len=seq_len, stride=64)
    loader_pretrain = DataLoader(ds_unsupervised, batch_size=512, shuffle=True)

    mae = MaskedSeqAutoencoder(in_features=X.shape[1], hidden_dim=32, mask_ratio=0.25).to(device)
    opt_pre = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-3)
    mse_loss = nn.MSELoss()

    mae.train()
    for epoch in range(1):
        for bx, _ in loader_pretrain:
            bx = bx.to(device)
            opt_pre.zero_grad()
            reconstructed, mask = mae(bx)
            loss = mse_loss(reconstructed[mask], bx[mask])
            loss.backward()
            opt_pre.step()

    m1 = SupervisedFineTuner(mae.encoder_proj, mae.encoder, in_features=X.shape[1], hidden_dim=32).to(device)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=8e-4, weight_decay=1e-3)
    m1.train()
    for epoch in range(1):
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt1.zero_grad()
            loss = criterion(m1(bx).squeeze(-1), by)
            loss.backward()
            opt1.step()

    # Model 2: Deep TCN-ResNet
    print("2. Training Family 2: Deep TCN-ResNet...", flush=True)
    m2 = DeepTCNResNet(in_features=X.shape[1], hidden_dim=64).to(device)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=1e-3)
    m2.train()
    for epoch in range(1):
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt2.zero_grad()
            loss = criterion(m2(bx).squeeze(-1), by)
            loss.backward()
            opt2.step()

    # Model 3: WaveNet-BiGRU
    print("3. Training Family 3: WaveNet-BiGRU-Attention...", flush=True)
    m3 = FastWaveNetBiGRU(in_features=X.shape[1], hidden_dim=64, gru_dim=32).to(device)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=1e-3, weight_decay=1e-3)
    m3.train()
    for epoch in range(1):
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt3.zero_grad()
            loss = criterion(m3(bx).squeeze(-1), by)
            loss.backward()
            opt3.step()

    # Model 4: ConvNeXt-V2 GRN
    print("4. Training Family 4: ConvNeXt-V2 GRN...", flush=True)
    m4 = StandaloneConvNeXtV2Engine(in_features=X.shape[1], hidden_dim=64).to(device)
    opt4 = torch.optim.AdamW(m4.parameters(), lr=1e-3, weight_decay=1e-3)
    m4.train()
    for epoch in range(1):
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt4.zero_grad()
            loss = criterion(m4(bx).squeeze(-1), by)
            loss.backward()
            opt4.step()

    # Inference & Ensembling
    print("\nEvaluating Multi-Family Pure Neural Pool Ensemble...", flush=True)
    m1.eval(); m2.eval(); m3.eval(); m4.eval()

    p_full1 = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_full2 = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_full3 = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_full4 = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for bx, _, end_indices in loader_te:
            bx = bx.to(device)
            p_full1[end_indices.numpy()] = torch.sigmoid(m1(bx).squeeze(-1) / 0.75).cpu().numpy()
            p_full2[end_indices.numpy()] = torch.sigmoid(m2(bx).squeeze(-1) / 0.75).cpu().numpy()
            p_full3[end_indices.numpy()] = torch.sigmoid(m3(bx).squeeze(-1) / 0.75).cpu().numpy()
            p_full4[end_indices.numpy()] = torch.sigmoid(m4(bx).squeeze(-1) / 0.75).cpu().numpy()

    def process_p(p_arr):
        df_p = pd.Series(p_arr)
        df_p[df_p == 0.5] = np.nan
        return df_p.ffill().bfill().to_numpy()

    p1, p2, p3, p4 = process_p(p_full1), process_p(p_full2), process_p(p_full3), process_p(p_full4)
    p_pure_neural_pool = 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL PURE NEURAL POOL RESULTS ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_pure_neural_pool, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pure Neural Pool Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_pure_neural_pool_ensemble(symbol="ETH", horizon_min=30)
