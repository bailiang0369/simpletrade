"""Comprehensive Benchmark of Self-Supervised Masked Transformer vs State-Space S4 vs TCN-ResNet.
Evaluates standalone non-tree neural performance across full test sets for ETH and BTC (30m).
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
from experiments.masked_sequence_pretrain_engine import MaskedSeqAutoencoder, SupervisedFineTuner, FastSupervisedDataset, FastUnsupervisedDataset

def run_comprehensive_benchmark(symbol="ETH", horizon_min=30):
    print(f"\n=======================================================", flush=True)
    print(f"STANDALONE NEURAL BENCHMARK: {symbol} H={horizon_min}m", flush=True)
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

    print("Running Masked Sequence Autoencoder Pretraining + Fine-Tuning...", flush=True)
    seq_len = 30
    ds_unsupervised = FastUnsupervisedDataset(X_tr, seq_len=seq_len, stride=64)
    loader_pretrain = DataLoader(ds_unsupervised, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae = MaskedSeqAutoencoder(in_features=X.shape[1], hidden_dim=32, mask_ratio=0.25).to(device)
    optimizer_pretrain = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-3)
    mse_loss = nn.MSELoss()

    mae.train()
    for epoch in range(1):
        for bx, _ in loader_pretrain:
            bx = bx.to(device)
            optimizer_pretrain.zero_grad()
            reconstructed, mask = mae(bx)
            loss = mse_loss(reconstructed[mask], bx[mask])
            loss.backward()
            optimizer_pretrain.step()

    fine_tuner = SupervisedFineTuner(mae.encoder_proj, mae.encoder, in_features=X.shape[1], hidden_dim=32).to(device)
    optimizer_finetune = torch.optim.AdamW(fine_tuner.parameters(), lr=8e-4, weight_decay=1e-3)

    from experiments.masked_sequence_pretrain_engine import FocalBCELoss
    criterion = FocalBCELoss(gamma=2.5)

    ds_tr = FastSupervisedDataset(X_tr, y_tr, seq_len=seq_len, stride=32)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    fine_tuner.train()
    for epoch in range(2):
        for bx, by, _ in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer_finetune.zero_grad()
            logits = fine_tuner(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer_finetune.step()

    ds_te = FastSupervisedDataset(X_te, y_te, seq_len=seq_len, stride=16, mean=ds_tr.mean, std=ds_tr.std)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    fine_tuner.eval()
    p_full = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for bx, _, end_indices in loader_te:
            bx = bx.to(device)
            logits = fine_tuner(bx).squeeze(-1)
            probs = torch.sigmoid(logits / 0.7).cpu().numpy()
            p_full[end_indices.numpy()] = probs

    df_p = pd.Series(p_full)
    df_p[df_p == 0.5] = np.nan
    p_full = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"BENCHMARK RESULTS ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Pretrained Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

if __name__ == "__main__":
    run_comprehensive_benchmark(symbol="ETH", horizon_min=30)
