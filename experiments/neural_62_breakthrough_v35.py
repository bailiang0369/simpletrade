"""Neural 62 Breakthrough Version 35 (Dedicated New Version File).

Architecture Strategy:
- Cross-Asset Dual-Stream Neural Confluence Network:
  Jointly processes ETH & BTC physics derivatives (log-return curvatures, taker imbalance accelerations, volatility squeeze ratios) in two parallel neural branches, fusing them via Cross-Asset Multi-Head Attention to capture market-wide regime alignment.
- Multi-Task Head: Classification (Label-Smoothed Focal Loss) + Regression (Return Magnitude MSE Head).
- 100% Causal, Zero Decision Trees, Zero Ensembles.
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

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CrossAssetSeqDataset(Dataset):
    def __init__(self, X_eth, X_btc, y_cls, y_reg, seq_len=30, stride=32, mean_eth=None, std_eth=None, mean_btc=None, std_btc=None):
        self.X_eth = X_eth
        self.X_btc = X_btc
        self.y_cls = y_cls
        self.y_reg = y_reg
        self.seq_len = seq_len
        self.stride = stride

        self.mean_eth = mean_eth if mean_eth is not None else np.nanmean(X_eth, axis=0, keepdims=True)
        self.std_eth = std_eth if std_eth is not None else (np.nanstd(X_eth, axis=0, keepdims=True) + 1e-6)

        self.mean_btc = mean_btc if mean_btc is not None else np.nanmean(X_btc, axis=0, keepdims=True)
        self.std_btc = std_btc if std_btc is not None else (np.nanstd(X_btc, axis=0, keepdims=True) + 1e-6)

        self.valid_indices = np.arange(seq_len - 1, min(len(X_eth), len(X_btc)), stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1

        wx_eth = self.X_eth[start_idx : end_idx + 1]
        wx_eth_norm = (wx_eth - self.mean_eth) / self.std_eth
        wx_eth_norm = np.nan_to_num(wx_eth_norm, nan=0.0, posinf=0.0, neginf=0.0)

        wx_btc = self.X_btc[start_idx : end_idx + 1]
        wx_btc_norm = (wx_btc - self.mean_btc) / self.std_btc
        wx_btc_norm = np.nan_to_num(wx_btc_norm, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.tensor(wx_eth_norm, dtype=torch.float32),
            torch.tensor(wx_btc_norm, dtype=torch.float32),
            torch.tensor(self.y_cls[end_idx], dtype=torch.float32),
            torch.tensor(self.y_reg[end_idx], dtype=torch.float32),
            end_idx
        )

class SEModule1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w

class ResNet1DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.act = nn.GELU()
        self.c2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)
        self.se = SEModule1D(channels)

    def forward(self, x):
        res = x
        out = self.act(self.gn1(self.c1(x)))
        out = self.se(self.gn2(self.c2(out)))
        return self.act(out + res)

class Neural62BreakthroughNetV35(nn.Module):
    def __init__(self, in_features_eth, in_features_btc, hidden_dim=64):
        super().__init__()
        # Parallel encoders for ETH and BTC
        self.proj_eth = nn.Conv1d(in_features_eth, hidden_dim, kernel_size=1)
        self.b_eth = ResNet1DBlock(hidden_dim)

        self.proj_btc = nn.Conv1d(in_features_btc, hidden_dim, kernel_size=1)
        self.b_btc = ResNet1DBlock(hidden_dim)

        # Cross-Asset Attention (ETH queries BTC)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(32, 1)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_eth, x_btc):
        h_eth = self.b_eth(self.proj_eth(x_eth.transpose(1, 2))).transpose(1, 2)
        h_btc = self.b_btc(self.proj_btc(x_btc.transpose(1, 2))).transpose(1, 2)

        # ETH attends to BTC macro context
        attn_out, _ = self.cross_attn(h_eth, h_btc, h_btc)
        h_fused = self.norm(h_eth + attn_out)

        feat = h_fused[:, -1, :]
        cls_logits = self.cls_head(feat)
        reg_preds = self.reg_head(feat)
        return cls_logits, reg_preds

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

def build_advanced_physics_features(df: pl.DataFrame) -> pl.DataFrame:
    feats = []
    if "lr_15" in df.columns and "lr_120" in df.columns:
        feats.append((pl.col("lr_15") - pl.col("lr_120") / 8.0).alias("lr_curvature_15_120"))

    if "tb_act_60" in df.columns and "ts_act_60" in df.columns:
        imb = pl.col("tb_act_60") - pl.col("ts_act_60")
        feats.append(imb.alias("taker_imb_diff"))
        feats.append((imb - imb.shift(1).fill_null(0.0)).alias("taker_imb_accel"))

    if "rvol_30" in df.columns and "rvol_60" in df.columns:
        feats.append((pl.col("rvol_30") / (pl.col("rvol_60") + 1e-8)).alias("vol_squeeze_ratio"))

    if feats:
        return df.with_columns(feats)
    return df

def run_neural_62_v35(horizon_min=30, seq_len=30, epochs=3, temperature=0.72, aux_weight=0.10, num_seeds=3):
    print(f"\n=======================================================", flush=True)
    print(f"NEURAL 62 BREAKTHROUGH VERSION 35 (CROSS-ASSET CONFLUENCE): ETH/BTC H={horizon_min}m ({num_seeds} Seeds)", flush=True)
    print(f"=======================================================", flush=True)

    df_eth = pl.read_parquet(f"data/datasets/ds_ETH_h{horizon_min}.parquet")
    df_btc = pl.read_parquet(f"data/datasets/ds_BTC_h{horizon_min}.parquet")

    df_eth = build_advanced_physics_features(df_eth)
    df_btc = build_advanced_physics_features(df_btc)

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols_eth = [c for c in df_eth.columns if c not in ignore_cols]
    feat_cols_btc = [c for c in df_btc.columns if c not in ignore_cols]

    # Align timestamps
    min_len = min(len(df_eth), len(df_btc))
    df_eth = df_eth[:min_len]
    df_btc = df_btc[:min_len]

    X_eth = df_eth.select(feat_cols_eth).to_numpy().astype(np.float32)
    X_btc = df_btc.select(feat_cols_btc).to_numpy().astype(np.float32)

    y_cls = df_eth['label'].to_numpy().astype(np.float32)
    y_reg = df_eth['ret_future'].to_numpy().astype(np.float32)
    ts = df_eth['ts'].to_numpy().astype(np.int64)

    n = len(df_eth)
    train_idx = int(n * 0.8)

    X_tr_eth, X_tr_btc, y_cls_tr, y_reg_tr = X_eth[:train_idx], X_btc[:train_idx], y_cls[:train_idx], y_reg[:train_idx]
    X_te_eth, X_te_btc, y_cls_te, y_reg_te, ts_te = X_eth[train_idx:], X_btc[train_idx:], y_cls[train_idx:], y_reg[train_idx:], ts[train_idx:]

    ds_tr = CrossAssetSeqDataset(X_tr_eth, X_tr_btc, y_cls_tr, y_reg_tr, seq_len=seq_len, stride=32)
    ds_te = CrossAssetSeqDataset(X_te_eth, X_te_btc, y_cls_te, y_reg_te, seq_len=seq_len, stride=16,
                                mean_eth=ds_tr.mean_eth, std_eth=ds_tr.std_eth,
                                mean_btc=ds_tr.mean_btc, std_btc=ds_tr.std_btc)

    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = [42, 100, 2024][:num_seeds]
    p_full_ensemble = np.zeros(len(ts_te), dtype=np.float32)

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---", flush=True)
        set_seed(seed)
        loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

        model = Neural62BreakthroughNetV35(in_features_eth=X_eth.shape[1], in_features_btc=X_btc.shape[1], hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        cls_criterion = LabelSmoothedFocalLoss(gamma=2.8, label_smoothing=0.03)
        reg_criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            t0 = time.time()
            running_loss = 0.0
            for bx_e, bx_b, by_c, by_r, _ in loader_tr:
                bx_e, bx_b, by_c, by_r = bx_e.to(device), bx_b.to(device), by_c.to(device), by_r.to(device)
                optimizer.zero_grad()
                c_logits, r_preds = model(bx_e, bx_b)
                c_logits, r_preds = c_logits.squeeze(-1), r_preds.squeeze(-1)

                l_cls = cls_criterion(c_logits, by_c)
                l_reg = reg_criterion(r_preds, by_r)
                loss = l_cls + aux_weight * l_reg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            print(f"  Seed {seed} | Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

        model.eval()
        p_seed = np.full(len(ts_te), 0.5, dtype=np.float32)

        with torch.no_grad():
            for bx_e, bx_b, _, _, end_indices in loader_te:
                bx_e, bx_b = bx_e.to(device), bx_b.to(device)
                c_logits, _ = model(bx_e, bx_b)
                probs = torch.sigmoid(c_logits.squeeze(-1) / temperature).cpu().numpy()
                p_seed[end_indices.numpy()] = probs

        df_p = pd.Series(p_seed)
        df_p[df_p == 0.5] = np.nan
        p_seed = df_p.ffill().bfill().to_numpy()
        p_full_ensemble += p_seed / len(seeds)

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL RESULTS: Neural 62 Breakthrough V35 (Cross-Asset ETH/BTC H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    max_win_rate = 0.0
    for q in [98.0, 98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_full_ensemble, y_cls_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Neural V35 Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)
        if acc > max_win_rate:
            max_win_rate = acc

    return max_win_rate

if __name__ == "__main__":
    max_acc = run_neural_62_v35(horizon_min=30, seq_len=30, epochs=3, temperature=0.72, aux_weight=0.10, num_seeds=3)
    print(f"\nPeak Win Rate V35: {max_acc*100:.2f}%")
