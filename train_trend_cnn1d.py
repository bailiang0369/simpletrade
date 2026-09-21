"""事件合约 1D CNN: 多尺度 Stoch 序列 -> 固定 H 分钟后涨跌.
核心输入 = 完整连续序列 (保留趋势线的几何轨迹), 不是像素图, 内存需求是图像的 ~1/20.
窗口 W=120 (>=20 bar 趋势线起点要求), 通道 C=4 (原始/EMA10/EMA30/长尺度 Stoch120).
"""
import os
import numpy as np
import pandas as pd
import time, config
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

W = 120         # 窗口长度 (K 线)
C = 4           # 输入通道数
BATCH = 256
EPOCHS = 8

def stoch(k, d=1, smooth=1):
    eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
    close = eth['close'].values.astype(np.float64)
    high = eth['high'].values.astype(np.float64)
    low = eth['low'].values.astype(np.float64)
    ll = pd.Series(low).rolling(k, min_periods=k).min().values
    hh = pd.Series(high).rolling(k, min_periods=k).max().values
    sk = np.nan_to_num((close - ll) / np.maximum(hh - ll, 1e-6), nan=0.5)
    if smooth > 1: sk = pd.Series(sk).rolling(smooth, min_periods=1).mean().values
    if d > 1: sk = pd.Series(sk).rolling(d, min_periods=1).mean().values
    del eth, high, low, ll, hh
    return sk, close

def build_tensors(stoch_raw, close, ts, idx_arr, H, W):
    """给 idx_arr 每个点构建 (C, W) tensor + label. 返回 np.array."""
    n = len(idx_arr)
    X = np.zeros((n, C, W), np.float32)
    y = np.zeros(n, np.int64)
    # 预计算 4 通道
    ema10 = pd.Series(stoch_raw).ewm(span=10, adjust=False).mean().values
    ema30 = pd.Series(stoch_raw).ewm(span=30, adjust=False).mean().values
    s120, _ = stoch(120, 1, 1)
    N = len(close)
    for j, t in enumerate(idx_arr):
        if t - W + 1 < 0 or t + H >= N: continue
        X[j, 0] = stoch_raw[t - W + 1: t + 1]
        X[j, 1] = ema10[t - W + 1: t + 1]
        X[j, 2] = ema30[t - W + 1: t + 1]
        X[j, 3] = s120[t - W + 1: t + 1]
        y[j] = 1 if close[t + H] > close[t] else 0
    return X, y

class Trend1DCNN(nn.Module):
    def __init__(self, C=4, W=120):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv1d(C, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU()
        )
        self.net = nn.Sequential(
            stem,
            nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=4, dilation=4), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=8, dilation=8), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def main():
    t0 = time.time()
    s60, close = stoch(60, 1, 1)
    N = len(close); H = config.HORIZON_MIN
    ts = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet', columns=['ts'])['ts'].values.astype(np.int64)

    def mkmap(s, e):
        a = int(pd.Timestamp(s, tz='UTC').timestamp())
        b = int(pd.Timestamp(e, tz='UTC').timestamp())
        return (ts >= a) & (ts < b) & (np.arange(N) >= W) & (np.arange(N) < N - H)
    tr_m = mkmap(*config.SPLITS['train'])
    es_m = mkmap(*config.SPLITS['early_stop'])
    te_m = mkmap(*config.SPLITS['test'])

    rng = np.random.RandomState(config.SEED)
    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) > 80_000: tr_idx = rng.choice(tr_idx, 80_000, replace=False)
    es_idx = np.where(es_m)[0]
    te_idx = np.where(te_m)[0]
    print(f'[data] TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,} ({time.time()-t0:.0f}s)', flush=True)

    print('[build tensors]', flush=True)
    Xtr, ytr = build_tensors(s60, close, ts, tr_idx, H, W)
    Xes, yes = build_tensors(s60, close, ts, es_idx, H, W)
    print(f'  TR {Xtr.shape} ES {Xes.shape} ({time.time()-t0:.0f}s)', flush=True)

    dev = torch.device('cpu'); torch.manual_seed(config.SEED)
    model = Trend1DCNN(C=C, W=W).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr).to(dev); ytr_t = torch.tensor(ytr, dtype=torch.float32).to(dev)
    Xes_t = torch.tensor(Xes).to(dev); yes_t = torch.tensor(yes, dtype=torch.float32).to(dev)
    del Xtr, ytr, Xes

    best_auc = 0
    print(f'\n[train]', flush=True)
    for ep in range(EPOCHS):
        model.train(); tot = 0; nb = 0
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i + BATCH]
            xb, yb = Xtr_t[bi], ytr_t[bi]
            opt.zero_grad(); out = model(xb); loss = lossf(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(bi); nb += len(bi)
        model.eval()
        with torch.no_grad(): pv = torch.sigmoid(model(Xes_t)).numpy()
        auc = roc_auc_score(yes_t.numpy(), pv)
        k1 = max(1, int(len(pv) * 0.01)); a1 = yes_t.numpy()[pv.argsort()[-k1:]].mean()*100
        k05 = max(1, int(len(pv) * 0.005)); a05 = yes_t.numpy()[pv.argsort()[-k05:]].mean()*100
        print(f'  ep{ep}: loss={tot/nb:.4f} es_auc={auc:.4f} es_t0.5={a05:.1f}% es_t1={a1:.1f}% ({time.time()-t0:.0f}s)', flush=True)
        if auc > best_auc:
            best_auc = auc
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f'{config.MODEL_DIR}/trend_cnn1d.pt')

    # test 分批推理
    print(f'\n[test] 分批推理 TE={len(te_idx):,}', flush=True)
    model.load_state_dict(torch.load(f'{config.MODEL_DIR}/trend_cnn1d.pt', weights_only=True))
    model.eval()
    chunks = np.array_split(te_idx, max(1, len(te_idx) // 20000))
    pv_all = []; y_all = []
    for ci, chunk in enumerate(chunks):
        Xb, yb = build_tensors(s60, close, ts, chunk, H, W)
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xb).to(dev))).numpy()
        pv_all.append(pv); y_all.append(yb)
        del Xb
        if (ci + 1) % 10 == 0:
            print(f'  chunk {ci+1}/{len(chunks)} ({time.time()-t0:.0f}s)', flush=True)
    pv_all = np.concatenate(pv_all); y_all = np.concatenate(y_all)

    auc = roc_auc_score(y_all, pv_all)
    print(f'\n====== TEST (事件合约, H={H}min, 2025-09→2026-08, {len(y_all):,} pts) ======')
    print(f'AUC = {auc:.4f}  (历史最佳 LGBM 87feats = 0.544)')
    for pct in [0.005, 0.01, 0.02, 0.05, 0.1]:
        k = max(1, int(len(pv_all) * pct))
        si = np.argsort(pv_all)
        aL = y_all[si[-k:]].mean() * 100
        aS = (1 - y_all[si[:k]]).mean() * 100
        tpd = k / 333
        mark = ' ⭐' if aL >= 65 else ''
        print(f'  top {pct*100:>5.1f}%: long={aL:>5.1f}% ({tpd:.1f}tpd) | short={aS:>5.1f}%{mark}')
    print(f'\n目标: top1%≥65% + tpd≥14')
    print(f'⏱ {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()