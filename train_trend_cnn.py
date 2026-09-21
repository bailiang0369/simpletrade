"""图像 CNN 识别 Stoch 趋势线 — 低内存分批实现.

约束: 5GB 内存 / 3 核 CPU. 方案:
- 训练/验证: 渲染各 8万 / 2万 张图, float16 存盘 (~3GB) 可控.
- test: 分批渲染 + 分批推理, 不把 48 万张全量图装入内存.
- 图像设计: (4, H, W). 通道0=Stoch原始 1=EMA10 2=EMA30 3=价格轨迹.
  折线段加粗, 让卷积核"看到"趋势线的斜率/拐点.
"""
import os
import numpy as np
import pandas as pd
import time, config
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SEG = 80
HEIGHT = 56
LINE_W = 2

def stoch_series(close, high, low, k=60, d=1, smooth=1):
    ll = pd.Series(low).rolling(k, min_periods=k).min().values
    hh = pd.Series(high).rolling(k, min_periods=k).max().values
    sk = np.nan_to_num((close - ll) / np.maximum(hh - ll, 1e-6), nan=0.5)
    if smooth > 1: sk = pd.Series(sk).rolling(smooth, min_periods=1).mean().values
    if d > 1: sk = pd.Series(sk).rolling(d, min_periods=1).mean().values
    return sk

def render(stoch_win, price_win):
    img = np.zeros((4, HEIGHT, SEG), np.float32)
    def line(chan, vals):
        xs = np.linspace(0, SEG - 1, SEG).astype(int)
        vs = vals.astype(np.float64)
        mn, mx = vs.min(), vs.max()
        if mx - mn < 1e-9:
            ys = np.full(SEG, HEIGHT // 2).astype(int)
        else:
            ys = np.clip(((1 - (vs - mn) / (mx - mn)) * (HEIGHT - 1)).round().astype(int), 0, HEIGHT - 1)
        for i in range(SEG - 1):
            x0, x1 = xs[i], xs[i + 1]; y0, y1 = ys[i], ys[i + 1]
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in np.linspace(0, 1, int(n) + 1):
                xx, yy = int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0)))
                for dx in range(-LINE_W, LINE_W + 1):
                    for dy in range(-LINE_W, LINE_W + 1):
                        if 0 <= xx + dx < SEG and 0 <= yy + dy < HEIGHT:
                            img[chan, yy + dy, xx + dx] = 1.0
    line(0, stoch_win)
    e10 = pd.Series(stoch_win).ewm(span=10, adjust=False).mean().values
    e30 = pd.Series(stoch_win).ewm(span=30, adjust=False).mean().values
    line(1, e10); line(2, e30)
    line(3, price_win)
    return img

class TrendCNN(nn.Module):
    def __init__(self, in_ch=4, H=56, W=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def render_batch(idx_arr, slice_, stoch, price_norm, close, H):
    n = len(idx_arr)
    X = np.zeros((n, 4, HEIGHT, SEG), np.float32)
    y = np.zeros(n, np.int64)
    for j, i in enumerate(idx_arr):
        X[j] = render(stoch[i - SEG + 1: i + 1], price_norm[i - SEG + 1: i + 1])
        y[j] = 1 if close[i + H] > close[i] else 0
    return X, y

def main():
    t0 = time.time()
    eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
    close = eth['close'].values.astype(np.float64)
    high = eth['high'].values.astype(np.float64)
    low = eth['low'].values.astype(np.float64)
    ts = eth['ts'].values.astype(np.int64)
    N = len(close); H = config.HORIZON_MIN
    del eth

    stoch = stoch_series(close, high, low, k=60, d=1, smooth=1)
    price_roll = pd.Series(close).rolling(SEG, min_periods=1).mean().values
    price_norm = close / np.maximum(price_roll, 1e-6)
    del high, low

    def mkmap(s, e):
        a = int(pd.Timestamp(s, tz='UTC').timestamp())
        b = int(pd.Timestamp(e, tz='UTC').timestamp())
        return (ts >= a) & (ts < b) & (np.arange(N) >= SEG) & (np.arange(N) < N - H)
    tr_m = mkmap(*config.SPLITS['train'])
    es_m = mkmap(*config.SPLITS['early_stop'])
    te_m = mkmap(*config.SPLITS['test'])

    rng = np.random.RandomState(config.SEED)
    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) > 80_000: tr_idx = rng.choice(tr_idx, 80_000, replace=False)
    es_idx = np.where(es_m)[0]
    if len(es_idx) > 20_000: es_idx = es_idx[::max(1, len(es_idx) // 20000)]
    te_idx = np.where(te_m)[0]
    print(f'[data] TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,} ({time.time()-t0:.0f}s)', flush=True)

    # 渲染训练/验证 (内存 ~3GB)
    Xtr, ytr = render_batch(tr_idx, 0, stoch, price_norm, close, H)
    Xes, yes = render_batch(es_idx, 0, stoch, price_norm, close, H)
    del stoch, price_norm
    print(f'[render] TR={Xtr.shape} ES={Xes.shape} ({time.time()-t0:.0f}s)', flush=True)

    dev = torch.device('cpu')
    torch.manual_seed(config.SEED)
    model = TrendCNN(in_ch=4, H=HEIGHT, W=SEG).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr).to(dev); ytr_t = torch.tensor(ytr, dtype=torch.float32).to(dev)
    Xes_t = torch.tensor(Xes).to(dev); yes_t = torch.tensor(yes, dtype=torch.float32).to(dev)
    del Xes

    B = 512
    best_auc = 0
    print(f'\n[2] Training', flush=True)
    for epoch in range(6):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0; nb = 0
        for i in range(0, len(perm), B):
            bi = perm[i:i + B]
            xb, yb = Xtr_t[bi], ytr_t[bi]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(bi); nb += len(bi)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xes_t)).numpy()
        auc = roc_auc_score(yes_t.numpy(), pv)
        k1 = max(1, int(len(pv) * 0.01)); a1 = yes_t.numpy()[pv.argsort()[-k1:]].mean()*100
        print(f'  ep{epoch}: loss={tot/nb:.4f} es_auc={auc:.4f} es_top1%={a1:.1f}% ({time.time()-t0:.0f}s)', flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'{config.MODEL_DIR}/trend_cnn.pt')

    # test 分批推理
    print(f'\n[3] Test 分批推理', flush=True)
    model.load_state_dict(torch.load(f'{config.MODEL_DIR}/trend_cnn.pt'))
    model.eval()
    n_chunk = 20000
    pv_all = np.zeros(len(te_idx), np.float32)
    for s in range(0, len(te_idx), n_chunk):
        e = min(s + n_chunk, len(te_idx))
        Xb, _ = render_batch(te_idx[s:e], 0, None, None, close, H)
        with torch.no_grad():
            pv_all[s:e] = torch.sigmoid(model(torch.tensor(Xb).to(dev))).numpy()
        del Xb
        print(f'  [{e:,}/{len(te_idx):,}] ({time.time()-t0:.0f}s)', flush=True)
    yte = (close[te_idx + H] > close[te_idx]).astype(int)
    auc = roc_auc_score(yte, pv_all)
    for pct in [0.005, 0.01, 0.02, 0.05]:
        k = max(1, int(len(pv_all) * pct))
        idx = np.argsort(pv_all)
        aL = yte[idx[-k:]].mean()*100
        aS = (1 - yte[idx[:k]]).mean()*100
        print(f'  top{pct*100:.1f}%: long={aL:.1f}% short={aS:.1f}%')
    print(f'\nTEST AUC={auc:.4f}')
    print(f'历史: 87feats LGBM top1%=60.1% AUC=0.544 | 目标 65%')
    print(f'⏱ {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()