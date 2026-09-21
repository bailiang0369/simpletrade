"""
Minimal CNN v2: 6-channel, bigger model, fixed funding, more samples.
- funding clip to p0.1/p99.9 before z-score
- Model ~3M params (stem=128, blocks=256)
- STRIDE=8 → ~400K train windows
- H=15 (config default)
"""
import os, sys, time, gc, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

import config

SEG = 64
STRIDE = 8
H = config.HORIZON_MIN
BATCH = 256
EPOCHS = 20
LR = 2e-3
WD = 1e-4
PATIENCE = 7
DEVICE = torch.device('cpu')

t0 = time.time()
print(f'[1] 加载数据 SEG={SEG} STRIDE={STRIDE} H={H}', flush=True)

eth = pd.read_parquet(os.path.join(config.DS_DIR, 'raw_ETH.parquet'))
btc = pd.read_parquet(os.path.join(config.DS_DIR, 'raw_BTC.parquet'))

eth_idx = pd.Index(eth['ts'].values)
btc_reorder = np.clip(btc['ts'].values.searchsorted(eth_idx.values, side='right') - 1, 0, len(btc)-1)

close = eth['close'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float32)[btc_reorder]

N = len(close)

# === 清洗 funding ===
_tr_end_ts = int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
_tr_mask = ts < _tr_end_ts
# 用 train 段算 clip 边界
fund_p01 = np.percentile(fund[_tr_mask & (fund > -100)], 0.1)
fund_p999 = np.percentile(fund[_tr_mask & (fund < 100)], 99.9)
fund = np.clip(fund, fund_p01, fund_p999)
fund_mean = fund[_tr_mask].mean()
fund_std = max(fund[_tr_mask].std(), 1e-6)
fund_z = (fund - fund_mean) / fund_std
print(f'  funding clip [{fund_p01:.3f}, {fund_p999:.3f}] z-scored OK')
del fund

# === bar-level feats ===
bar_bs = buy_v - sell_v
bar_br = np.log(np.maximum(buy_v / np.maximum(sell_v, 1.0), 1e-6))
bar_tv = np.log(np.maximum(buy_v + sell_v, 1.0))
del buy_v, sell_v; gc.collect()

print(f'[2] 构建窗口 ...', flush=True)
win_list, target_list, win_ts_list = [], [], []
cnt = 0
for start in range(0, N - SEG - H, STRIDE):
    end = start + H
    end_win = start + SEG
    if end_win + H > N: break
    # label: close[end_win + H] vs close[end_win]
    target = 1 if close[end_win + H] > close[end_win] else 0
    
    c0 = close[start]; bc0 = btc_c[start]
    w = np.zeros((6, SEG), np.float32)
    w[0] = np.log(close[start:end_win] / max(c0, 1e-6))          # close_ret
    bs_win = bar_bs[start:end_win]
    cum_bs = np.cumsum(bs_win)
    w[1] = cum_bs / max(np.abs(cum_bs[-1]), 1.0)                  # cvd_diff norm
    w[2] = bar_br[start:end_win]                                  # buy_ratio
    tv_win = bar_tv[start:end_win]
    w[3] = tv_win - tv_win.mean()                                  # vol detrended
    w[4] = fund_z[start:end_win]                                   # funding z
    w[5] = np.log(btc_c[start:end_win] / max(bc0, 1e-6))          # btc_ret
    
    if np.isnan(w).any() or np.isinf(w).any():
        continue
    win_list.append(w)
    target_list.append(target)
    win_ts_list.append(ts[end_win])
    cnt += 1
    if cnt % 200_000 == 0:
        print(f'  ... {cnt:,} windows ({time.time()-t0:.0f}s)', flush=True)

X_all = np.array(win_list, np.float32)
y_all = np.array(target_list, np.int64)
ts_all = np.array(win_ts_list, np.int64)
del win_list, target_list, win_ts_list, bar_bs, bar_br, bar_tv, close, btc_c, fund_z, ts; gc.collect()
print(f'  ✅ {X_all.shape}  pos_rate={y_all.mean():.3f}  ({time.time()-t0:.0f}s)', flush=True)

# === 切分 ===
def _mask(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_all >= a) & (ts_all < b)

tr_m = _mask(*config.SPLITS['train'])
es_m = _mask(*config.SPLITS['early_stop'])
te_m = _mask(*config.SPLITS['test'])
X_tr, y_tr = X_all[tr_m], y_all[tr_m]
X_es, y_es = X_all[es_m], y_all[es_m]
X_te, y_te = X_all[te_m], y_all[te_m]
del X_all, y_all, ts_all, tr_m, es_m, te_m; gc.collect()
print(f'  TR={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}', flush=True)

# === 模型 ~3M ===
class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.ln1 = nn.LayerNorm([ch, SEG])
        self.ln2 = nn.LayerNorm([ch, SEG])
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.15)
    def forward(self, x):
        res = x
        x = self.drop(self.act(self.ln1(self.conv1(x))))
        x = self.drop(self.act(self.ln2(self.conv2(x))))
        return x + res

class CNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 128, 3, padding=1, bias=False),
            nn.LayerNorm([128, SEG]),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.b1 = DilatedBlock(128, dilation=2)   # RF=5
        self.b2 = DilatedBlock(128, dilation=4)   # RF=13
        self.b3 = DilatedBlock(128, dilation=8)   # RF=29
        self.b4 = DilatedBlock(128, dilation=16)  # RF=61
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

model = CNNv2()
n_params = sum(p.numel() for p in model.parameters())
print(f'\n[3] 模型 {n_params:,} ({n_params/1e6:.2f}M)', flush=True)

# === 训练 ===
ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
ds_es = TensorDataset(torch.from_numpy(X_es), torch.from_numpy(y_es))
dl_tr = DataLoader(ds_tr, BATCH, shuffle=True, num_workers=0, drop_last=True)
dl_es = DataLoader(ds_es, BATCH, shuffle=False, num_workers=0)
del X_tr, y_tr, X_es, y_es; gc.collect()

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=4, T_mult=2, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_es_auc = 0
best_state = None
no_improve = 0

print(f'\n[4] 训练 {EPOCHS} epochs patience={PATIENCE}', flush=True)
for ep in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []
    for xb, yb in dl_tr:
        optim.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tr_losses.append(loss.item())
    sched.step()
    
    # ES AUC
    model.eval()
    pv_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in dl_es:
            pv_all.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
            y_all.append(yb.numpy())
    pv_es = np.concatenate(pv_all); y_es = np.concatenate(y_all)
    es_auc = roc_auc_score(y_es, pv_es)
    tr_loss = np.mean(tr_losses)
    
    if es_auc > best_es_auc + 1e-5:
        best_es_auc = es_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    
    print(f'  ep {ep:2d}  tr={tr_loss:.4f}  es_auc={es_auc:.4f}  best={best_es_auc:.4f}  '
          f'lr={sched.get_last_lr()[0]:.2e} {"★" if no_improve==0 else ""} ({time.time()-t0:.0f}s)', flush=True)
    
    if no_improve >= PATIENCE:
        print(f'  ⏹ early stop', flush=True); break

# === Test ===
print(f'\n[5] Test 评估', flush=True)
if best_state:
    model.load_state_dict(best_state)
    print('  loaded best ES state')

model.eval()
ds_te = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
dl_te = DataLoader(ds_te, BATCH, shuffle=False, num_workers=0)
pv_all, y_all = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        pv_all.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
        y_all.append(yb.numpy())
pv_te = np.concatenate(pv_all); y_te = np.concatenate(y_all)

auc = roc_auc_score(y_te, pv_te)
print(f'\n  📊 Test AUC = {auc:.4f}')

n_days = 356
for pct in [0.005, 0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(pv_te) * pct))
    top_idx = pv_te.argsort()[-k:]
    acc = y_te[top_idx].mean() * 100
    tpd = k / n_days
    flag = '✅' if pct <= 0.02 and acc >= 65 else ''
    print(f'  top-{pct*100:.1f}%: acc={acc:.1f}%  tpd≈{tpd:.1f}  {flag}')

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
