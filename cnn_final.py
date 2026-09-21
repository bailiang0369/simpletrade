"""
Final CNN: 6-channel minimal feats, SEG=128, filtered labels, bigger model.

核心洞察: 6-feat 原始序列 AUC 上限 ~0.525 → 用 label filter (ret > 0.2%)
          过滤噪声 bar, 让模型只学有意义的 move.
"""
import os, sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import config

SEG = 128; STRIDE = 16; H = config.HORIZON_MIN
BATCH = 256; EPOCHS = 15; LR = 3e-3; WD = 1e-4; PAT = 7

t0 = time.time()
print(f'[1] 加载 SEG={SEG} STRIDE={STRIDE} H={H}', flush=True)

eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)

close = eth['close'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float32)[br]
N = len(close)

# funding clean
_tre = int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
_trm = ts < _tre
fund = np.clip(fund, np.percentile(fund[_trm], 0.5), np.percentile(fund[_trm], 99.5))
fund_z = (fund - fund[_trm].mean()) / max(fund[_trm].std(), 1e-6)
del fund

bar_bs = buy_v - sell_v
bar_br = np.log(np.maximum(buy_v / np.maximum(sell_v, 1.0), 1e-6))
bar_tv = np.log(np.maximum(buy_v + sell_v, 1.0))
del buy_v, sell_v

# === 窗口构建 + label filter ===
print(f'[2] 构建窗口 + label filter', flush=True)
W, T, Ts = [], [], []
cnt = 0
for start in range(0, N - SEG - H, STRIDE):
    end = start + SEG
    if end + H > N: break
    ret = close[end + H] / close[end] - 1
    
    # LABEL FILTER: 只留 |ret| > 0.2% 的 bar 过滤噪声
    if abs(ret) < 0.002: continue
    label = 1 if ret > 0 else 0
    
    c0 = close[start]; bc0 = btc_c[start]
    w = np.zeros((6, SEG), np.float32)
    w[0] = np.log(close[start:end] / max(c0, 1e-6))
    bs_win = bar_bs[start:end]; cum = np.cumsum(bs_win)
    w[1] = cum / max(np.abs(cum[-1]), 1.0)
    w[2] = bar_br[start:end]
    tvw = bar_tv[start:end]
    w[3] = tvw - tvw.mean()
    w[4] = fund_z[start:end]
    w[5] = np.log(btc_c[start:end] / max(bc0, 1e-6))
    if np.isnan(w).any() or np.isinf(w).any(): continue
    
    W.append(w); T.append(label); Ts.append(ts[end])
    cnt += 1
    if cnt % 100_000 == 0:
        print(f'  ... {cnt:,} ({time.time()-t0:.0f}s)', flush=True)

X_all = np.array(W, np.float32); y_all = np.array(T, np.int64); ts_all = np.array(Ts)
del W, T, Ts, bar_bs, bar_br, bar_tv, close, btc_c, fund_z, ts; gc.collect()
print(f'  ✅ {X_all.shape} pos={y_all.mean():.3f} ({time.time()-t0:.0f}s)', flush=True)

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_all >= a) & (ts_all < b)
tr_m = mk(*config.SPLITS['train']); es_m = mk(*config.SPLITS['early_stop']); te_m = mk(*config.SPLITS['test'])
X_tr, y_tr = X_all[tr_m], y_all[tr_m]
X_es, y_es = X_all[es_m], y_all[es_m]
X_te, y_te = X_all[te_m], y_all[te_m]
del X_all, y_all, ts_all; gc.collect()
print(f'  TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}', flush=True)

# === 模型 ===
class ResDilated(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil, bias=False)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil, bias=False)
        self.ln = nn.LayerNorm([ch, SEG])
    def forward(self, x):
        return x + F.gelu(self.ln(self.c2(F.gelu(self.ln(self.c1(x))))))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(6, 128, 3, padding=1, bias=False),
                                  nn.LayerNorm([128, SEG]), nn.GELU(), nn.Dropout(0.1))
        self.b1 = ResDilated(128, 2); self.b2 = ResDilated(128, 4)
        self.b3 = ResDilated(128, 8); self.b4 = ResDilated(128, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, 128),
                                  nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return self.head(self.pool(x).squeeze(-1))

mdl = Net()
print(f'\n[3] params={sum(p.numel() for p in mdl.parameters()):,}', flush=True)

# === 训练 ===
ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
ds_es = TensorDataset(torch.from_numpy(X_es), torch.from_numpy(y_es))
dl_tr = DataLoader(ds_tr, BATCH, shuffle=True, num_workers=0, drop_last=True)
dl_es = DataLoader(ds_es, BATCH, shuffle=False, num_workers=0)
del X_tr, y_tr, X_es, y_es; gc.collect()

opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=WD)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
crit = nn.BCEWithLogitsLoss()
best_auc, best_st, ni = 0, None, 0

print(f'\n[4] 训练', flush=True)
for ep in range(1, EPOCHS + 1):
    mdl.train(); tl = []
    for xb, yb in dl_tr:
        opt.zero_grad(); log = mdl(xb).squeeze(-1); loss = crit(log, yb.float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        tl.append(loss.item())
    sch.step()
    mdl.eval(); pva, ya = [], []
    with torch.no_grad():
        for xb, yb in dl_es:
            pva.append(torch.sigmoid(mdl(xb).squeeze(-1)).numpy())
            ya.append(yb.numpy())
    pv = np.concatenate(pva); y = np.concatenate(ya)
    auc = roc_auc_score(y, pv)
    if auc > best_auc + 1e-5:
        best_auc = auc; best_st = {k: v.clone() for k, v in mdl.state_dict().items()}; ni = 0
    else:
        ni += 1
    print(f'  ep {ep:2d} tr={np.mean(tl):.4f} es_auc={auc:.4f} best={best_auc:.4f} '
          f'lr={sch.get_last_lr()[0]:.2e} {"★" if ni==0 else ""} ({time.time()-t0:.0f}s)', flush=True)
    if ni >= PAT: print(f'  ⏹ early stop'); break

# === Test ===
print(f'\n[5] Test', flush=True)
if best_st: mdl.load_state_dict(best_st)
mdl.eval(); dste = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
dlte = DataLoader(dste, BATCH, shuffle=False, num_workers=0)
pva, ya = [], []
with torch.no_grad():
    for xb, yb in dlte:
        pva.append(torch.sigmoid(mdl(xb).squeeze(-1)).numpy())
        ya.append(yb.numpy())
pv = np.concatenate(pva); y = np.concatenate(ya)
auc = roc_auc_score(y, pv)
print(f'  📊 AUC={auc:.4f}  (过滤后测试样本={len(y):,})')
for pct in [0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(pv) * pct)); ti = pv.argsort()[-k:]
    acc = y[ti].mean() * 100
    tpd = k / 356
    print(f'  top-{pct*100:.0f}%: acc={acc:.1f}%  tpd≈{tpd:.1f}')
print(f'\n⏱ {time.time()-t0:.0f}s')
