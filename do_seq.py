"""Compact sequence model with careful memory management."""
import os, sys, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

# =====================================
# 0. Load raw data
# =====================================
print("Loading...", flush=True); t0 = time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)
ts = eth['ts'].values.astype(np.int64)
C = eth['close'].values.astype(np.float64)
O = eth['open'].values.astype(np.float64)
H = eth['high'].values.astype(np.float64)
L = eth['low'].values.astype(np.float64)
BV = eth['buy_vol'].values.astype(np.float64)
SV = eth['sell_vol'].values.astype(np.float64)
FUND = eth['funding'].values.astype(np.float64)
BTC_C = btc['close'].values.astype(np.float64)[br]
del eth, btc; gc.collect()
N = len(C)
print(f"N={N:,}  ({time.time()-t0:.0f}s)", flush=True)

# =====================================
# 1. Build 10 compact channels (all float32, z-scored)
# =====================================
print("Building 10 compact channels...", flush=True)
def z(x, w=2880):
    s = pd.Series(x.astype(np.float64))
    mu = s.rolling(w, min_periods=w//4).mean().values
    sd = s.rolling(w, min_periods=w//4).std().values + 1e-8
    return ((x - mu) / sd).astype(np.float32)
def rs(x, w):
    return pd.Series(x.astype(np.float64)).rolling(w, min_periods=w//4).std().values.astype(np.float32)

lr1 = np.zeros(N, dtype=np.float64); lr1[1:] = np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8))
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre
f_lo, f_hi = np.percentile(FUND[trm], 0.5), np.percentile(FUND[trm], 99.5)
FC = np.clip(FUND, f_lo, f_hi)
TV = BV + SV
TV_s = np.where(TV>0, TV, 1.0)
CVD = (BV-SV)/TV_s
B_lr1 = np.zeros(N, dtype=np.float64); B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))

# === 10 channels ===
channels = np.stack([
    z(lr1),                        # 0: 1min lr z
    z(rs(lr1, 60)),                # 1: 60min vol z
    z(rs(lr1, 240)),               # 2: 240min vol z
    z(np.log(np.maximum(CVD+1, 0.5))),  # 3: CVD sign log
    z(np.log(np.maximum(TV, 1.0))),   # 4: total vol log
    z(FC),                         # 5: funding
    z(B_lr1),                      # 6: BTC 1min lr
    z(rs(B_lr1, 60)),              # 7: BTC 60min vol
    np.sin(2*np.pi*(ts%86400)/86400).astype(np.float32),   # 8: hour sin
    np.cos(2*np.pi*(ts%86400)/86400).astype(np.float32),   # 9: hour cos
], axis=0).astype(np.float32)   # (10, 3,501,194) = 140 MB
print(f"channels={channels.shape}  {channels.nbytes/1e9:.3f}GB", flush=True)
del lr1, trm, FC, TV, TV_s, CVD, B_lr1, FUND, BV, SV, BTC_C; gc.collect()

# =====================================
# 2. Windows: W=64, S=16 (anchors every 16 min), H=15
# =====================================
H = 15; W = 64; S = 16
anchor_start = W  # first window ends at index W
anchor_indices = np.arange(anchor_start, N - H, S, dtype=np.int64)
anchor_ts = ts[anchor_indices]
print(f"total anchors: {len(anchor_indices):,}", flush=True)

def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)

tr_a = tmask_arr(anchor_ts, *config.SPLITS['train'])
es_a = tmask_arr(anchor_ts, *config.SPLITS['early_stop'])
te_a = tmask_arr(anchor_ts, *config.SPLITS['test'])
print(f"  anchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}", flush=True)

rets = C[anchor_indices + H] / C[anchor_indices] - 1
tr_sel = tr_a & (np.abs(rets) > 0.0003)
MAX_TR = 400_000
if tr_sel.sum() > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(np.where(tr_sel)[0], MAX_TR, replace=False)
    tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
print(f"  filtered tr={tr_sel.sum():,}", flush=True)

# =====================================
# 3. Sliding window via stride tricks, memmap intermediate
# =====================================
print("\nMaterializing windows via stride tricks...", flush=True); t0 = time.time()
from numpy.lib.stride_tricks import sliding_window_view

# Save channels as memmap to free RAM
ch_mmap = '/tmp/ch10.bin'
ch_m = np.memmap(ch_mmap, dtype=np.float32, mode='w+', shape=channels.shape)
ch_m[:] = channels[:]; ch_m.flush()
del channels; gc.collect()

ch_rm = np.memmap(ch_mmap, dtype=np.float32, mode='r', shape=(10, N))
print(f"channels as memmap", flush=True)

# Create sliding windows (C, N-W, W) as view
swv = sliding_window_view(ch_rm, W, axis=1)    # view: (10, N-W, 64)
# Transpose → (N-W, C, W) as view
wa = np.transpose(swv, (1,0,2))
# anchor at end of window: anchor_indices[i] = W + i*S  (first anchor=W)
# window index in wa is (anchor - W) exactly!
win_idx = anchor_indices - W   # start in wa
del swv; gc.collect()

# Now slice wa[win_idx] in chunks → materialize
def chunk_materialize(name, sel, batch=20000):
    picked = win_idx[sel]
    n = len(picked); parts = []
    t0 = time.time()
    for i in range(0, n, batch):
        part = wa[picked[i:i+batch]].copy()   # (b, 10, 64)
        parts.append(part); del part; gc.collect()
    X = np.concatenate(parts, axis=0)
    print(f"  {name}: {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    return X

X_tr = chunk_materialize('X_tr', tr_sel)
X_es = chunk_materialize('X_es', es_a)
X_te = chunk_materialize('X_te', te_a)

y_tr = (rets[tr_sel] > 0).astype(np.int64)
r_tr = rets[tr_sel].astype(np.float32)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64)
ts_te = anchor_ts[te_a]

# Save memmap of X arrays too
np.savez_compressed('/workspace/models_saved/seq_data.npz',
    X_tr=X_tr, y_tr=y_tr, r_tr=r_tr,
    X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f"\nSaved seq_data.npz  X_tr={X_tr.shape} ({X_tr.nbytes/1e9:.2f}GB)", flush=True)
print(f"TR pos={y_tr.mean():.3f}  ES pos={y_es.mean():.3f}  TE pos={y_te.mean():.3f}", flush=True)
del ch_rm, wa; gc.collect()

# =====================================
# 4. Models
# =====================================
print("\n" + "="*60, flush=True)
print("Training...", flush=True)

class LSTMAttn(nn.Module):
    def __init__(self, C_in, W, h=96):
        super().__init__()
        self.lstm = nn.LSTM(C_in, h, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.attn = nn.MultiheadAttention(h*2, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(h*2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(h*2, h//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(h//2, 1))
    def forward(self, x):
        x = x.transpose(1,2); o,_ = self.lstm(x)
        a,_ = self.attn(o,o,o); o = self.norm(o+a)
        p = self.gap(o.transpose(1,2)).flatten(1)
        return self.head(p).squeeze(-1)

class TCN(nn.Module):
    def __init__(self, C_in, W, chs=[48, 96, 96, 48]):
        super().__init__()
        prev = C_in; d = 1; blocks = []
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, 3, padding=2*d, dilation=d), nn.GELU(),
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.GELU(), nn.Dropout(0.15)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(chs[-1]//2, 1))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def mk(name, C, W):
    return {'lstm': lambda: LSTMAttn(C,W,h=96), 'lstm_big': lambda: LSTMAttn(C,W,h=128),
            'tcn': lambda: TCN(C,W,[48,96,96,48]), 'tcn_big': lambda: TCN(C,W,[64,128,128,128,64])}[name]()

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(X[i:i+bs]))).numpy())
    return np.concatenate(pv)

def train(name, Xtr, ytr, rtr, Xes, yes, ep=25, lr=2e-3, wd=1e-4, bs=256, pat=7, jitter=0.0, drop_ch=0.0, smooth=0.0):
    m = mk(name, Xtr.shape[1], Xtr.shape[2])
    n_params = sum(p.numel() for p in m.parameters())
    pos = ytr.mean(); pw = np.where(ytr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
    rw = np.clip(np.abs(rtr)*200, 0.2, 5.0).astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    best = 0; bst = None; ni = 0; t0 = time.time()
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr)); tl = 0; nb = 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(Xtr[bi].copy())
            if jitter > 0: xb = xb + torch.randn_like(xb) * jitter
            if drop_ch > 0:
                mask_k = (torch.rand(xb.shape[0], 1, xb.shape[2]) > drop_ch).float()
                xb = xb * mask_k
            yb = torch.from_numpy(ytr[bi]).float()
            wb = torch.from_numpy(pw[bi]*rw[bi]).float()
            if smooth > 0: yb = yb*(1-smooth) + 0.5*smooth
            loss = F.binary_cross_entropy_with_logits(m(xb), yb, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tl += loss.item(); nb += 1
        sch.step()
        pv_es = evaluate(m, Xes); auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0
        else:
            ni += 1
            if ni >= pat: break
        print(f"  ep{e+1:2d} loss={tl/nb:.4f} es={auc_es:.4f}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te); pv_es = evaluate(m, Xes)
    return m, pv_te, pv_es, best

def do_eval(label, pv_es, yes, pv_te, yte, ts_te):
    auc_es = roc_auc_score(yes, pv_es); auc_te = roc_auc_score(yte, pv_te)
    DAYS = (ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        k = max(1, int(len(pv_te)*pct/100))
        acc = yte[np.argsort(-pv_te)[:k]].mean()*100
        tpd = k / DAYS
        flag = '🏆' if pct==1.0 and acc>=60 else ''
        print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)
    # Monthly
    dt = pd.to_datetime(ts_te, unit='s', utc=True); month = dt.to_period('M').values
    all_m = sorted(pd.PeriodIndex(np.unique(month)))
    accs = []; bad = 0
    for mm in all_m:
        hm = month==mm; n = hm.sum(); k = max(1, int(n*0.01))
        a = yte[hm][np.argsort(-pv_te[hm])[:k]].mean()*100; accs.append(a); bad += int(a<55)
    print(f"  monthly: {np.mean(accs):.1f}±{np.std(accs):.1f}  bad(<55)={bad}/{len(all_m)}", flush=True)
    return auc_es, auc_te, [(pct, yte[np.argsort(-pv_te)[:max(1,int(len(pv_te)*pct/100))]].mean()*100) for pct in [0.5,1,2,3,5,10]]

# =====================================
# 5. Run experiments
# =====================================
C_IN = 10
W = 64

print("\n" + "="*60); print("ROUND 1: LSTM+Attn baseline"); print("="*60)
m1, pv1_te, pv1_es, _ = train('lstm', X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=2e-3, wd=1e-4, bs=256, pat=7)
do_eval('R1 LSTM+Attn', pv1_es, y_es, pv1_te, y_te, ts_te)

print("\n" + "="*60); print("ROUND 2: TCN baseline"); print("="*60)
m2, pv2_te, pv2_es, _ = train('tcn', X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=3e-3, wd=1e-4, bs=256, pat=7)
do_eval('R2 TCN', pv2_es, y_es, pv2_te, y_te, ts_te)

print("\n" + "="*60); print("ROUND 3: LSTM+Attn BIG + aug"); print("="*60)
m3, pv3_te, pv3_es, _ = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es, ep=30, lr=1.5e-3, wd=2e-4, bs=192, pat=9, jitter=0.03, drop_ch=0.1, smooth=0.02)
do_eval('R3 LSTM big aug', pv3_es, y_es, pv3_te, y_te, ts_te)

print("\n" + "="*60); print("ROUND 4: TCN BIG + aug"); print("="*60)
m4, pv4_te, pv4_es, _ = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es, ep=30, lr=2e-3, wd=2e-4, bs=256, pat=9, jitter=0.03, drop_ch=0.1)
do_eval('R4 TCN big aug', pv4_es, y_es, pv4_te, y_te, ts_te)

print("\n" + "="*60); print("ROUND 5: 3-seed LSTM_big rank ensemble"); print("="*60)
def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=1.5e-3, wd=2e-4, bs=192, pat=8, jitter=0.02)

pvs_es = []; pvs_te = []
for s in [42, 49, 56]:
    _, pv_te, pv_es, _ = train_seed(s)
    pvs_es.append(pv_es); pvs_te.append(pv_te)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i] = np.argsort(np.argsort(p)).astype(np.float64)/(len(p)-1)
    return R.mean(0)

pv_ens_es = rank_agg(pvs_es); pv_ens_te = rank_agg(pvs_te)
do_eval('R5 3-seed LSTM_big rank ens', pv_ens_es, y_es, pv_ens_te, y_te, ts_te)

print("\n" + "="*60); print("ROUND 6: LSTM_big + TCN_big rank ensemble"); print("="*60)
m_a, pv_a_te, pv_a_es, _ = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=1.5e-3, wd=2e-4, bs=192, pat=8, jitter=0.02)
m_b, pv_b_te, pv_b_es, _ = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=2e-3, wd=2e-4, bs=256, pat=8, jitter=0.02)
do_eval('R6 LSTM_big × 2 seeds', rank_agg([pv_a_es, pv_b_es]), y_es, rank_agg([pv_a_te, pv_b_te]), y_te, ts_te)

print(f"\nTOTAL {time.time()-t0:.0f}s", flush=True)
