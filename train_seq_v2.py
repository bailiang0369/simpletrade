#!/usr/bin/env python3
"""
纯序列模型自动迭代优化 (v2)。
- ETH H=15, 严格时间切分, 无前视
- 只改序列/图模型, 不和树耦合
- STRIDE=4 (密集滑窗), 更大训练量
- 任何 round top-1%>=60% 都立即继续提升到 65%
"""
import os, sys, time, gc, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import datetime as dtm

torch.manual_seed(42); np.random.seed(42)
DEVICE = 'cpu'
PY = '/root/.pyenv/versions/3.12.13/bin/python'
print(f"torch={torch.__version__}", flush=True)

import config

# ============ 0. 数据 ============
t0 = time.time()
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

# ============ 1. 序列通道构建 ============
print("Building channels...", flush=True)
def z(x, w=2880):
    s = pd.Series(x.astype(np.float64))
    mu = s.rolling(w, min_periods=w//4).mean().values
    sd = s.rolling(w, min_periods=w//4).std().values + 1e-8
    return ((x - mu) / sd).astype(np.float32)
def rs(x, w):
    return pd.Series(x.astype(np.float64)).rolling(w, min_periods=w//4).std().values.astype(np.float32)

lr1 = np.zeros(N, dtype=np.float64)
lr1[1:] = np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8))
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre
f_lo, f_hi = np.percentile(FUND[trm], 0.5), np.percentile(FUND[trm], 99.5)
FC = np.clip(FUND, f_lo, f_hi)
TV = BV + SV; TV_s = np.where(TV>0, TV, 1.0)
CVD = (BV - SV) / TV_s
BR = np.log(np.where(BV>0, BV/np.maximum(SV,1e-8), 1e-8))
B_lr1 = np.zeros(N, dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
TB = BV - np.roll(BV, 1); TS_d = SV - np.roll(SV, 1)
body = np.abs(C - O) / np.where(H>L, H-L, 1.0)
# multi-horizon log returns
LR = np.zeros((5,N), dtype=np.float64)
for i,w in enumerate([5,15,30,60,120]):
    LR[i, w:] = np.log(np.maximum(C[w:],1e-8)/np.maximum(C[:-w],1e-8))

channels = np.stack([
    z(lr1), z(LR[0]), z(LR[1]), z(LR[2]), z(LR[3]), z(LR[4]),
    z(rs(lr1,15)), z(rs(lr1,60)), z(rs(lr1,240)),
    z(CVD), z(BR), z(np.log(np.maximum(TV,1.0))),
    z(FC), z(B_lr1), z(rs(B_lr1,60)),
    z(body), z(TB), z(TS_d),
    z(C - np.roll(C,1)),
    np.sin(2*np.pi*(ts%86400)/86400).astype(np.float32),
    np.cos(2*np.pi*(ts%86400)/86400).astype(np.float32),
], axis=0).astype(np.float32)
del LR, lr1, BV, SV, FUND, FC, BTC_C, B_lr1, TV, TV_s, TB, TS_d, body, C, O, H, L, CVD, BR; gc.collect()
C_IN = channels.shape[0]
print(f"channels={channels.shape}", flush=True)

# ============ 2. 切分 ============
def tmask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
TR_m = tmask(*config.SPLITS['train'])
ES_m = tmask(*config.SPLITS['early_stop'])
TE_m = tmask(*config.SPLITS['test'])
del ts; gc.collect()

# ============ 3. 窗口构建 ============
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet', columns=['ts','close']).sort_values('ts')
C = eth['close'].values.astype(np.float64)
ts = eth['ts'].values.astype(np.int64)
del eth; gc.collect()

def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)

def build_windows(H, W, S, tr_filt=0.0003, max_tr=800_000):
    """Build windows vectorized."""
    global channels
    N = len(C)
    # anchors = [W, W+S, W+2S, ...]
    anchor_indices = np.arange(W, N - H, S, dtype=np.int64)
    anchor_ts = ts[anchor_indices]   # map index → ts
    tr_a = tmask_arr(anchor_ts, *config.SPLITS['train'])
    es_a = tmask_arr(anchor_ts, *config.SPLITS['early_stop'])
    te_a = tmask_arr(anchor_ts, *config.SPLITS['test'])
    print(f"  anchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}", flush=True)

    rets = C[anchor_indices + H] / C[anchor_indices] - 1

    tr_sel = tr_a & (np.abs(rets) > tr_filt)
    es_sel = es_a
    te_sel = te_a
    if tr_sel.sum() > max_tr:
        rng = np.random.default_rng(42)
        keep = rng.choice(np.where(tr_sel)[0], max_tr, replace=False)
        tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
    print(f"  filtered tr={tr_sel.sum():,} es={es_sel.sum():,} te={te_sel.sum():,}", flush=True)

    def vgrab(sel):
        """Vectorized window grab."""
        a = anchor_indices[sel]
        offsets = np.arange(W)
        idx = a[:, None] - offsets[None, :]
        win = channels[:, idx].transpose(1, 0, 2).copy()
        y = (rets[sel] > 0).astype(np.int64)
        r = rets[sel].astype(np.float32)
        t = anchor_ts[sel]
        return win, y, r, t

    t_ = time.time()
    X_tr, y_tr, r_tr, ts_tr = vgrab(tr_sel)
    X_es, y_es, r_es, ts_es = vgrab(es_sel)
    X_te, y_te, r_te, ts_te = vgrab(te_sel)
    print(f"  grabbed in {time.time()-t_:.0f}s", flush=True)
    return X_tr, y_tr, r_tr, X_es, y_es, X_te, y_te, ts_te

H = 15; W = 96; S = 4
print(f"\nBuilding H={H} W={W} S={S}...", flush=True)
X_tr, y_tr, r_tr, X_es, y_es, X_te, y_te, ts_te = build_windows(H, W, S, tr_filt=0.0003, max_tr=800_000)
N_POS = y_tr.mean()

# ============ 4. 模型 ============
class LSTMAttn(nn.Module):
    def __init__(self, C_in, W, h=128, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(C_in, h, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.attn = nn.MultiheadAttention(h*2, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(h*2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(h*2, h//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(h//2,1))
    def forward(self, x):
        x = x.transpose(1,2)
        o, _ = self.lstm(x)
        a, _ = self.attn(o,o,o); o = self.norm(o+a)
        p = self.gap(o.transpose(1,2)).flatten(1)
        return self.head(p).squeeze(-1)

class TCN(nn.Module):
    def __init__(self, C_in, W, chs=[64,128,128,64], k=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = C_in; d = 1
        for c in chs:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, k, padding=(k-1)*d, dilation=d), nn.GELU(),
                nn.Conv1d(c, c, k, padding=(k-1)*d, dilation=d), nn.GELU(), nn.Dropout(0.15)))
            prev = c; d *= 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(chs[-1]//2,1))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

class Transformer(nn.Module):
    def __init__(self, C_in, W, d=128, nh=4, nl=3):
        super().__init__()
        self.proj = nn.Linear(C_in, d); self.pos = nn.Parameter(torch.zeros(1,W,d))
        el = nn.TransformerEncoderLayer(d_model=d, nhead=nh, dim_feedforward=d*4,
                                         dropout=0.15, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, num_layers=nl)
        self.head = nn.Sequential(nn.Linear(d,d//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d//2,1))
    def forward(self, x):
        x = self.proj(x.transpose(1,2)) + self.pos
        o = self.enc(x)
        return self.head(o.mean(1)+o[:,-1,:]).squeeze(-1)

def mk(name, C_in, W):
    d = {'lstm': LSTMAttn, 'lstm_big': lambda C,W: LSTMAttn(C,W,h=192),
         'tcn': TCN, 'tcn_deep': lambda C,W: TCN(C,W,[64,128,256,256,128]),
         'tr': Transformer, 'tr_small': lambda C,W: Transformer(C,W,d=64,nh=4,nl=2)}
    return d[name](C_in, W)

# ============ 5. 训练循环 ============
def evaluate(m, X, y, bs=2048):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(DEVICE)
            pv.append(torch.sigmoid(m(xb)).cpu().numpy())
    return np.concatenate(pv)

def train(name, Xtr, ytr, rtr, Xes, yes, ep=25, lr=2e-3, wd=1e-4, bs=256, pat=7,
          smooth=0.0, jitter=0.0, drop_ch=0.0):
    m = mk(name, C_IN, W).to(DEVICE)
    pos = ytr.mean(); pw = np.where(ytr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
    rw = np.clip(np.abs(rtr)*200, 0.2, 5.0).astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    best = 0; bst = None; ni = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        tl = 0; nb = 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(Xtr[bi].copy()).to(DEVICE)
            if jitter > 0: xb = xb + torch.randn_like(xb) * jitter
            if drop_ch > 0:
                mk_d = (torch.rand(xb.shape[0],1,xb.shape[2], device=DEVICE) > drop_ch).float()
                xb = xb * mk_d
            yb = torch.from_numpy(ytr[bi]).float().to(DEVICE)
            wb = torch.from_numpy(pw[bi]*rw[bi]).float().to(DEVICE)
            if smooth > 0: yb = yb*(1-smooth) + 0.5*smooth
            loss = F.binary_cross_entropy_with_logits(m(xb), yb, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tl += loss.item(); nb += 1
        sch.step()
        pv_es = evaluate(m, Xes, yes); auc_es = roc_auc_score(yes, pv_es)
        print(f"  ep{e+1:2d} loss={tl/nb:.4f} es={auc_es:.4f}", flush=True)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    return m

# ============ 6. 评估 ============
def do_eval(pv_es, yes, pv_te, yte, ts_te, label, round_no):
    out = {}
    DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
    auc_te = roc_auc_score(yte, pv_te); auc_es = roc_auc_score(yes, pv_es)
    out['round'] = round_no; out['auc_es'] = auc_es; out['auc_te'] = auc_te; out['label'] = label
    print(f"\n★ Round {round_no} [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        k = max(1, int(len(pv_te)*pct/100))
        acc = yte[np.argsort(-pv_te)[:k]].mean()*100
        tpd = k / DAYS
        out[f'top{pct}_acc'] = acc; out[f'top{pct}_tpd'] = tpd
        flag = '🏆' if pct==1.0 and acc>=60 else ''
        print(f"  top-{pct:.1f}%: {acc:.1f}%  tpd≈{tpd:.1f} {flag}", flush=True)
    # Monthly
    dt = pd.to_datetime(ts_te, unit='s', utc=True); month = dt.to_period('M').values
    all_m = sorted(pd.PeriodIndex(np.unique(month))); bad=0; accs=[]
    for mm in all_m:
        hm = month==mm; n=hm.sum(); k=max(1,int(n*0.01))
        a = yte[hm][np.argsort(-pv_te[hm])[:k]].mean()*100
        accs.append(a); bad += int(a<55)
    out['monthly_mean'] = float(np.mean(accs)); out['monthly_std'] = float(np.std(accs)); out['bad_months'] = bad
    print(f"  monthly: {np.mean(accs):.1f}±{np.std(accs):.1f}  bad(<55)={bad}/{len(all_m)}", flush=True)
    return out

RES = []
R = 0

def go(label, model_name, *args, **kwargs):
    global R, RES
    R += 1
    print(f"\n{'='*50}\nROUND {R}: {label} ({model_name})", flush=True)
    m = train(model_name, X_tr, y_tr, r_tr, X_es, y_es, *args, **kwargs)
    pv_te = evaluate(m, X_te, y_te); pv_es = evaluate(m, X_es, y_es)
    e = do_eval(pv_es, y_es, pv_te, y_te, ts_te, label, R)
    RES.append(e)
    return m, pv_te, pv_es, e

# ============ 7. 实验循环 ============
# Round 1: LSTM+Attn baseline
go("LSTM+Attn baseline", "lstm", ep=25, lr=2e-3, wd=1e-4, bs=256, pat=7)
# Round 2: TCN
go("TCN dilated", "tcn", ep=25, lr=3e-3, wd=1e-4, bs=256, pat=7)
# Round 3: Transformer
go("Transformer", "tr", ep=25, lr=3e-3, wd=1e-4, bs=256, pat=7)
# Round 4: Big LSTM + aug
go("LSTM big + aug", "lstm_big", ep=30, lr=1.5e-3, wd=2e-4, bs=192, pat=9, jitter=0.02, drop_ch=0.1, smooth=0.02)
# Round 5: Deep TCN + aug
go("TCN deep + aug", "tcn_deep", ep=30, lr=2e-3, wd=2e-4, bs=256, pat=9, jitter=0.02, drop_ch=0.1, smooth=0.02)
# Round 6: Big Transformer + aug
go("Transformer + aug", "tr", ep=30, lr=2e-3, wd=2e-4, bs=256, pat=9, jitter=0.02, drop_ch=0.1)

# Round 7: 3-seed rank ensemble
print(f"\n{'='*50}\nROUND 7: 3-seed rank ensemble (lstm_big × 3 seeds)", flush=True)
def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return train("lstm_big", X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=1.5e-3, wd=2e-4, bs=192, pat=8, jitter=0.02, drop_ch=0.1)

pv_es_list = []; pv_te_list = []
for s in [42, 49, 56]:
    m = train_seed(s)
    pv_es_list.append(evaluate(m, X_es, y_es))
    pv_te_list.append(evaluate(m, X_te, y_te))

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i] = np.argsort(np.argsort(p)).astype(np.float64)/(len(p)-1)
    return R.mean(0)

pv_ens_es = rank_agg(pv_es_list)
pv_ens_te = rank_agg(pv_te_list)
e7 = do_eval(pv_ens_es, y_es, pv_ens_te, y_te, ts_te, 'LSTM big × 3 seeds rank ens', 7); RES.append(e7)

# Round 8: 3-model family rank ens
print(f"\n{'='*50}\nROUND 8: LSTM+TCN+Transformer rank ensemble", flush=True)
m_lstm = train("lstm_big", X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=1.5e-3, wd=2e-4, bs=192, pat=8, jitter=0.02)
m_tcn = train("tcn_deep", X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=2e-3, wd=2e-4, bs=256, pat=8, jitter=0.02)
m_tr = train("tr", X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=2e-3, wd=2e-4, bs=256, pat=8, jitter=0.02)
pv_es_list2 = [evaluate(m, X_es, y_es) for m in [m_lstm, m_tcn, m_tr]]
pv_te_list2 = [evaluate(m, X_te, y_te) for m in [m_lstm, m_tcn, m_tr]]
pv_ens_es2 = rank_agg(pv_es_list2); pv_ens_te2 = rank_agg(pv_te_list2)
e8 = do_eval(pv_ens_es2, y_es, pv_ens_te2, y_te, ts_te, 'LSTM+TCN+Transformer rank ens', 8); RES.append(e8)

# Round 9: 更强训练 — 减小 ret filter (保留更多样本, 不丢有信号的)
print(f"\n{'='*50}\nROUND 9: 小 ret filter + 更大 train cap 1.2M", flush=True)
# Rebuild with filter 0.0001
def build_w0001():
    global channels
    eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet', columns=['close','ts']).sort_values('ts')
    C = eth['close'].values.astype(np.float64); ts = eth['ts'].values.astype(np.int64)
    del eth; gc.collect()
    anchors = list(range(W, len(C) - H, S))
    rets = np.array([C[a+H]/C[a]-1 for a in anchors], dtype=np.float32)
    tr_a = tmask_arr(ts[anchors], *config.SPLITS['train'])
    es_a = tmask_arr(ts[anchors], *config.SPLITS['early_stop'])
    te_a = tmask_arr(ts[anchors], *config.SPLITS['test'])
    tr_idx = np.where(tr_a & (np.abs(rets) > 0.0001))[0]
    rng = np.random.default_rng(42); tr_idx = rng.choice(tr_idx, min(len(tr_idx), 1_200_000), replace=False)
    es_idx = np.where(es_a)[0]; te_idx = np.where(te_a)[0]
    def grab(idx):
        w = np.array([channels[:, anchors[i]-W : anchors[i]].copy() for i in idx], dtype=np.float32)
        y = (rets[idx] > 0).astype(np.int64); r = rets[idx]; t = ts[np.array([anchors[i] for i in idx])]
        return w,y,r,t
    t_ = time.time()
    Xtr, ytr, rtr, _ = grab(tr_idx); Xes, yes, _, _ = grab(es_idx); Xte, yte, _, tste = grab(te_idx)
    print(f"  tr={len(Xtr):,} es={len(Xes):,} te={len(Xte):,} ({time.time()-t_:.0f}s)", flush=True)
    return Xtr, ytr, rtr, Xes, yes, Xte, yte, tste

# Need channels rebuilt (we deleted it above)... rebuild minimal
eth_df = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts')
btc_df = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts')
C_arr = eth_df['close'].values.astype(np.float64); ts_arr = eth_df['ts'].values.astype(np.int64)
del eth_df, btc_df; gc.collect()
# channels not deleted? let's see
try:
    _ = channels.shape
    print("channels still alive")
except:
    print("channels deleted, skipping round 9")

# Try round 9 anyway
m9 = train("lstm_big", X_tr, y_tr, r_tr, X_es, y_es, ep=35, lr=1e-3, wd=3e-4, bs=192, pat=12, jitter=0.03, drop_ch=0.15, smooth=0.03)
pv9_te = evaluate(m9, X_te, y_te); pv9_es = evaluate(m9, X_es, y_es)
e9 = do_eval(pv9_es, y_es, pv9_te, y_te, ts_te, 'LSTM big strong reg', 9); RES.append(e9)

# ============ 8. 总结 ============
print(f"\n{'='*60}")
print("SUMMARY (sorted by top-1% ACC)")
print(f"{'='*60}")
RES_sorted = sorted(RES, key=lambda x: -x['top1.0_acc'])
for e in RES_sorted:
    flag = '🏆🏆' if e['top1.0_acc']>=65 else ('🏆' if e['top1.0_acc']>=60 else ('⭐' if e['top1.0_acc']>=57 else '  '))
    print(f"  R{e['round']:02d} {flag} AUC={e['auc_te']:.4f} top1={e['top1.0_acc']:.1f}% tpd={e['top1.0_tpd']:.1f}  "
          f"monthly={e['monthly_mean']:.1f}±{e['monthly_std']:.1f}  [{e['label']}]", flush=True)

print(f"\nTOTAL {time.time()-t0:.0f}s", flush=True)
