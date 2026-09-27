"""Stable sequence model v3: bigger data, 16 channels, TCN-first."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

# =====================================
print("Loading seq_data_v3.npz...", flush=True)
d = np.load('/workspace/models_saved/seq_data_v4.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
print(f"  TR={X_tr.shape}  ES={X_es.shape}  TE={X_te.shape}", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]
print(f"  C_IN={C_IN} W={W}", flush=True)
gc.collect()

# =====================================
# Models
# =====================================
class LSTMAttn(nn.Module):
    def __init__(self, C_in, h=96):
        super().__init__()
        self.lstm = nn.LSTM(C_in, h, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.attn = nn.MultiheadAttention(h*2, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(h*2); self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(h*2, h//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(h//2,1))
    def forward(self, x):
        x = x.transpose(1,2); o,_ = self.lstm(x)
        a,_ = self.attn(o,o,o); o = self.norm(o+a)
        p = self.gap(o.transpose(1,2)).flatten(1)
        return self.head(p).squeeze(-1)

class TCN(nn.Module):
    def __init__(self, C_in, chs=[48, 96, 128, 96]):
        super().__init__()
        prev = C_in; d = 1; blocks = []
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, 3, padding=2*d, dilation=d), nn.GELU(),
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.GELU(), nn.Dropout(0.15)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(chs[-1]//2,1))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def mk(name):
    return {'lstm': lambda: LSTMAttn(C_IN, h=64),
            'lstm_big': lambda: LSTMAttn(C_IN, h=128),
            'tcn': lambda: TCN(C_IN, [48,96,128,96]),
            'tcn_big': lambda: TCN(C_IN, [64,128,128,256,128]),
            'tcn_mega': lambda: TCN(C_IN, [64,128,256,256,256,128])}[name]()

def evaluate(m, X, bs=2048):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i+bs]))
            pv.append(torch.sigmoid(m(x)).numpy())
    return np.concatenate(pv)

def train(name, Xtr, ytr, rtr, Xes, yes,
          ep=25, lr=2e-3, wd=1e-3, bs=256, pat=8,
          jitter=0.0, smooth=0.05, log=0.0):
    m = mk(name)
    np_ = sum(p.numel() for p in m.parameters())
    pos = ytr.mean()
    pw = np.where(ytr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
    rw = np.clip(np.abs(rtr)*200, 0.2, 5.0).astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; t0 = time.time()
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        tl=0; nb=0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            if jitter > 0: xb = xb + torch.randn_like(xb)*jitter
            yb = torch.from_numpy(ytr[bi]).float()
            wb = torch.from_numpy(pw[bi]*rw[bi]).float()
            if smooth > 0: yb = yb*(1-smooth) + 0.5*smooth
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            tl += loss.item(); nb += 1
        pv_es = evaluate(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0
        else:
            ni += 1
            if ni >= pat: break
        if e % 3 == 0 or e < 3:
            print(f"  ep{e+1:2d} loss={tl/max(nb,1):.4f} auc_es={auc_es:.4f}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te); pv_es = evaluate(m, Xes)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  → auc_te={auc_te:.4f} (best es={best:.4f}) params={np_:,} [{time.time()-t0:.0f}s]", flush=True)
    return m, pv_te, pv_es

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es = roc_auc_score(yes, np.nan_to_num(pv_es, nan=0.5))
    auc_te = roc_auc_score(yte, np.nan_to_num(pv_te, nan=0.5))
    DAYS = (ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        k = max(1, int(len(pv_te)*pct/100))
        acc = yte[np.argsort(-pv_te)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=60 else ''
        print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)
    dt = pd.to_datetime(ts_te, unit='s', utc=True); month = dt.to_period('M').values
    all_m = sorted(pd.PeriodIndex(np.unique(month))); accs = []; bad = 0
    for mm in all_m:
        hm = month==mm; n=hm.sum(); k=max(1, int(n*0.01))
        a = yte[hm][np.argsort(-pv_te[hm])[:k]].mean()*100; accs.append(a); bad += int(a<55)
    print(f"  monthly: {np.mean(accs):.1f}±{np.std(accs):.1f}  bad(<55)={bad}/{len(all_m)}", flush=True)
    return auc_es, auc_te

# =====================================
# Run
# =====================================
print(f"\n{'='*60}\nR1: TCN (start with TCN, proven on tables)\n{'='*60}", flush=True)
m1, pv1_t, pv1_e = train('tcn', X_tr, y_tr, r_tr, X_es, y_es,
                         ep=30, lr=2e-3, wd=1e-3, bs=256, pat=10, jitter=0.03, smooth=0.05)
do_eval('R1 TCN', pv1_e, y_es, pv1_t, y_te)

print(f"\n{'='*60}\nR2: TCN_big\n{'='*60}", flush=True)
m2, pv2_t, pv2_e = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es,
                          ep=30, lr=1.5e-3, wd=2e-3, bs=256, pat=10, jitter=0.03, smooth=0.05)
do_eval('R2 TCN_big', pv2_e, y_es, pv2_t, y_te)

print(f"\n{'='*60}\nR3: TCN_mega\n{'='*60}", flush=True)
m3, pv3_t, pv3_e = train('tcn_mega', X_tr, y_tr, r_tr, X_es, y_es,
                          ep=35, lr=1e-3, wd=2e-3, bs=192, pat=12, jitter=0.04, smooth=0.05)
do_eval('R3 TCN_mega', pv3_e, y_es, pv3_t, y_te)

print(f"\n{'='*60}\nR4: LSTM_big\n{'='*60}", flush=True)
m4, pv4_t, pv4_e = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es,
                          ep=30, lr=1.5e-3, wd=2e-3, bs=128, pat=10, jitter=0.03, smooth=0.05)
do_eval('R4 LSTM_big', pv4_e, y_es, pv4_t, y_te)

print(f"\n{'='*60}\nR5: 3-seed TCN_big rank ens\n{'='*60}", flush=True)
pvs_t = []; pvs_e = []
for s in [42, 49, 56]:
    torch.manual_seed(s); np.random.seed(s)
    _, pt, pe = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es,
                      ep=25, lr=1.5e-3, wd=2e-3, bs=256, pat=8, jitter=0.03, smooth=0.05)
    pvs_t.append(pt); pvs_e.append(pe)
do_eval('R5 3-seed TCN_big rank', rank_agg(pvs_e), y_es, rank_agg(pvs_t), y_te)

print(f"\n{'='*60}\nR6: TCN_big × TCN_mega rank\n{'='*60}", flush=True)
do_eval('R6 TCN_big+TCN_mega rank', rank_agg([pv2_e, pv3_e]), y_es, rank_agg([pv2_t, pv3_t]), y_te)

print(f"\nTOTAL {time.time()-__import__('time').time():.0f}s", flush=True)
