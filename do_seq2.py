"""Stable sequence model v2. Loads data from seq_data.npz, trains carefully."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
DEVICE='cpu'
print(f"torch={torch.__version__}", flush=True)

# =====================================
# 0. Load saved data
# =====================================
t0 = time.time()
print("Loading seq_data.npz...", flush=True)
d = np.load('/workspace/models_saved/seq_data.npz')
X_tr = d['X_tr']; y_tr = d['y_tr']; r_tr = d['r_tr']
X_es = d['X_es']; y_es = d['y_es']
X_te = d['X_te']; y_te = d['y_te']; ts_te = d['ts_te']
print(f"  TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape} ({time.time()-t0:.0f}s)", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]

# sanity check
print(f"  X_tr range: [{X_tr.min():.2f}, {X_tr.max():.2f}]  mean={X_tr.mean():.3f}", flush=True)

# =====================================
# 1. Models
# =====================================
class LSTMAttn(nn.Module):
    def __init__(self, C_in, W, h=96):
        super().__init__()
        self.lstm = nn.LSTM(C_in, h, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(h*2, 4, batch_first=True, dropout=0.15)
        self.norm = nn.LayerNorm(h*2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(h*2, h//2), nn.GELU(), nn.Dropout(0.25), nn.Linear(h//2, 1))
    def forward(self, x):
        x = x.transpose(1,2)
        o,_ = self.lstm(x)
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
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.GELU(), nn.Dropout(0.2)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.25), nn.Linear(chs[-1]//2, 1))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def mk(name):
    d = {'lstm': lambda: LSTMAttn(C_IN, W, h=64),
         'lstm_big': lambda: LSTMAttn(C_IN, W, h=128),
         'tcn': lambda: TCN(C_IN, W, [48, 96, 96, 48]),
         'tcn_big': lambda: TCN(C_IN, W, [64, 128, 128, 128, 64])}
    return d[name]()

def evaluate(m, X, bs=512):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i+bs])).to(DEVICE)
            out = m(x)
            if torch.isnan(out).any():
                print(f"  ⚠ NaN in output at batch {i//bs}", flush=True)
                out = torch.nan_to_num(out, nan=0.0)
            pv.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(pv)

def train(name, Xtr, ytr, rtr, Xes, yes,
          ep=20, lr=1e-3, wd=1e-3, bs=128, pat=8,
          jitter=0.0, smooth=0.0, label_smooth=0.0):
    m = mk(name)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"\n  model={name} params={n_params:,}", flush=True)

    pos = ytr.mean()
    pw = np.where(ytr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
    rw = np.clip(np.abs(rtr)*200, 0.2, 5.0).astype(np.float32)

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best = 0.0; bst = None; ni = 0; t0 = time.time()
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        tl = 0.0; nb = 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy())).to(DEVICE)
            if jitter > 0:
                xb = xb + torch.randn_like(xb) * jitter
            yb = torch.from_numpy(ytr[bi]).float().to(DEVICE)
            wb = torch.from_numpy(pw[bi]*rw[bi]).float().to(DEVICE)
            if smooth > 0:
                yb = yb*(1-smooth) + 0.5*smooth
            logits = m(xb)
            if torch.isnan(logits).any():
                print(f"  ⚠ NaN in forward ep{e+1} batch{i}", flush=True)
                continue
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            if torch.isnan(loss):
                print(f"  ⚠ NaN loss ep{e+1} batch{i}", flush=True); continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tl += loss.item(); nb += 1

        pv_es = evaluate(m, Xes)
        auc_es = roc_auc_score(yes, np.nan_to_num(pv_es, nan=0.5))
        sch.step(tl/max(nb,1))

        cur_lr = opt.param_groups[0]['lr']
        print(f"  ep{e+1:2d} loss={tl/max(nb,1):.4f} auc_es={auc_es:.4f} lr={cur_lr:.1e}", flush=True)

        if auc_es > best + 1e-5:
            best = auc_es
            bst = {k: v.detach().clone() for k,v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat:
                print(f"  ⏹ early stop @ ep{e+1}", flush=True); break

    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te); pv_es = evaluate(m, Xes)
    print(f"  best auc_es={best:.4f}  te={roc_auc_score(y_te, np.nan_to_num(pv_te, nan=0.5)):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return m, pv_te, pv_es, best

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        p = np.nan_to_num(p, nan=0.5)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es = roc_auc_score(yes, np.nan_to_num(pv_es, nan=0.5))
    auc_te = roc_auc_score(yte, np.nan_to_num(pv_te, nan=0.5))
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
    return auc_es, auc_te

# =====================================
# 2. Run experiments
# =====================================
# Fix random seed before anything
torch.manual_seed(42); np.random.seed(42)

print(f"\n{'='*60}\nROUND 1: LSTM+Attn small (lr=1e-3, dropout=0.2)\n{'='*60}", flush=True)
m1, pv1_te, pv1_es, _ = train('lstm', X_tr, y_tr, r_tr, X_es, y_es,
                               ep=25, lr=1e-3, wd=1e-3, bs=128, pat=8, jitter=0.02, smooth=0.02)
do_eval('R1 LSTM_small', pv1_es, y_es, pv1_te, y_te)

print(f"\n{'='*60}\nROUND 2: TCN baseline\n{'='*60}", flush=True)
m2, pv2_te, pv2_es, _ = train('tcn', X_tr, y_tr, r_tr, X_es, y_es,
                               ep=25, lr=1e-3, wd=1e-3, bs=256, pat=8, jitter=0.02, smooth=0.02)
do_eval('R2 TCN', pv2_es, y_es, pv2_te, y_te)

print(f"\n{'='*60}\nROUND 3: LSTM_big\n{'='*60}", flush=True)
m3, pv3_te, pv3_es, _ = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es,
                               ep=30, lr=8e-4, wd=2e-3, bs=128, pat=10, jitter=0.03, smooth=0.03)
do_eval('R3 LSTM_big', pv3_es, y_es, pv3_te, y_te)

print(f"\n{'='*60}\nROUND 4: TCN_big\n{'='*60}", flush=True)
m4, pv4_te, pv4_es, _ = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es,
                               ep=30, lr=1e-3, wd=2e-3, bs=256, pat=10, jitter=0.03, smooth=0.03)
do_eval('R4 TCN_big', pv4_es, y_es, pv4_te, y_te)

print(f"\n{'='*60}\nROUND 5: 3-seed LSTM_big rank ensemble\n{'='*60}", flush=True)
pvs_es = []; pvs_te = []
for s in [42, 49, 56]:
    torch.manual_seed(s); np.random.seed(s)
    _, pv_t, pv_e, _ = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es,
                              ep=25, lr=8e-4, wd=2e-3, bs=128, pat=8, jitter=0.02)
    pvs_es.append(pv_e); pvs_te.append(pv_t)

pv_ens_es = rank_agg(pvs_es); pv_ens_te = rank_agg(pvs_te)
do_eval('R5 3-seed LSTM_big rank', pv_ens_es, y_es, pv_ens_te, y_te)

print(f"\n{'='*60}\nROUND 6: LSTM_big × 2 (LSTM+TCN_big) rank ens\n{'='*60}", flush=True)
pv_ens2_es = rank_agg([pv3_es, pv4_es])
pv_ens2_te = rank_agg([pv3_te, pv4_te])
do_eval('R6 LSTM_big+TCN_big rank', pv_ens2_es, y_es, pv_ens2_te, y_te)

print(f"\n{'='*60}\nROUND 7: Try larger LR + weight decay schedule\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m7, pv7_te, pv7_es, _ = train('lstm_big', X_tr, y_tr, r_tr, X_es, y_es,
                               ep=40, lr=5e-4, wd=3e-3, bs=128, pat=12, jitter=0.04, smooth=0.04)
do_eval('R7 LSTM_big slow', pv7_es, y_es, pv7_te, y_te)

print(f"\nTOTAL {time.time()-t0:.0f}s", flush=True)
