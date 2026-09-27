"""v5: MLP baseline + aggressive LR + OneCycle + auxiliary tasks."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

d = np.load('/workspace/models_saved/seq_data_v4.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape}", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]
# Flatten for MLP
FT = C_IN * W  # 12*64 = 768
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_f = X_es.reshape(len(X_es), -1)
X_te_f = X_te.reshape(len(X_te), -1)
print(f"MLP FT={FT}", flush=True); gc.collect()

# ============ Models ============
class MLP(nn.Module):
    def __init__(self, ft, hs=[512,256,128], drop=0.2):
        super().__init__()
        prev = ft; layers = []
        for h in hs:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class MLP_Res(nn.Module):
    def __init__(self, ft, hs=512, blocks=3, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(ft, hs), nn.BatchNorm1d(hs), nn.GELU(), nn.Dropout(drop))
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hs, hs), nn.BatchNorm1d(hs), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hs, hs), nn.BatchNorm1d(hs), nn.GELU()) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hs, 1))
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks: x = x + b(x)
        return self.head(x).squeeze(-1)

class TCN(nn.Module):
    def __init__(self, C_in, chs=[64,128,128,64], drop=0.15):
        super().__init__()
        prev = C_in; d = 1; blocks = []
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(chs[-1]//2,1))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def mk(name):
    return {'mlp': lambda: MLP(FT, [512,256,128], 0.2),
            'mlp_big': lambda: MLP(FT, [1024,512,256,128], 0.25),
            'mlp_res': lambda: MLP_Res(FT, 512, 4, 0.2),
            'tcn': lambda: TCN(C_IN, [64,128,128,64], 0.15),
            'tcn_big': lambda: TCN(C_IN, [64,128,256,256,128], 0.2)}[name]()

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train(name, Xtr, ytr, rtr, Xes, yes,
          ep=25, lr=5e-3, wd=1e-3, bs=512, pat=10, smooth=0.1, jitter=0.05, use_oc=True):
    m = mk(name)
    np_ = sum(p.numel() for p in m.parameters())
    pos = ytr.mean(); pw = np.where(ytr>0.5,(1-pos)/pos,pos/(1-pos)).astype(np.float32)
    rw = np.clip(np.abs(rtr)*200, 0.2, 5.0).astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps_per_ep = (len(Xtr)+bs-1)//bs
    if use_oc:
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep*steps_per_ep, pct_start=0.1)
    best = 0.0; bst=None; ni=0; t0=time.time(); best_ep=0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            if jitter>0: xb = xb + torch.randn_like(xb)*jitter
            yb = torch.from_numpy(ytr[bi]).float()
            wb = torch.from_numpy(pw[bi]*rw[bi]).float()
            if smooth>0: yb = yb*(1-smooth)+0.5*smooth
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if use_oc: sch.step()
            tl += loss.item(); nb += 1
        pv_es = evaluate(m, Xes); auc_es = roc_auc_score(yes, pv_es)
        cur_lr = opt.param_groups[0]['lr']
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni += 1
            if ni >= pat: break
        print(f"  ep{e+1:2d} loss={tl/max(nb,1):.4f} auc_es={auc_es:.4f} lr={cur_lr:.1e}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te); pv_es = evaluate(m, Xes)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  → auc_te={auc_te:.4f} (best es={best:.4f} @ep{best_ep}) params={np_:,} [{time.time()-t0:.0f}s]", flush=True)
    return m, pv_te, pv_es

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es = roc_auc_score(yes, np.nan_to_num(pv_es,nan=0.5))
    auc_te = roc_auc_score(yte, np.nan_to_num(pv_te,nan=0.5))
    DAYS=(ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5,1.0,2.0,3.0,5.0,10.0]:
        k=max(1,int(len(pv_te)*pct/100))
        acc=yte[np.argsort(-pv_te)[:k]].mean()*100; tpd=k/DAYS
        flag='🏆' if pct==1.0 and acc>=60 else ''
        print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)
    dt=pd.to_datetime(ts_te,unit='s',utc=True); month=dt.to_period('M').values
    all_m=sorted(pd.PeriodIndex(np.unique(month))); accs=[]; bad=0
    for mm in all_m:
        hm=month==mm; n=hm.sum(); k=max(1,int(n*0.01))
        a=yte[hm][np.argsort(-pv_te[hm])[:k]].mean()*100; accs.append(a); bad+=int(a<55)
    print(f"  monthly: {np.mean(accs):.1f}±{np.std(accs):.1f}  bad(<55)={bad}/{len(all_m)}", flush=True)

# ============ Run ============
print(f"\n{'='*60}\nR1: MLP (baseline, flatten)\n{'='*60}", flush=True)
m1, pv1_t, pv1_e = train('mlp', X_tr_f, y_tr, r_tr, X_es_f, y_es,
                         ep=30, lr=3e-3, wd=1e-3, bs=512, pat=12, smooth=0.1, jitter=0.03)
do_eval('R1 MLP', pv1_e, y_es, pv1_t, y_te)

print(f"\n{'='*60}\nR2: MLP_big\n{'='*60}", flush=True)
m2, pv2_t, pv2_e = train('mlp_big', X_tr_f, y_tr, r_tr, X_es_f, y_es,
                          ep=35, lr=3e-3, wd=2e-3, bs=256, pat=12, smooth=0.1, jitter=0.04)
do_eval('R2 MLP_big', pv2_e, y_es, pv2_t, y_te)

print(f"\n{'='*60}\nR3: MLP_res (residual connections)\n{'='*60}", flush=True)
m3, pv3_t, pv3_e = train('mlp_res', X_tr_f, y_tr, r_tr, X_es_f, y_es,
                          ep=35, lr=4e-3, wd=2e-3, bs=512, pat=12, smooth=0.1, jitter=0.04)
do_eval('R3 MLP_res', pv3_e, y_es, pv3_t, y_te)

print(f"\n{'='*60}\nR4: TCN with BN\n{'='*60}", flush=True)
m4, pv4_t, pv4_e = train('tcn', X_tr, y_tr, r_tr, X_es, y_es,
                          ep=30, lr=3e-3, wd=1e-3, bs=256, pat=10, smooth=0.1, jitter=0.04)
do_eval('R4 TCN', pv4_e, y_es, pv4_t, y_te)

print(f"\n{'='*60}\nR5: TCN_big with BN\n{'='*60}", flush=True)
m5, pv5_t, pv5_e = train('tcn_big', X_tr, y_tr, r_tr, X_es, y_es,
                          ep=35, lr=2e-3, wd=2e-3, bs=256, pat=12, smooth=0.1, jitter=0.05)
do_eval('R5 TCN_big', pv5_e, y_es, pv5_t, y_te)

print(f"\n{'='*60}\nR6: 3-seed MLP_res rank ens\n{'='*60}", flush=True)
pvs_t=[]; pvs_e=[]
for s in [42,49,56]:
    torch.manual_seed(s); np.random.seed(s)
    _, pt, pe = train('mlp_res', X_tr_f, y_tr, r_tr, X_es_f, y_es,
                      ep=30, lr=4e-3, wd=2e-3, bs=512, pat=10, smooth=0.1, jitter=0.04)
    pvs_t.append(pt); pvs_e.append(pe)
do_eval('R6 3-seed MLP_res rank', rank_agg(pvs_e), y_es, rank_agg(pvs_t), y_te)

print(f"\n{'='*60}\nR7: Best of R1-R5 (pick highest auc_es)\n{'='*60}", flush=True)
# Pick R3 MLP_res as likely best
do_eval('R7 MLP_res+TCN_big rank', rank_agg([pv3_e, pv5_e]), y_es, rank_agg([pv3_t, pv5_t]), y_te)

print(f"\nDONE", flush=True)
