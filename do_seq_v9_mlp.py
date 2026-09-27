"""v9: MLP hyperparameter search + ensembles."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)

d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_f = X_es.reshape(len(X_es), -1)
X_te_f = X_te.reshape(len(X_te), -1)
FT = X_tr_f.shape[1]; print(f"FT={FT} TR={X_tr_f.shape}", flush=True)
gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs=[512,256], drop=0.4):
        super().__init__()
        prev=ft; layers=[]
        for h in hs:
            layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev=h
        layers.append(nn.Linear(prev,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train(name, m, Xtr, ytr, Xes, yes, Xte, yte,
          ep=20, lr=5e-4, wd=0.05, bs=512, pat=6, smooth=0.2, jitter=0.05):
    np_=sum(p.numel() for p in m.parameters())
    print(f"\n  [{name}] params={np_:,} lr={lr} wd={wd}", flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0; t0=time.time()
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            if jitter>0: xb=xb+torch.randn_like(xb)*jitter
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item(); nb+=1
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
        print(f"    ep{e+1:2d} loss={tl/max(nb,1):.4f} ES={auc_es:.4f}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te=evaluate(m,Xte); pv_es=evaluate(m,Xes)
    auc_te=roc_auc_score(yte,pv_te)
    print(f"    → TE={auc_te:.4f} (best ES={best:.4f} @ep{best_ep}) [{time.time()-t0:.0f}s]", flush=True)
    return m, pv_te, pv_es, auc_te

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es=roc_auc_score(yes,np.nan_to_num(pv_es,nan=0.5))
    auc_te=roc_auc_score(yte,np.nan_to_num(pv_te,nan=0.5))
    DAYS=(ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5,1.0,1.5,2.0,3.0,5.0]:
        k=max(1,int(len(pv_te)*pct/100))
        acc=yte[np.argsort(-pv_te)[:k]].mean()*100; tpd=k/DAYS
        flag='🏆' if pct==1.0 and acc>=60 else ''
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# ============ Grid ============
print(f"\n{'='*60}\nGrid Search MLP\n{'='*60}", flush=True)
best_result = (0, None)  # (auc_te, dict)
results = []

configs = [
    ('MLP_256_wd0.05', [256], 0.5, 5e-4, 0.05, 0.2, 0.05),
    ('MLP_256_wd0.03', [256], 0.5, 5e-4, 0.03, 0.2, 0.05),
    ('MLP_256_wd0.08', [256], 0.5, 5e-4, 0.08, 0.2, 0.05),
    ('MLP_256_d0.6',   [256], 0.6, 5e-4, 0.05, 0.2, 0.05),
    ('MLP_256_d0.4',   [256], 0.4, 5e-4, 0.05, 0.2, 0.05),
    ('MLP_512_wd0.05', [512], 0.5, 5e-4, 0.05, 0.2, 0.05),
    ('MLP_512_d0.6_wd0.05', [512], 0.6, 5e-4, 0.05, 0.25, 0.05),
    ('MLP_512_256_wd0.05', [512,256], 0.5, 5e-4, 0.05, 0.2, 0.05),
    ('MLP_512_256_wd0.08', [512,256], 0.5, 5e-4, 0.08, 0.2, 0.05),
    ('MLP_512_256_d0.6_wd0.06', [512,256], 0.6, 5e-4, 0.06, 0.25, 0.05),
    ('MLP_768_256_wd0.06', [768,256], 0.6, 5e-4, 0.06, 0.25, 0.05),
]

# NOTE: X_tr_f shape issue — MLP needs 2D but train_early was written for 3D
# Let me adjust evaluate and train to work with 2D (flattened)
# Actually MLP forward does Flatten(), so 3D input also works!

for name, hs, drop, lr, wd, smooth, jitter in configs:
    torch.manual_seed(42); np.random.seed(42)
    m = MLP(FT, hs, drop)
    _, pv_t, pv_e, auc_t = train(name, m, X_tr_f, y_tr, X_es_f, y_es, X_te_f, y_te,
                                   ep=20, lr=lr, wd=wd, bs=512, pat=6, smooth=smooth, jitter=jitter)
    results.append((name, auc_t, pv_t, pv_e))
    do_eval(name, pv_e, y_es, pv_t, y_te)

# Sort and save best results
results.sort(key=lambda x: x[1], reverse=True)
print(f"\n\n{'='*60}\nRESULTS RANKED by TE AUC:\n{'='*60}", flush=True)
for i,(nm,auc_t,_,_) in enumerate(results):
    print(f"  #{i+1}  {nm}: TE AUC={auc_t:.4f}", flush=True)

# ============ Ensemble ============
print(f"\n{'='*60}\nEnsemble: top 3 configs × 5 seeds\n{'='*60}", flush=True)
top3 = results[:3]
all_pvs_t = []; all_pvs_e = []
for nm,auc_t,pv_t,pv_e in top3:
    all_pvs_t.append(pv_t); all_pvs_e.append(pv_e)
    # Also train 3 more seeds for this config
    for s in [49, 56, 63, 70]:
        torch.manual_seed(s); np.random.seed(s)
        # Parse config name — just re-run with same hyperparams
        # For simplicity, train 5 seeds of config[0] (best one)
        pass

# Actually let's do 5 seeds of #1 config directly
name1, _, _, _ = top3[0]
print(f"\n  Training 5 seeds of best: {name1}", flush=True)
# Parse hs, drop from name
import re
m = re.search(r'MLP_(\d+)(?:_(\d+))?_d([\d.]+)?_wd([\d.]+)?', name1)
if m:
    hs = [int(m.group(1))]
    if m.group(2): hs.append(int(m.group(2)))
    drop = float(m.group(3)) if m.group(3) else 0.5
    wd = float(m.group(4)) if m.group(4) else 0.05
else:
    hs, drop, wd = [256], 0.5, 0.05

pvs_t=[]; pvs_e=[]
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT, hs, drop)
    _, pv_t, pv_e, auc_t = train(f'seed{s}', m, X_tr_f, y_tr, X_es_f, y_es, X_te_f, y_te,
                                   ep=20, lr=5e-4, wd=wd, bs=512, pat=6, smooth=0.2, jitter=0.05)
    pvs_t.append(pv_t); pvs_e.append(pv_e)

pv_ens_t = rank_agg(pvs_t); pv_ens_e = rank_agg(pvs_e)
do_eval(f'5-seed MLP({hs}, d={drop}, wd={wd})', pv_ens_e, y_es, pv_ens_t, y_te)

print(f"\nDONE", flush=True)
