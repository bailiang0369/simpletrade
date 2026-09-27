"""v10: Bigger MLP + 10 seeds on v6."""
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
FT = X_tr_f.shape[1]; print(f"FT={FT}", flush=True)
gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
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

def train(m, Xtr, ytr, Xes, yes, Xte, yte, ep=20, lr=5e-4, wd=0.06, bs=512, pat=6, smooth=0.2, jitter=0.05):
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0
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
    if bst: m.load_state_dict(bst)
    pv_te=evaluate(m,Xte); pv_es=evaluate(m,Xes)
    auc_te=roc_auc_score(yte,pv_te)
    return pv_te, pv_es, auc_te, bst

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

# 10 seeds of best config
hs = [512, 256]; drop = 0.6; wd = 0.06
seeds = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
pvs_t=[]; pvs_e=[]
print(f"\n{'='*60}\n10-seed MLP({hs}, d={drop}, wd={wd})\n{'='*60}", flush=True)
for s in seeds:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT, hs, drop)
    np_ = sum(p.numel() for p in m.parameters())
    pv_t, pv_e, auc_t, _ = train(m, X_tr_f, y_tr, X_es_f, y_es, X_te_f, y_te,
                                  ep=20, lr=5e-4, wd=wd, bs=512, pat=6, smooth=0.2, jitter=0.05)
    print(f"  seed{s}: TE={auc_t:.4f} params={np_:,}", flush=True)
    pvs_t.append(pv_t); pvs_e.append(pv_e)

pv_ens_t = rank_agg(pvs_t); pv_ens_e = rank_agg(pvs_e)
do_eval(f'10-seed MLP({hs})', pv_ens_e, y_es, pv_ens_t, y_te)

# Try bigger MLP
print(f"\n{'='*60}\n5-seed MLP(768, 512, 256, d=0.6, wd=0.05)\n{'='*60}", flush=True)
hs2 = [768, 512, 256]; drop2 = 0.6; wd2 = 0.05
pvs_t2=[]; pvs_e2=[]
for s in seeds[:5]:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT, hs2, drop2)
    pv_t, pv_e, auc_t, _ = train(m, X_tr_f, y_tr, X_es_f, y_es, X_te_f, y_te,
                                  ep=20, lr=5e-4, wd=wd2, bs=256, pat=7, smooth=0.2, jitter=0.05)
    print(f"  seed{s}: TE={auc_t:.4f}", flush=True)
    pvs_t2.append(pv_t); pvs_e2.append(pv_e)
pv_ens_t2 = rank_agg(pvs_t2); pv_ens_e2 = rank_agg(pvs_e2)
do_eval(f'5-seed MLP({hs2})', pv_ens_e2, y_es, pv_ens_t2, y_te)

# Save predictions for potential stacking
import numpy as np2
np2.savez_compressed('/workspace/models_saved/nn_predictions_v6.npz',
    pv_mlp10_te=pv_ens_t, pv_mlp10_es=pv_ens_e,
    pv_mlp768_te=pv_ens_t2, pv_mlp768_es=pv_ens_e2,
    ts_te=ts_te)
print(f"\nSaved predictions!", flush=True)
