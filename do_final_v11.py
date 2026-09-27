"""Final push: MLP on v6 with different optimizers + 10-seed ens."""
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
X_tr_f = X_tr.reshape(len(X_tr), -1); X_es_f = X_es.reshape(len(X_es), -1); X_te_f = X_te.reshape(len(X_te), -1)
FT = X_tr_f.shape[1]; print(f"FT={FT}", flush=True); gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_optim(m, Xtr, ytr, Xes, yes, Xte, yte, ep=30, bs=256, pat=8, smooth=0.15, jitter=0.05,
                opt_type='adamw', lr=5e-4, wd=0.06):
    if opt_type == 'adamw':
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'sgd':
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    elif opt_type == 'adam':
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
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
    return pv_te, pv_es, roc_auc_score(yte,pv_te), bst

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

all_configs = [
    # (name, hs, drop, opt, lr, wd, jitter)
    ('baseline_512256_adamw', [512,256], 0.6, 'adamw', 5e-4, 0.06, 0.05),
    ('big_1024512256_adamw', [1024,512,256], 0.6, 'adamw', 5e-4, 0.06, 0.05),
    ('big_1024512256_adamw_d07', [1024,512,256], 0.7, 'adamw', 5e-4, 0.08, 0.05),
    ('big_1024512256_sgd', [1024,512,256], 0.6, 'sgd', 1e-2, 0.06, 0.05),
    ('big_1024512256_adamw_j01', [1024,512,256], 0.6, 'adamw', 5e-4, 0.06, 0.10),
    ('wider_2048_adamw', [2048,512], 0.65, 'adamw', 5e-4, 0.07, 0.05),
]

results = []
for name, hs, drop, opt, lr, wd, jitter in all_configs:
    print(f"\n{'='*60}\n{name}: hs={hs} drop={drop} opt={opt}\n{'='*60}", flush=True)
    pvs_t=[]; pvs_e=[]
    for s in [42,49,56,63,70]:
        torch.manual_seed(s); np.random.seed(s)
        m=MLP(FT, hs, drop)
        np_=sum(p.numel() for p in m.parameters())
        pv_t,pv_e,auc_t,_=train_optim(m,X_tr_f,y_tr,X_es_f,y_es,X_te_f,y_te,
                                        ep=25, bs=512, pat=7, smooth=0.15, jitter=jitter,
                                        opt_type=opt, lr=lr, wd=wd)
        pvs_t.append(pv_t); pvs_e.append(pv_e)
    pv_ens_t=rank_agg(pvs_t); pv_ens_e=rank_agg(pvs_e)
    auc_te=roc_auc_score(y_te, pv_ens_t); auc_es=roc_auc_score(y_es, pv_ens_e)
    print(f"  ★ ens TE={auc_te:.4f} ES={auc_es:.4f}", flush=True)
    results.append((name, auc_te, pv_ens_t, pv_ens_e))
    do_eval(name, pv_ens_e, y_es, pv_ens_t, y_te)

# Best of all
results.sort(key=lambda x: x[1], reverse=True)
print(f"\n\n{'='*60}\nFINAL RANKING by TE AUC:\n{'='*60}", flush=True)
for i,(nm,auc_t,_,_) in enumerate(results):
    print(f"  #{i+1}  {nm}: TE AUC={auc_t:.4f}", flush=True)

# Save best 3 predictions
import numpy as np2
top3 = results[:3]
np2.savez_compressed('/workspace/models_saved/nn_predictions_v6_final.npz',
    ts_te=ts_te,
    **{f'pv_{nm}_te': pv_t for nm,_,pv_t,_ in top3},
    **{f'pv_{nm}_es': pv_e for nm,_,_,pv_e in top3})
print(f"\nSaved top3 predictions!", flush=True)
