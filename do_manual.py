"""Quick check: MLP + LGBM on manual feature sequence (67, 64)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, gc, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

torch.manual_seed(42); np.random.seed(42)

d = np.load('/workspace/models_saved/seq_manual.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_f = X_es.reshape(len(X_es), -1)
X_te_f = X_te.reshape(len(X_te), -1)
FT = X_tr_f.shape[1]
print(f"MANUAL SEQ: C={X_tr.shape[1]} W={X_tr.shape[2]} FT={FT}", flush=True)
print(f"TR={X_tr.shape}  range=[{X_tr.min():.2f}, {X_tr.max():.2f}]", flush=True)
gc.collect()

# Clip outlier per-feature (more aggressive)
for j in range(FT):
    col = X_tr_f[:,j]
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    X_tr_f[:,j] = np.clip(col, lo, hi)
    X_es_f[:,j] = np.clip(X_es_f[:,j], lo, hi)
    X_te_f[:,j] = np.clip(X_te_f[:,j], lo, hi)
print(f"After clip: range=[{X_tr_f.min():.2f}, {X_tr_f.max():.2f}]", flush=True)

# ========== LGBM Sanity ==========
print(f"\n{'='*60}\nLGBM on flattened manual seq\n{'='*60}", flush=True)
t0=time.time()
params = dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=50,
              feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,verbose=-1,seed=42,n_jobs=-1)
tr_ds = lgb.Dataset(X_tr_f, label=y_tr); es_ds = lgb.Dataset(X_es_f, label=y_es, reference=tr_ds)
bst = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(300)])
pv_te = bst.predict(X_te_f); pv_es = bst.predict(X_es_f)
auc_es_l = roc_auc_score(y_es, pv_es); auc_te_l = roc_auc_score(y_te, pv_te)
print(f"  LGBM: ES={auc_es_l:.4f} TE={auc_te_l:.4f} trees={bst.best_iteration} [{time.time()-t0:.0f}s]", flush=True)

# ========== MLP ==========
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

def train(m, Xtr, ytr, Xes, yes, Xte, yte, ep=20, lr=5e-4, wd=0.05, bs=512, pat=6, smooth=0.2):
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
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
    pv_te=evaluate(m,Xte); auc_te=roc_auc_score(yte,pv_te)
    return pv_te, pv_es, auc_te

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

# 5-seed MLP
print(f"\n{'='*60}\n5-seed MLP(768, 512, 256, d=0.6, wd=0.05)\n{'='*60}", flush=True)
hs=[768,512,256]; drop=0.6; wd=0.05
pvs_t=[]; pvs_e=[]
for s in [42,49,56,63,70]:
    torch.manual_seed(s); np.random.seed(s)
    m=MLP(FT,hs,drop)
    pv_t,pv_e,auc_t=train(m,X_tr_f,y_tr,X_es_f,y_es,X_te_f,y_te,ep=20,lr=5e-4,wd=wd,bs=512,pat=7,smooth=0.2)
    print(f"  seed{s}: TE={auc_t:.4f}", flush=True)
    pvs_t.append(pv_t); pvs_e.append(pv_e)
pv_ens_t=rank_agg(pvs_t); pv_ens_e=rank_agg(pvs_e)
do_eval(f'5-seed MLP({hs})',pv_ens_e,y_es,pv_ens_t,y_te)

# Also LGBM 3-seed
print(f"\n{'='*60}\n3-seed LGBM\n{'='*60}", flush=True)
pvs_te_l=[]; pvs_es_l=[]
for s in [42,49,56]:
    params['seed']=s
    tr_ds=lgb.Dataset(X_tr_f,label=y_tr); es_ds=lgb.Dataset(X_es_f,label=y_es,reference=tr_ds)
    bst=lgb.train(params,tr_ds,num_boost_round=5000,valid_sets=[es_ds],callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    pvs_te_l.append(bst.predict(X_te_f)); pvs_es_l.append(bst.predict(X_es_f))
do_eval('3-seed LGBM_manual', rank_agg(pvs_es_l), y_es, rank_agg(pvs_te_l), y_te)

print(f"\nDONE", flush=True)
