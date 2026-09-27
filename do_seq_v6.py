"""v6: TCN + MLP + LGBM on v6 data (global robust z-score, 14 channels)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape} range=[{X_tr.min():.2f}, {X_tr.max():.2f}]", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]
FT = C_IN * W
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_f = X_es.reshape(len(X_es), -1)
X_te_f = X_te.reshape(len(X_te), -1)

pos = y_tr.mean(); pw = np.where(y_tr>0.5,(1-pos)/pos,pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
sw = pw * rw
gc.collect()

# ============ Models ============
class TCN(nn.Module):
    def __init__(self, C_in, chs=[64,128,256,128], drop=0.2):
        super().__init__()
        prev=C_in; d=1; blocks=[]
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev,c,3,padding=2*d,dilation=d), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c,c,3,padding=2*d,dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            prev=c; d*=2
        self.blocks=nn.ModuleList(blocks)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Linear(chs[-1],chs[-1]//2), nn.GELU(), nn.Dropout(0.25), nn.Linear(chs[-1]//2,1))
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

class CNN2D(nn.Module):
    def __init__(self, C_in, chs=[32,64,128,64], drop=0.2):
        super().__init__()
        prev=1; blocks=[]  # treat as grayscale: (N, 1, C_in, W)
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv2d(prev,c,3,padding=1), nn.BatchNorm2d(c), nn.GELU(),
                nn.Conv2d(c,c,3,padding=1), nn.BatchNorm2d(c), nn.GELU(), nn.MaxPool2d(2), nn.Dropout(drop)))
            prev=c
        self.blocks=nn.ModuleList(blocks)
        with torch.no_grad():
            z=torch.zeros(1,1,C_IN,W)
            for b in self.blocks: z=b(z)
        flat = z.numel()
        self.head=nn.Sequential(nn.Linear(flat,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,1))
    def forward(self,x):
        x=x.unsqueeze(1)
        for b in self.blocks: x=b(x)
        return self.head(x.flatten(1)).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train(name, m, Xtr, ytr, Xes, yes, ep=25, lr=3e-3, wd=1e-3, bs=256, pat=10, smooth=0.1):
    np_=sum(p.numel() for p in m.parameters())
    print(f"\n  model={name} params={np_:,}", flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps_per_ep=(len(Xtr)+bs-1)//bs
    sch=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep*steps_per_ep, pct_start=0.1)
    best=0.0; bst=None; ni=0; t0=time.time(); best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            wb=torch.from_numpy(sw[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
            tl+=loss.item(); nb+=1
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        cur_lr=opt.param_groups[0]['lr']
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
        if e%3==0 or e<3:
            print(f"    ep{e+1:2d} loss={tl/max(nb,1):.4f} auc_es={auc_es:.4f} lr={cur_lr:.1e}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te=evaluate(m,X_te if Xtr.shape==X_te.shape else X_te_f); pv_es=evaluate(m,Xes)
    auc_te=roc_auc_score(y_te,pv_te)
    print(f"    → auc_te={auc_te:.4f} (best es={best:.4f} @ep{best_ep}) [{time.time()-t0:.0f}s]", flush=True)
    return m, pv_te, pv_es

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es=roc_auc_score(yes,np.nan_to_num(pv_es,nan=0.5))
    auc_te=roc_auc_score(yte,np.nan_to_num(pv_te,nan=0.5))
    DAYS=(ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5,1.0,2.0,3.0,5.0,10.0]:
        k=max(1,int(len(pv_te)*pct/100))
        acc=yte[np.argsort(-pv_te)[:k]].mean()*100; tpd=k/DAYS
        flag='🏆' if pct==1.0 and acc>=60 else ''
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# ============ Run ============
# First sanity check: LGBM on flattened v6
print(f"\n{'='*60}\nSANITY: LightGBM on flattened v6 seq\n{'='*60}", flush=True)
t0=time.time()
lgb_params=dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=50,
                feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=5,verbose=-1,seed=42,n_jobs=-1)
tr_ds=lgb.Dataset(X_tr_f,label=y_tr,weight=sw); es_ds=lgb.Dataset(X_es_f,label=y_es,reference=tr_ds)
bst=lgb.train(lgb_params,tr_ds,num_boost_round=5000,valid_sets=[es_ds],callbacks=[lgb.early_stopping(200),lgb.log_evaluation(300)])
pv_te_lgbm=bst.predict(X_te_f); pv_es_lgbm=bst.predict(X_es_f)
auc_es_l=roc_auc_score(y_es,pv_es_lgbm); auc_te_l=roc_auc_score(y_te,pv_te_lgbm)
print(f"  LGBM ES={auc_es_l:.4f} TE={auc_te_l:.4f} trees={bst.best_iteration} [{time.time()-t0:.0f}s]", flush=True)

# 3-seed LGBM
pvs_te_l=[]; pvs_es_l=[]
for s in [42,49,56]:
    lgb_params['seed']=s
    tr_ds=lgb.Dataset(X_tr_f,label=y_tr,weight=sw); es_ds=lgb.Dataset(X_es_f,label=y_es,reference=tr_ds)
    bst=lgb.train(lgb_params,tr_ds,num_boost_round=5000,valid_sets=[es_ds],callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    pvs_te_l.append(bst.predict(X_te_f)); pvs_es_l.append(bst.predict(X_es_f))
pv_lgbm_ens_te=rank_agg(pvs_te_l); pv_lgbm_ens_es=rank_agg(pvs_es_l)
auc_es_le=roc_auc_score(y_es,pv_lgbm_ens_es); auc_te_le=roc_auc_score(y_te,pv_lgbm_ens_te)
print(f"  LGBM 3-seed ES={auc_es_le:.4f} TE={auc_te_le:.4f}", flush=True)
do_eval('LGBM_3seed_v6_flat',pv_lgbm_ens_es,y_es,pv_lgbm_ens_te,y_te)

# TCN
print(f"\n{'='*60}\nR1: TCN_big on v6\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m_tcn=TCN(C_IN,[64,128,256,256,128],0.2)
m1,pv1_t,pv1_e=train('TCN_big_v6',m_tcn,X_tr,y_tr,X_es,y_es,ep=30,lr=2e-3,wd=2e-3,bs=256,pat=10,smooth=0.1)
do_eval('TCN_big_v6',pv1_e,y_es,pv1_t,y_te)

# 2D CNN
print(f"\n{'='*60}\nR2: 2D CNN on v6\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m_cnn=CNN2D(C_IN,[32,64,128,64],0.2)
m2,pv2_t,pv2_e=train('CNN2D_v6',m_cnn,X_tr,y_tr,X_es,y_es,ep=30,lr=2e-3,wd=2e-3,bs=256,pat=10,smooth=0.1)
do_eval('CNN2D_v6',pv2_e,y_es,pv2_t,y_te)

# 3-seed TCN
print(f"\n{'='*60}\nR3: 3-seed TCN_big rank\n{'='*60}", flush=True)
pvs_t=[]; pvs_e=[]
for s in [42,49,56]:
    torch.manual_seed(s); np.random.seed(s)
    m=TCN(C_IN,[64,128,256,256,128],0.2)
    _,pt,pe=train(f'TCN_s{s}',m,X_tr,y_tr,X_es,y_es,ep=25,lr=2e-3,wd=2e-3,bs=256,pat=8,smooth=0.1)
    pvs_t.append(pt); pvs_e.append(pe)
do_eval('TCN_3seed',rank_agg(pvs_e),y_es,rank_agg(pvs_t),y_te)

# TCN+CNN2D ens
print(f"\n{'='*60}\nR4: TCN+CNN2D+LGBM rank\n{'='*60}", flush=True)
do_eval('TCN+CNN2D+LGBM',rank_agg([pv1_e,pv2_e,pv_lgbm_ens_es]),y_es,rank_agg([pv1_t,pv2_t,pv_lgbm_ens_te]),y_te)

print(f"\nTOTAL {time.time()-__import__('time').time():.0f}s", flush=True)
