"""v8: Extreme regularization. Small models, heavy drop, big wd, fast stop."""
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
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape}", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]
gc.collect()

# ============ Tiny models ============
class TinyTCN(nn.Module):
    def __init__(self, C_in, chs=[32,48,32], drop=0.4):
        super().__init__()
        prev=C_in; d=1; blocks=[]
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev,c,3,padding=2*d,dilation=d), nn.GELU(),
                nn.Conv1d(c,c,3,padding=2*d,dilation=d), nn.GELU(), nn.Dropout(drop)))
            prev=c; d*=2
        self.blocks=nn.ModuleList(blocks)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Dropout(drop), nn.Linear(chs[-1],1))
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

class TinyMLP(nn.Module):
    def __init__(self, ft, hs=256, drop=0.4):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),
            nn.Linear(ft,hs), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hs,hs//2), nn.GELU(), nn.Dropout(drop), nn.Linear(hs//2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class TinyLSTM(nn.Module):
    def __init__(self, C_in, h=32, drop=0.4):
        super().__init__()
        self.lstm=nn.LSTM(C_in, h, 1, batch_first=True, bidirectional=True, dropout=drop)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Dropout(drop), nn.Linear(h*2,1))
    def forward(self,x):
        x=x.transpose(1,2); o,_=self.lstm(x)
        p=self.pool(o.transpose(1,2)).flatten(1)
        return self.head(p).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_early(name, m, Xtr, ytr, Xes, yes, Xte, yte,
                ep=15, lr=1e-3, wd=1e-2, bs=512, pat=5, smooth=0.15, jitter=0.05):
    np_=sum(p.numel() for p in m.parameters())
    print(f"\n  [{name}] params={np_:,} lr={lr} wd={wd} drop_in_model=yes smooth={smooth} jitter={jitter}", flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; t0=time.time(); best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            if jitter>0: xb=xb+torch.randn_like(xb)*jitter
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),10.0); opt.step()
            tl+=loss.item(); nb+=1
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        pv_tr=evaluate(m,Xtr[:5000]); auc_tr=roc_auc_score(ytr[:5000],pv_tr)
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
        print(f"    ep{e+1:2d} loss={tl/max(nb,1):.4f} trAUC={auc_tr:.4f} ES={auc_es:.4f}", flush=True)
    if bst: m.load_state_dict(bst)
    pv_te=evaluate(m,Xte); pv_es=evaluate(m,Xes)
    auc_te=roc_auc_score(yte,pv_te)
    print(f"    → TE={auc_te:.4f} (best ES={best:.4f} @ep{best_ep}) [{time.time()-t0:.0f}s]", flush=True)
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
FT = C_IN * W

# Test 1: Tiny MLP, heavy reg
print(f"\n{'='*60}\nT1: TinyMLP(256, wd=0.05, drop=0.5, smooth=0.2)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m1,pv1_t,pv1_e=train_early('TinyMLP',TinyMLP(FT,256,0.5),X_tr,y_tr,X_es,y_es,X_te,y_te,ep=20,lr=5e-4,wd=0.05,bs=512,pat=6,smooth=0.2,jitter=0.05)
do_eval('T1 TinyMLP',pv1_e,y_es,pv1_t,y_te)

# Test 2: Tiny TCN
print(f"\n{'='*60}\nT2: TinyTCN([32,48,32], wd=0.05, drop=0.5)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m2,pv2_t,pv2_e=train_early('TinyTCN',TinyTCN(C_IN,[32,48,32],0.5),X_tr,y_tr,X_es,y_es,X_te,y_te,ep=20,lr=5e-4,wd=0.05,bs=512,pat=6,smooth=0.2,jitter=0.05)
do_eval('T2 TinyTCN',pv2_e,y_es,pv2_t,y_te)

# Test 3: Tiny LSTM
print(f"\n{'='*60}\nT3: TinyLSTM(h=32, wd=0.05, drop=0.5)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m3,pv3_t,pv3_e=train_early('TinyLSTM',TinyLSTM(C_IN,32,0.5),X_tr,y_tr,X_es,y_es,X_te,y_te,ep=20,lr=5e-4,wd=0.05,bs=256,pat=6,smooth=0.2,jitter=0.05)
do_eval('T3 TinyLSTM',pv3_e,y_es,pv3_t,y_te)

# Test 4: TinyTCN with higher lr
print(f"\n{'='*60}\nT4: TinyTCN lr=2e-3 wd=0.01\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m4,pv4_t,pv4_e=train_early('TinyTCN2',TinyTCN(C_IN,[32,48,32],0.4),X_tr,y_tr,X_es,y_es,X_te,y_te,ep=20,lr=2e-3,wd=0.01,bs=512,pat=6,smooth=0.15,jitter=0.03)
do_eval('T4 TinyTCN_lr2e-3',pv4_e,y_es,pv4_t,y_te)

# Test 5: TinyTCN bigger channels
print(f"\n{'='*60}\nT5: TinyTCN([48,64,48,32], wd=0.02, drop=0.4)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m5,pv5_t,pv5_e=train_early('TinyTCN3',TinyTCN(C_IN,[48,64,48,32],0.4),X_tr,y_tr,X_es,y_es,X_te,y_te,ep=20,lr=1e-3,wd=0.02,bs=512,pat=6,smooth=0.15,jitter=0.03)
do_eval('T5 TinyTCN_bigger',pv5_e,y_es,pv5_t,y_te)

# Test 6: Best of T1-T5 ens
print(f"\n{'='*60}\nT6: Best 2-of-5 rank ens\n{'='*60}", flush=True)
# Pick best ES (assuming T2 or T5)
all_es=[pv1_e,pv2_e,pv3_e,pv4_e,pv5_e]; all_te=[pv1_t,pv2_t,pv3_t,pv4_t,pv5_t]
auc_es_list=[roc_auc_score(y_es,np.nan_to_num(p,nan=0.5)) for p in all_es]
order=np.argsort(auc_es_list)[::-1]
print(f"  auc_es rank: {[(i+1, f'{auc_es_list[i]:.4f}') for i in order]}", flush=True)
best2=[all_es[order[0]],all_es[order[1]]]; best2te=[all_te[order[0]],all_te[order[1]]]
do_eval('T6 best2 rank',rank_agg(best2),y_es,rank_agg(best2te),y_te)

print(f"\nDONE total {time.time()-__import__('time').time():.0f}s", flush=True)
