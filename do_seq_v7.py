"""v7: TCN + CNN + LSTM on v6 data WITHOUT flatten (pure sequential)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape}", flush=True)
C_IN = X_tr.shape[1]; W = X_tr.shape[2]
pos = y_tr.mean(); pw = np.where(y_tr>0.5,(1-pos)/pos,pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
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

class LSTMAttn(nn.Module):
    def __init__(self, C_in, h=96):
        super().__init__()
        self.lstm=nn.LSTM(C_in,h,2,batch_first=True,bidirectional=True,dropout=0.2)
        self.attn=nn.MultiheadAttention(h*2,4,batch_first=True,dropout=0.15)
        self.norm=nn.LayerNorm(h*2); self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Linear(h*2,h//2),nn.GELU(),nn.Dropout(0.25),nn.Linear(h//2,1))
    def forward(self,x):
        x=x.transpose(1,2); o,_=self.lstm(x)
        a,_=self.attn(o,o,o); o=self.norm(o+a)
        p=self.pool(o.transpose(1,2)).flatten(1)
        return self.head(p).squeeze(-1)

class Transformer(nn.Module):
    def __init__(self, C_in, d_model=96, nhead=4, nlayers=3, drop=0.2):
        super().__init__()
        self.inp=nn.Linear(C_in,d_model)
        self.pe=nn.Parameter(torch.randn(1,W,d_model)*0.02)
        enc_layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=d_model*4,dropout=drop,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(enc_layer,num_layers=nlayers)
        self.head=nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),nn.Dropout(0.25),nn.Linear(d_model//2,1))
    def forward(self,x):
        x=x.transpose(1,2); x=self.inp(x)+self.pe
        o=self.enc(x).mean(dim=1)
        return self.head(o).squeeze(-1)

def evaluate(m, X, bs=2048):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            x=torch.from_numpy(np.ascontiguousarray(X[i:i+bs]))
            pv.append(torch.sigmoid(m(x)).numpy())
    return np.concatenate(pv)

def train(name, m, Xtr, ytr, Xes, yes, Xte, yte,
          ep=25, lr=2e-3, wd=2e-3, bs=256, pat=10, smooth=0.1):
    np_=sum(p.numel() for p in m.parameters())
    print(f"\n  [{name}] params={np_:,}", flush=True)
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
            wb=torch.from_numpy(pw[bi]*rw[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb,weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); sch.step()
            tl+=loss.item(); nb+=1
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
        if e%2==0 or e<2:
            print(f"    ep{e+1:2d} loss={tl/max(nb,1):.4f} auc_es={auc_es:.4f} lr={opt.param_groups[0]['lr']:.1e}", flush=True)
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
    dt=pd.to_datetime(ts_te,unit='s',utc=True); month=dt.to_period('M').values
    all_m=sorted(pd.PeriodIndex(np.unique(month))); accs=[]; bad=0
    for mm in all_m:
        hm=month==mm; n=hm.sum(); k=max(1,int(n*0.01))
        a=yte[hm][np.argsort(-pv_te[hm])[:k]].mean()*100; accs.append(a); bad+=int(a<55)
    print(f"    monthly: {np.mean(accs):.1f}±{np.std(accs):.1f}  bad(<55)={bad}/{len(all_m)}", flush=True)

# ============ Run ============
print(f"\n{'='*60}\nR1: TCN_big (64→128→256→256→128)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m1, pv1_t, pv1_e = train('TCN_big', TCN(C_IN,[64,128,256,256,128],0.2),
                          X_tr, y_tr, X_es, y_es, X_te, y_te, ep=30, lr=2e-3, wd=2e-3, bs=256, pat=12)
do_eval('R1 TCN_big', pv1_e, y_es, pv1_t, y_te)

print(f"\n{'='*60}\nR2: TCN_mega (deeper, bigger)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m2, pv2_t, pv2_e = train('TCN_mega', TCN(C_IN,[64,128,256,512,256,128],0.25),
                          X_tr, y_tr, X_es, y_es, X_te, y_te, ep=35, lr=1.5e-3, wd=3e-3, bs=192, pat=14)
do_eval('R2 TCN_mega', pv2_e, y_es, pv2_t, y_te)

print(f"\n{'='*60}\nR3: LSTM+Attn (h=128)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m3, pv3_t, pv3_e = train('LSTM_big', LSTMAttn(C_IN,h=128),
                          X_tr, y_tr, X_es, y_es, X_te, y_te, ep=30, lr=1.5e-3, wd=3e-3, bs=128, pat=12)
do_eval('R3 LSTM_big', pv3_e, y_es, pv3_t, y_te)

print(f"\n{'='*60}\nR4: Transformer (d=96, 3L)\n{'='*60}", flush=True)
torch.manual_seed(42); np.random.seed(42)
m4, pv4_t, pv4_e = train('Transformer', Transformer(C_IN,d_model=96,nhead=4,nlayers=3,drop=0.2),
                          X_tr, y_tr, X_es, y_es, X_te, y_te, ep=30, lr=1e-3, wd=3e-3, bs=128, pat=12)
do_eval('R4 Transformer', pv4_e, y_es, pv4_t, y_te)

print(f"\n{'='*60}\nR5: 3-seed TCN_big rank ens\n{'='*60}", flush=True)
pvs_t=[]; pvs_e=[]
for s in [42,49,56]:
    torch.manual_seed(s); np.random.seed(s)
    _, pt, pe = train(f'TCN_s{s}', TCN(C_IN,[64,128,256,256,128],0.2),
                       X_tr, y_tr, X_es, y_es, X_te, y_te, ep=25, lr=2e-3, wd=2e-3, bs=256, pat=10)
    pvs_t.append(pt); pvs_e.append(pe)
do_eval('R5 TCN_3seed', rank_agg(pvs_e), y_es, rank_agg(pvs_t), y_te)

print(f"\n{'='*60}\nR6: TCN_mega×TCN_big rank\n{'='*60}", flush=True)
do_eval('R6 TCN_big+TCN_mega', rank_agg([pv1_e,pv2_e]), y_es, rank_agg([pv1_t,pv2_t]), y_te)

print(f"\n{'='*60}\nR7: TCN_big×TCN_mega×LSTM rank\n{'='*60}", flush=True)
do_eval('R7 TCN×TCN×LSTM', rank_agg([pv1_e,pv2_e,pv3_e]), y_es, rank_agg([pv1_t,pv2_t,pv3_t]), y_te)

print(f"\nDONE total {time.time()-__import__('time').time():.0f}s", flush=True)
