"""
TCN V4: 预计算序列(float16) + 手动batch + TCN 4层
CPU only, 1.5M train, 4 configs ensemble

目标: 超越 MLP AUC=0.543 → 0.55+, top1%=58.8% → 65%
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config

t_all = time.time()

def log(msg): print(msg, flush=True)

# ============ 1. 加载 + per-bar特征 ============
log("Loading raw data...")
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
ts = eth['ts'].to_numpy()
O_e=eth['open'].cast(pl.Float32).to_numpy(); H_e=eth['high'].cast(pl.Float32).to_numpy()
L_e=eth['low'].cast(pl.Float32).to_numpy(); C_e=eth['close'].cast(pl.Float32).to_numpy()
BV_e=eth['buy_vol'].cast(pl.Float32).to_numpy(); SV_e=eth['sell_vol'].cast(pl.Float32).to_numpy()
F_e=eth['funding'].cast(pl.Float32).to_numpy()
del eth
ts_b=btc['ts'].to_numpy(); C_b=btc['close'].cast(pl.Float32).to_numpy()
BV_b=btc['buy_vol'].cast(pl.Float32).to_numpy(); SV_b=btc['sell_vol'].cast(pl.Float32).to_numpy()
del btc
idx_b=np.searchsorted(ts_b,ts,side='right')-1; idx_b=np.clip(idx_b,0,len(C_b)-1)
C_b_a=C_b[idx_b]; BV_b_a=BV_b[idx_b]; SV_b_a=SV_b[idx_b]
del ts_b,C_b,BV_b,SV_b,idx_b
N = len(ts)

log("Computing per-bar features...")
lr_e=np.zeros(N,np.float32); lr_e[1:]=np.log(np.maximum(C_e[1:],1e-8)/np.maximum(C_e[:-1],1e-8))
lr_b=np.zeros(N,np.float32); lr_b[1:]=np.log(np.maximum(C_b_a[1:],1e-8)/np.maximum(C_b_a[:-1],1e-8))
range_e=(H_e-L_e)/(np.maximum(C_e,1e-8)); body_e=(C_e-O_e)/(np.maximum(O_e,1e-8))
bsi_e=(BV_e-SV_e)/(np.maximum(BV_e+SV_e,1e-8)); bsi_b=(BV_b_a-SV_b_a)/(np.maximum(BV_b_a+SV_b_a,1e-8))
vr=BV_e/(SV_e+1e-8)
FEAT=np.stack([lr_e,range_e,body_e,bsi_e,F_e,lr_b,bsi_b,vr],axis=1).astype(np.float32)
del lr_e,lr_b,range_e,body_e,bsi_e,bsi_b,vr,O_e,H_e,L_e,BV_e,SV_e,BV_b_a,SV_b_a
gc.collect()

# ============ 2. Label + 切分 ============
HORIZON=15; LOOKBACK=60
label=(C_e[HORIZON:]>C_e[:-HORIZON]).astype(np.int8)
del C_e; gc.collect()
VS=LOOKBACK; VE=N-HORIZON

tre=int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end=int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end=int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx=np.where(ts[VS:VE]<tre)[0]+VS
es_idx=np.where((ts[VS:VE]>=tre)&(ts[VS:VE]<es_end))[0]+VS
mv_idx=np.where((ts[VS:VE]>=es_end)&(ts[VS:VE]<meta_end))[0]+VS
te_idx=np.where(ts[VS:VE]>=meta_end)[0]+VS

np.random.seed(42); tr_sub=np.random.choice(tr_idx,1_500_000,replace=False); tr_sub.sort()
log(f"Splits: tr={len(tr_sub)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}")

# ============ 3. Global Norm + 预计算序列 ============
log("Global norm...")
np.random.seed(42); ns=np.random.choice(tr_idx,200_000,replace=False)
MU=FEAT[ns].mean(0); SD=FEAT[ns].std(0)+1e-6
FEAT_N=np.clip((FEAT-MU)/SD,-5,5).astype(np.float16)
del FEAT,ns,MU,SD; gc.collect()

def make_seq(idx_arr, chunk=300_000):
    out=np.empty((len(idx_arr),8,LOOKBACK),dtype=np.float16)
    for s in range(0,len(idx_arr),chunk):
        e=min(s+chunk,len(idx_arr)); ic=idx_arr[s:e]
        sc=np.stack([FEAT_N[i-LOOKBACK:i] for i in ic],axis=0)
        out[s:e]=sc.transpose(0,2,1)
    return out

t0=time.time()
tr_seq=make_seq(tr_sub); log(f"tr_seq: {tr_seq.shape} {tr_seq.nbytes/1e9:.2f}GB, {time.time()-t0:.1f}s")
t0=time.time()
es_seq=make_seq(es_idx); log(f"es_seq: {es_seq.shape} {es_seq.nbytes/1e6:.0f}MB, {time.time()-t0:.1f}s")

# MV/TE 现算 (eval时用)
log(f"FEAT_N kept for eval: {FEAT_N.nbytes/1e6:.0f}MB")

# ============ 4. TCN 模型 ============
class Chomp1d(nn.Module):
    def __init__(self,s): super().__init__(); self.s=s
    def forward(self,x): return x[:,:,:-self.s].contiguous() if self.s>0 else x

class TBlock(nn.Module):
    def __init__(self,ic,oc,k,d,drop=0.3):
        super().__init__(); p=(k-1)*d
        self.c1=nn.Conv1d(ic,oc,k,dilation=d,padding=p); self.ch1=Chomp1d(p); self.bn1=nn.BatchNorm1d(oc)
        self.c2=nn.Conv1d(oc,oc,k,dilation=d,padding=p); self.ch2=Chomp1d(p); self.bn2=nn.BatchNorm1d(oc)
        self.down=nn.Conv1d(ic,oc,1) if ic!=oc else None
        self.r=nn.ReLU(); self.dp=nn.Dropout(drop)
    def forward(self,x):
        o=self.r(self.bn1(self.ch1(self.c1(x)))); o=self.dp(o)
        o=self.r(self.bn2(self.ch2(self.c2(o)))); o=self.dp(o)
        res=x if self.down is None else self.down(x)
        return self.r(o+res)

class TCN(nn.Module):
    def __init__(self,in_ch=8,channels=[64,64,128,128],k=3,drop=0.3):
        super().__init__(); layers=[]; ic=in_ch
        for i,oc in enumerate(channels): layers.append(TBlock(ic,oc,k,2**i,drop)); ic=oc
        self.net=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(channels[-1],64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
    def forward(self,x): return self.head(self.net(x))

# ============ 5. 训练 ============
def train_one(tr_seq_arr, es_seq_arr, channels, dropout, seed,
              lr=3e-4, wd=1e-3, ls=0.15, batch=1024, epochs=15, patience=7):
    torch.manual_seed(seed); np.random.seed(seed)
    model=TCN(8,channels,k=3,drop=dropout)
    log(f"  Seed {seed}, params={sum(p.numel() for p in model.parameters()):,}, ch={channels}, drop={dropout}")
    
    es_x=torch.from_numpy(es_seq_arr.astype(np.float32))
    es_y=torch.tensor(label[es_idx],dtype=torch.float32)
    
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    
    best_auc=0; best_state=None; pat=0
    n=len(tr_seq_arr); n_b=n//batch
    
    for ep in range(epochs):
        t_ep=time.time(); model.train(); tl=0; perm=np.random.permutation(n)
        for b in range(n_b):
            idx=perm[b*batch:(b+1)*batch]
            xb=torch.from_numpy(tr_seq_arr[idx].astype(np.float32))
            yb=torch.tensor(label[tr_sub[idx]],dtype=torch.float32)
            yt=yb*(1-ls)+0.5*ls
            lg=model(xb).squeeze(-1)
            loss=F.binary_cross_entropy_with_logits(lg,yt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tl+=loss.item()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            ep_auc=roc_auc_score(es_y.numpy(), torch.sigmoid(model(es_x).squeeze(-1)).numpy())
        dt=time.time()-t_ep
        log(f"    Ep{ep+1:2d}: loss={tl/n_b:.4f} ES_AUC={ep_auc:.4f} {dt:.0f}s lr={sched.get_last_lr()[0]:.5f}")
        
        if ep_auc>best_auc:
            best_auc=ep_auc; best_state={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=patience: log(f"    Early stop"); break
    
    if best_state: model.load_state_dict(best_state)
    del es_x,es_y; gc.collect()
    return model, best_auc

# ============ 6. Eval ============
def get_preds(model, idx_arr, batch=2048):
    seq=make_seq(idx_arr)
    n=len(seq); preds=np.zeros(n,dtype=np.float32)
    model.eval()
    for s in range(0,n,batch):
        e=min(s+batch,n); xb=torch.from_numpy(seq[s:e].astype(np.float32))
        with torch.no_grad(): preds[s:e]=torch.sigmoid(model(xb).squeeze(-1)).numpy()
    del seq; gc.collect()
    return preds

def rolling_eval(preds, idx_arr, qs=[98,99,99.5]):
    ts_e=ts[idx_arr]; y=label[idx_arr]
    auc=roc_auc_score(y,preds)
    nd=(ts_e[-1]-ts_e[0])/86400
    res={}
    for q in qs:
        thrs=np.zeros(len(idx_arr))
        for i in range(len(idx_arr)):
            tnow=ts_e[i]; t30d=tnow-30*86400
            hm=(ts_e[:i]>=t30d)&(ts_e[:i]<tnow)
            thrs[i]=np.percentile(preds[hm],q) if hm.sum()>=100 else np.percentile(preds[:max(i,1)],q)
        sig=preds>thrs
        acc=y[sig].mean() if sig.sum()>0 else 0
        res[q]=(acc,sig.sum()/nd,int(sig.sum()))
    return auc,res

# ============ 7. 主实验 ============
configs=[
    ([64,64,128,128],0.3),
    ([32,64,128,128],0.3),
    ([64,128,256,256],0.35),
    ([64,64,64,64],0.25),
]

te_preds_all=[]; te_y_all=label[te_idx]; ts_te_all=ts[te_idx]

for ci,(ch,dp) in enumerate(configs):
    log(f"\n{'='*55}"); log(f"Config {ci}: ch={ch}, drop={dp}"); log(f"{'='*55}")
    model,best_auc=train_one(tr_seq,es_seq,ch,dp,seed=42+ci*7)
    log(f"  Best ES AUC: {best_auc:.4f}")
    
    log("  Meta-Val eval...")
    mv_preds=get_preds(model,mv_idx)
    mv_auc,mv_res=rolling_eval(mv_preds,mv_idx)
    log(f"  MV AUC={mv_auc:.4f}")
    for q,(a,t,n) in mv_res.items(): log(f"    q{q}: acc={a:.4f} tpd={t:.1f} n={n}")
    
    log("  Test eval...")
    te_preds=get_preds(model,te_idx)
    te_preds_all.append(te_preds)
    te_auc,te_res=rolling_eval(te_preds,te_idx)
    log(f"  TE AUC={te_auc:.4f}")
    for q,(a,t,n) in te_res.items(): log(f"    q{q}: acc={a:.4f} tpd={t:.1f} n={n}")
    
    del model,mv_preds,te_preds; gc.collect()

# ============ 8. Ensemble ============
log(f"\n{'='*55}"); log("ENSEMBLE"); log(f"{'='*55}")
ens=np.mean(te_preds_all,axis=0)
ens_auc=roc_auc_score(te_y_all,ens)
log(f"Ensemble TE AUC: {ens_auc:.4f}")
nd=(ts_te_all[-1]-ts_te_all[0])/86400
for q in [98,99,99.5,99.7]:
    thrs=np.zeros(len(te_idx))
    for i in range(len(te_idx)):
        tnow=ts_te_all[i]; t30d=tnow-30*86400
        hm=(ts_te_all[:i]>=t30d)&(ts_te_all[:i]<tnow)
        thrs[i]=np.percentile(ens[hm],q) if hm.sum()>=100 else np.percentile(ens[:max(i,1)],q)
    sig=ens>thrs
    acc=te_y_all[sig].mean() if sig.sum()>0 else 0
    log(f"  q={q}: acc={acc:.4f} tpd={sig.sum()/nd:.1f} n={sig.sum()}")

log(f"\nTotal: {time.time()-t_all:.0f}s = {(time.time()-t_all)/60:.1f}min")
