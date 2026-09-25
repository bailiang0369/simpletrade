"""修 CNN 归一化: close_zscore (保留位置) + close_ret (保留方向) + stoch_k
WINDOW=128, 4 channels, Tiny CNN v2
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'
print(f"torch={torch.__version__}")

print("[1] Build sequences (W=128, 4ch)...", flush=True)
raw=pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
ds=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
L14=raw['low'].rolling(14,min_periods=1).min(); H14=raw['high'].rolling(14,min_periods=1).max()
raw['sk14']=np.where(H14-L14>1e-9,(raw['close']-L14)/(H14-L14)*100,50.0).astype(np.float32)
C=raw['close'].to_numpy().astype(np.float64)
# Rolling zscore (240min = 4h)
rm=pd.Series(C).rolling(240,min_periods=60).mean().to_numpy()
rs=pd.Series(C).rolling(240,min_periods=60).std().to_numpy()
Cz=np.where(rs>1e-9,(C-rm)/rs,0.0).astype(np.float32)
# Close 1min ret
Cr=np.zeros(len(C),dtype=np.float32); Cr[1:]=(C[1:]-C[:-1])/np.maximum(C[:-1],1e-9)
# Stoch_k 0-100 → 0-1
Sk=(raw['sk14'].to_numpy()/100.0).astype(np.float32)
# Close normalized by window-start (形态相对变化)
rts=raw['ts'].to_numpy(); dts=ds['ts'].to_numpy()
del raw, L14, H14; gc.collect()

idx=np.searchsorted(rts,dts); idx=np.clip(idx,512,len(C)-1)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
W=128

tr_idx=idx[(dts>=0)&(dts<TRAIN_END)]
es_idx=idx[(dts>=TRAIN_END)&(dts<ES_END)]
te_idx=idx[dts>=META_END]
labels=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
tr_y=labels.loc[(labels['ts']>=0)&(labels['ts']<TRAIN_END),'label'].to_numpy().astype(np.int64)
es_y=labels.loc[(labels['ts']>=TRAIN_END)&(labels['ts']<ES_END),'label'].to_numpy().astype(np.int64)
te_y=labels.loc[labels['ts']>=META_END,'label'].to_numpy().astype(np.int64)
del labels, rts, dts, idx; gc.collect()
N=min(len(tr_idx),len(tr_y)); tr_idx=tr_idx[:N]; tr_y=tr_y[:N]
N=min(len(es_idx),len(es_y)); es_idx=es_idx[:N]; es_y=es_y[:N]
N=min(len(te_idx),len(te_y)); te_idx=te_idx[:N]; te_y=te_y[:N]
np.random.seed(42); sub=np.random.choice(len(tr_idx),min(500000,len(tr_idx)),replace=False)
tr_idx=tr_idx[sub]; tr_y=tr_y[sub]
print(f"  tr={len(tr_idx):,} es={len(es_idx):,} te={len(te_idx):,} W={W} 4ch")

class SeqDS(Dataset):
    def __init__(self,idx,y,w=128): self.idx=idx.astype(np.int64); self.y=y.astype(np.int64); self.w=w
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        e=self.idx[i]; s=e-self.w+1
        c=C[s:e+1]; cz=Cz[s:e+1]; cr=Cr[s:e+1]; sk=Sk[s:e+1]
        c0=c[0]; cnorm=np.where(c0>1e-9,c/c0-1.0,0.0).astype(np.float32)
        x=np.stack([cz.astype(np.float32),cnorm,cr,sk],axis=0)  # (4, W)
        return torch.from_numpy(x), torch.tensor(self.y[i],dtype=torch.long)

class TinyV2(nn.Module):
    def __init__(self, in_ch=4, hidden=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.MaxPool1d(2),  # 128 → 64
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.MaxPool1d(2),  # 64 → 32
            nn.Conv1d(hidden, hidden*2, 5, padding=2), nn.BatchNorm1d(hidden*2), nn.SiLU(),
            nn.MaxPool1d(2),  # 32 → 16
            nn.Conv1d(hidden*2, hidden*2, 5, padding=2), nn.BatchNorm1d(hidden*2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc=nn.Sequential(nn.Linear(hidden*2,64),nn.SiLU(),nn.Dropout(0.3),nn.Linear(64,2))
    def forward(self,x): x=self.net(x).flatten(1); return self.fc(x)

m=TinyV2(); tp=sum(p.numel() for p in m.parameters()); print(f"\n[2] TinyV2 CNN params={tp:,}")

BATCH=2048; EPOCHS=25; LR=1e-3
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader=DataLoader(SeqDS(tr_idx,tr_y,W),BATCH,shuffle=True,num_workers=0)
    es_loader=DataLoader(SeqDS(es_idx,es_y,W),BATCH*2,shuffle=False,num_workers=0)
    te_loader=DataLoader(SeqDS(te_idx,te_y,W),BATCH*2,shuffle=False,num_workers=0)
    m=TinyV2(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); ce=nn.CrossEntropyLoss()
    best=0; best_state=None; t0=time.time()
    for ep in range(1,EPOCHS+1):
        m.train(); tl=0; n=0
        for xb,yb in tr_loader:
            opt.zero_grad(); out=m(xb); loss=ce(out,yb); loss.backward(); opt.step()
            tl+=loss.item()*len(xb); n+=len(xb)
        sch.step()
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb,yb in es_loader:
                ps.append(F.softmax(m(xb),1)[:,1].numpy()); ys.append(yb.numpy())
        auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
        if auc>best: best=auc; best_state={k:v.clone() for k,v in m.state_dict().items()}
        if ep%5==0: print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.load_state_dict(best_state); m.eval(); ps=[]
    with torch.no_grad():
        for xb,_ in te_loader: ps.append(F.softmax(m(xb),1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[3] Train 3 seeds...")
pte=[]
for sd in [42,56,70]:
    print(f"=== seed={sd} ===", flush=True)
    p,_=run(sd); pte.append(p)
p=np.mean(pte,axis=0); auc=roc_auc_score(te_y,p)
print(f"\n★★ TinyV2 CNN TEST AUC = {auc:.4f}")
print(f"   ret-based LGBM (500k):  AUC = 0.5431")
print(f"   {(auc-0.5431)*100:+.2f}pp")

# 简单 tpd sweep
print(f"\n  tpd vicinity:")
for q in [0.990,0.991,0.992,0.993,0.994,0.995]:
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm; n=tm.sum()
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100; tpd=n/332
    if 10<=tpd<=20: print(f"    q={q:.3f} tpd={tpd:5.1f} ACC={acc:5.1f}%")
print(f"\nTOTAL: {time.time()-T0:.0f}s")
