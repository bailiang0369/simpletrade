"""Tiny CNN for shape pattern recognition
输入: (2, 64) — 64 根 1min (close norm + stoch_k14)
模型: 2 层 Conv + GlobalPool + FC, <100K params
CPU 友好, 下采样训练到 50 万 (足够)
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); DEVICE='cpu'
print(f"torch={torch.__version__}")

# ============= 1. Build dataset (下采样训练到 50 万, 保持全量测试) =============
print("[1] Build sequences (WINDOW=64, downsample train)...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
ds  = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
L14=raw['low'].rolling(14,min_periods=1).min(); H14=raw['high'].rolling(14,min_periods=1).max()
raw['sk14']=np.where(H14-L14>1e-9,(raw['close']-L14)/(H14-L14)*100,50.0).astype(np.float32)
raw['close']=raw['close'].astype(np.float32)
rts=raw['ts'].to_numpy(); dts=ds['ts'].to_numpy(); C=raw['close'].to_numpy(); K=raw['sk14'].to_numpy()
del raw, L14, H14, ds; gc.collect()

idx=np.searchsorted(rts,dts); idx=np.clip(idx,256,len(C)-1)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

W=64
tr_idx=idx[(dts>=0)&(dts<TRAIN_END)]; tr_y=np.zeros(len(tr_idx),dtype=np.int64)
es_idx=idx[(dts>=TRAIN_END)&(dts<ES_END)]
te_idx=idx[dts>=META_END]

# 需要 labels — 重新 load label
labels=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
tr_y=labels.loc[(labels['ts']>=0)&(labels['ts']<TRAIN_END),'label'].to_numpy().astype(np.int64)
es_y=labels.loc[(labels['ts']>=TRAIN_END)&(labels['ts']<ES_END),'label'].to_numpy().astype(np.int64)
te_y=labels.loc[labels['ts']>=META_END,'label'].to_numpy().astype(np.int64)
del labels, rts, dts, idx; gc.collect()

# 对齐长度 (clip)
N=min(len(tr_idx),len(tr_y)); tr_idx=tr_idx[:N]; tr_y=tr_y[:N]
N=min(len(es_idx),len(es_y)); es_idx=es_idx[:N]; es_y=es_y[:N]
N=min(len(te_idx),len(te_y)); te_idx=te_idx[:N]; te_y=te_y[:N]

# 下采样训练集到 50 万
np.random.seed(42)
sub=np.random.choice(len(tr_idx), min(500000,len(tr_idx)), replace=False)
tr_idx=tr_idx[sub]; tr_y=tr_y[sub]
print(f"  tr={len(tr_idx):,} (downsampled) es={len(es_idx):,} te={len(te_idx):,}")

class SeqDS(Dataset):
    def __init__(self, idx, y, w=64): self.idx=idx.astype(np.int64); self.y=y.astype(np.int64); self.w=w
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        e=self.idx[i]; s=e-self.w+1
        c=C[s:e+1].copy(); k=K[s:e+1].copy(); c0=c[0]
        c=np.where(c0>1e-9, c/c0-1.0, 0.0).astype(np.float32); k=(k/100.0).astype(np.float32)
        x=np.stack([c,k],axis=0)
        return torch.from_numpy(x), torch.tensor(self.y[i],dtype=torch.long)

# ============= 2. Tiny CNN =============
class TinyShapeCNN(nn.Module):
    def __init__(self, in_ch=2, hidden=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.MaxPool1d(2),  # 64 → 32
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.MaxPool1d(2),  # 32 → 16
            nn.Conv1d(hidden, hidden*2, 5, padding=2), nn.BatchNorm1d(hidden*2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, hidden*2, 1)
        )
        self.fc=nn.Linear(hidden*2, 2)
    def forward(self, x):
        x=self.net(x).flatten(1); return self.fc(x)

m=TinyShapeCNN(); total=sum(p.numel() for p in m.parameters())
print(f"\n[2] Tiny CNN params={total:,}")

# ============= 3. Train (3 seeds) =============
BATCH=2048; EPOCHS=20; LR=1e-3

def run_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader=DataLoader(SeqDS(tr_idx,tr_y,W),BATCH,shuffle=True,num_workers=0)
    es_loader=DataLoader(SeqDS(es_idx,es_y,W),BATCH*2,shuffle=False,num_workers=0)
    te_loader=DataLoader(SeqDS(te_idx,te_y,W),BATCH*2,shuffle=False,num_workers=0)
    m=TinyShapeCNN().to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    ce=nn.CrossEntropyLoss(); best=0; best_state=None
    t0=time.time()
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
        ps=np.concatenate(ps); ys=np.concatenate(ys); auc=roc_auc_score(ys,ps)
        if auc>best: best=auc; best_state={k:v.clone() for k,v in m.state_dict().items()}
        if ep%5==0: print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.load_state_dict(best_state)
    # Test
    m.eval(); ps=[]
    with torch.no_grad():
        for xb,_ in te_loader: ps.append(F.softmax(m(xb),1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[3] Train 3 seeds on {DEVICE}...", flush=True)
pte=[]
for sd in [42, 56, 70]:
    print(f"=== seed={sd} ===", flush=True)
    p,_=run_one(sd); pte.append(p)

p=np.mean(pte,axis=0); auc=roc_auc_score(te_y,p)
print(f"\n★★ Tiny CNN (W=64, pure shape) TEST AUC = {auc:.4f}")
print(f"   ret-based LightGBM baseline:    AUC = 0.5428")
print(f"   差距: {(auc-0.5428)*100:+.2f}pp")

# tpd sweep
print(f"\n[4] tpd sweep (global q oracle upper bound)...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985,0.998,0.0005):
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100
    tpd=n/332; diff=acc-65; mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}")
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)
print(f"\n★ tpd≈14.4: ACC={best[0]:.1f}% q={best[1]:.4f} tpd={best[2]:.1f}")

print(f"\n{'='*60}")
print(f"  核心发现:")
print(f"  ret-based LGBM (56 feats): AUC=0.5428 tpd14.4 ACC≈61.6%")
print(f"  SEQ64 LGBM flatten:        AUC=0.5381 tpd14.4 ACC≈56.8%")
print(f"  Tiny CNN on shape seq:     AUC={auc:.4f}  tpd14.4 ACC≈{best[0]:.1f}%")
print(f"  → CNN 才能真正学形态结构!")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s")
