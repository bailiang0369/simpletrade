"""金字塔 CNN — 用 mmap 懒加载, 分批 train/eval"""
import numpy as np, pandas as pd, time, gc, sys, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'
print(f"DEVICE={DEVICE}", flush=True)

STEPS=[1,2,4,8]; W=100; NL=len(STEPS); NPY='data/pyramid'; os.makedirs(NPY,exist_ok=True)

# ============= 1. 构建 pyramid tensors (存 .npy 用 mmap) =============
print("[1] Build + save pyramid tensors...", flush=True)
raw=pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
C=raw['close'].to_numpy().astype(np.float64); H=raw['high'].to_numpy().astype(np.float64)
L=raw['low'].to_numpy().astype(np.float64); rts=raw['ts'].to_numpy().astype(np.int64); N=len(C)
rm=pd.Series(C).rolling(240,min_periods=60).mean().to_numpy(); rs=pd.Series(C).rolling(240,min_periods=60).std().to_numpy()
Cz=np.where(rs>1e-9,(C-rm)/rs,0.0).astype(np.float32)
L14=pd.Series(L).rolling(14,min_periods=1).min().to_numpy(); H14=pd.Series(H).rolling(14,min_periods=1).max().to_numpy()
Sk=np.where(H14-L14>1e-9,(C-L14)/(H14-L14)*100,50.0).astype(np.float32)
del raw,rm,rs,L14,H14,H,L,C; gc.collect()

ds=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
dts=ds['ts'].to_numpy().astype(np.int64); labels=ds['label'].to_numpy().astype(np.int64)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
del ds; gc.collect()
idx=np.searchsorted(rts,dts); idx=np.clip(idx,1000,N-1); del rts; gc.collect()

def split(lo,hi): m=(dts>=lo)&(dts<hi); return idx[m], labels[m]
tr_idx,tr_y=split(0,TRAIN_END); es_idx,es_y=split(TRAIN_END,ES_END); te_idx,te_y=split(META_END,10**18)
del idx,dts,labels; gc.collect()
np.random.seed(42); sub=np.random.choice(len(tr_idx),min(500000,len(tr_idx)),replace=False)
tr_idx,tr_y=tr_idx[sub],tr_y[sub]
print(f"  tr={len(tr_idx):,}, es={len(es_idx):,}, te={len(te_idx):,}", flush=True)

def save_py(idx_arr, name):
    """分批构建 + 追加写 .npy"""
    Ns=len(idx_arr); CH=5000; t0=time.time()
    for s in range(0,Ns,CH):
        e=min(s+CH,Ns)
        out=np.zeros((e-s,NL,2,W),dtype=np.float32)
        for li,step in enumerate(STEPS):
            si=idx_arr[s:e]-(W-1)*step
            for j in range(e-s):
                out[j,li,0]=Cz[si[j]:idx_arr[s:e][j]+1:step]
                out[j,li,1]=Sk[si[j]:idx_arr[s:e][j]+1:step]/100.0
        if s==0: np.save(f'{NPY}/{name}.npy', out)     # create
        else:
            existing=np.load(f'{NPY}/{name}.npy'); np.save(f'{NPY}/{name}.npy', np.vstack([existing,out])); del existing
    np.save(f'{NPY}/{name}_y.npy', np.load(f'{NPY}/{name}_y.npy') if os.path.exists(f'{NPY}/{name}_y.npy') else np.array([]))  # placeholder
    print(f"  {name}: {Ns} rows ({time.time()-t0:.0f}s)", flush=True)

# 直接一次性存, 分块 append 用 vstack 太慢. 用纯 numpy 预分配.
for sname, sidx in [('tr',tr_idx),('es',es_idx),('te',te_idx)]:
    Ns=len(sidx); t0=time.time()
    out=np.zeros((Ns,NL,2,W),dtype=np.float32)
    for li,step in enumerate(STEPS):
        si=sidx-(W-1)*step
        print(f"    {sname} layer step={step}...", flush=True)
        for j in range(Ns):
            out[j,li,0]=Cz[si[j]:sidx[j]+1:step]
            out[j,li,1]=Sk[si[j]:sidx[j]+1:step]/100.0
    np.save(f'{NPY}/{sname}.npy', out); del out; gc.collect()
    print(f"  {sname}: saved ({time.time()-t0:.0f}s)", flush=True)

np.save(f'{NPY}/tr_y.npy', tr_y); np.save(f'{NPY}/es_y.npy', es_y); np.save(f'{NPY}/te_y.npy', te_y)
del Cz,Sk; gc.collect()

# ============= 2. Dataset =============
class PyDS(Dataset):
    def __init__(self, py, y): self.py=py; self.y=y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        x=self.py[i].reshape(-1,W)  # (4,2,100) → (8,100)
        return torch.from_numpy(x), torch.tensor(self.y[i],dtype=torch.long)

class PyramidCNN(nn.Module):
    def __init__(self,nl=4,h=24):
        super().__init__(); self.nl=nl
        self.enc=nn.Sequential(
            nn.Conv1d(2,h,7,padding=3),nn.BatchNorm1d(h),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h,h,5,padding=2),nn.BatchNorm1d(h),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h,h*2,5,padding=2),nn.BatchNorm1d(h*2),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h*2,h*2,3,padding=1),nn.BatchNorm1d(h*2),nn.SiLU(),nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Sequential(nn.Linear(h*2*nl,64),nn.SiLU(),nn.Dropout(0.4),nn.Linear(64,2))
    def forward(self,x):
        B=x.shape[0]; x=x.view(B*self.nl,2,-1)
        return self.fc(self.enc(x).view(B,self.nl,-1).flatten(1))

tp=sum(p.numel() for p in PyramidCNN().parameters()); print(f"\n[2] CNN params={tp:,}", flush=True)

# ============= 3. Train (mmap 懒加载) =============
BATCH=2048; EPOCHS=10; LR=1e-3

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    # 每次 run 都 fresh load, 避免内存泄漏
    Py_tr=np.load(f'{NPY}/tr.npy', mmap_mode='r'); y_tr=np.load(f'{NPY}/tr_y.npy')
    Py_es=np.load(f'{NPY}/es.npy', mmap_mode='r'); y_es=np.load(f'{NPY}/es_y.npy')
    Py_te=np.load(f'{NPY}/te.npy', mmap_mode='r'); y_te=np.load(f'{NPY}/te_y.npy')
    
    tr_l=DataLoader(PyDS(Py_tr,y_tr),BATCH,shuffle=True,num_workers=0,pin_memory=False)
    es_l=DataLoader(PyDS(Py_es,y_es),BATCH*2,shuffle=False,num_workers=0,pin_memory=False)
    te_l=DataLoader(PyDS(Py_te,y_te),BATCH*2,shuffle=False,num_workers=0,pin_memory=False)
    
    m=PyramidCNN(NL); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); best=0; best_state=None; t0=time.time()
    for ep in range(1,EPOCHS+1):
        m.train(); tl=0; n=0
        for xb,yb in tr_l: opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step(); tl+=loss.item()*len(xb); n+=len(xb)
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb,yb in es_l: ps.append(F.softmax(m(xb),1)[:,1].numpy()); ys.append(yb.numpy())
        auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
        if auc>best: best=auc; best_state={k:v.clone() for k,v in m.state_dict().items()}
        print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.load_state_dict(best_state); m.eval(); ps=[]
    with torch.no_grad():
        for xb,_ in te_l: ps.append(F.softmax(m(xb),1)[:,1].numpy())
    del Py_tr,Py_es,Py_te,y_tr,y_es,y_te; gc.collect()
    return np.concatenate(ps), best, np.load(f'{NPY}/te_y.npy')

print(f"\n[3] Train 3 seeds FAST pyramid (mmap)...", flush=True)
pte=[]
for sd in [42,56,70]:
    print(f"=== seed={sd} ===", flush=True)
    p,bes,te_y = run(sd); pte.append(p); print(f"  es_best={bes:.4f}", flush=True)

p=np.mean(pte,axis=0); auc=roc_auc_score(te_y,p)
print(f"\n{'='*60}")
print(f"  ★ FINAL FAST PYRAMID CNN  TEST AUC = {auc:.4f}")
print(f"    vs LGBM ret+negw (0.5428): {(auc-0.5428)*100:+.2f}pp")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s", flush=True)
