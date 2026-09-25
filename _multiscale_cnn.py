"""多尺度 CNN: 同时看 1min×100 + 5min×100, 学形态共振
输入: 4 channels — 1min close_z, 1min sk, 5min close_z, 5min sk
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'
print(f"torch={torch.__version__}")

# ============= 1. Build 1min + 5min features =============
print("[1] Build 1min + 5min...", flush=True)
raw1 = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
ds   = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')

# 5min resample
print("  1min → 5min resample...", flush=True)
raw1['dt']=pd.to_datetime(raw1['ts'], unit='s')
raw5 = raw1.set_index('dt').resample('5min').agg({'close':'last','high':'max','low':'max'}).dropna().reset_index()
raw5['ts']=(raw5['dt'].astype(np.int64)//10**9).astype(np.int64)
del raw1['dt']; gc.collect()

def build_seq_cols(df, name):
    L14=df['low'].rolling(14,min_periods=1).min(); H14=df['high'].rolling(14,min_periods=1).max()
    df[f'sk_{name}']=np.where(H14-L14>1e-9,(df['close']-L14)/(H14-L14)*100,50.0).astype(np.float32)
    C=df['close'].to_numpy().astype(np.float64)
    rm=pd.Series(C).rolling(240,min_periods=60).mean().to_numpy()
    rs=pd.Series(C).rolling(240,min_periods=60).std().to_numpy()
    df[f'cz_{name}']=np.where(rs>1e-9,(C-rm)/rs,0.0).astype(np.float32)
    df[f'c_{name}']=C.astype(np.float32)
    return df

raw1=build_seq_cols(raw1,'1m'); raw5=build_seq_cols(raw5,'5m')
print(f"  raw1={raw1.shape} raw5={raw5.shape}", flush=True)

r1_ts=raw1['ts'].to_numpy(); r5_ts=raw5['ts'].to_numpy()
C1z=raw1['cz_1m'].to_numpy(); S1k=raw1['sk_1m'].to_numpy(); C1c=raw1['c_1m'].to_numpy()
C5z=raw5['cz_5m'].to_numpy(); S5k=raw5['sk_5m'].to_numpy(); C5c=raw5['c_5m'].to_numpy()
del raw1, raw5; gc.collect()

dts=ds['ts'].to_numpy(); labels=ds['label'].to_numpy().astype(np.int64)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

# 对齐 ds ts → raw index
idx1=np.searchsorted(r1_ts,dts); idx1=np.clip(idx1,600,len(C1z)-1)
idx5=np.searchsorted(r5_ts,dts); idx5=np.clip(idx5,300,len(C5z)-1)
del r1_ts, r5_ts; gc.collect()

W=100  # 每个尺度看 100 根 bar

def split_mask(lo,hi): return (dts>=lo)&(dts<hi)

tr_m=split_mask(0,TRAIN_END); es_m=split_mask(TRAIN_END,ES_END); te_m=split_mask(META_END,10**18)
tr1=idx1[tr_m]; tr5=idx5[tr_m]; tr_y=labels[tr_m]
es1=idx1[es_m]; es5=idx5[es_m]; es_y=labels[es_m]
te1=idx1[te_m]; te5=idx5[te_m]; te_y=labels[te_m]
del idx1,idx5,dts,labels,tr_m,es_m,te_m; gc.collect()

# 下采样训练到 50 万
np.random.seed(42); sub=np.random.choice(len(tr1),min(500000,len(tr1)),replace=False)
tr1=tr1[sub]; tr5=tr5[sub]; tr_y=tr_y[sub]
print(f"  tr={len(tr1):,} (down) es={len(es1):,} te={len(te1):,}", flush=True)

class MultiScaleDS(Dataset):
    def __init__(self,idx1,idx5,y,w=100):
        self.i1=idx1.astype(np.int64); self.i5=idx5.astype(np.int64)
        self.y=y.astype(np.int64); self.w=w
    def __len__(self): return len(self.i1)
    def __getitem__(self,i):
        e1=self.i1[i]; s1=e1-self.w+1
        e5=self.i5[i]; s5=e5-self.w+1
        x=np.stack([
            C1z[s1:e1+1], S1k[s1:e1+1]/100.0,   # 1min: zscore + sk
            C5z[s5:e5+1], S5k[s5:e5+1]/100.0,   # 5min: zscore + sk
        ], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[i],dtype=torch.long)

# ============= 2. MultiScale CNN =============
class MS_CNN(nn.Module):
    """2 streams: 1min stream + 5min stream, fuse at end"""
    def __init__(self, in_ch_per_stream=2, h=32):
        super().__init__()
        def stream(in_ch, h):
            return nn.Sequential(
                nn.Conv1d(in_ch,h,7,padding=3), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),   # 100→50
                nn.Conv1d(h,h,5,padding=2),    nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),   # 50→25
                nn.Conv1d(h,h*2,5,padding=2),  nn.BatchNorm1d(h*2), nn.SiLU(), nn.MaxPool1d(2),   # 25→12
                nn.Conv1d(h*2,h*2,3,padding=1), nn.BatchNorm1d(h*2), nn.SiLU(),
                nn.AdaptiveAvgPool1d(1))
        self.s1 = stream(in_ch_per_stream,h)  # 1min stream: input (B,2,100)
        self.s5 = stream(in_ch_per_stream,h)  # 5min stream
        self.fc = nn.Sequential(
            nn.Linear(h*4, 64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 2))
    def forward(self,x):
        # x: (B,4,W) — ch0,1=1min; ch2,3=5min
        o1=self.s1(x[:,0:2,:]); o5=self.s5(x[:,2:4,:])
        return self.fc(torch.cat([o1,o5],dim=1).flatten(1))

m=MS_CNN(); tp=sum(p.numel() for p in m.parameters())
print(f"\n[2] MultiScale CNN params={tp:,}", flush=True)

# ============= 3. Train =============
BATCH=2048; EPOCHS=20; LR=1e-3
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader=DataLoader(MultiScaleDS(tr1,tr5,tr_y,W),BATCH,shuffle=True,num_workers=0)
    es_loader=DataLoader(MultiScaleDS(es1,es5,es_y,W),BATCH*2,shuffle=False,num_workers=0)
    te_loader=DataLoader(MultiScaleDS(te1,te5,te_y,W),BATCH*2,shuffle=False,num_workers=0)
    m=MS_CNN(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); best=0; best_state=None; t0=time.time()
    for ep in range(1,EPOCHS+1):
        m.train(); tl=0; n=0
        for xb,yb in tr_loader: opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step(); tl+=loss.item()*len(xb); n+=len(xb)
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb,yb in es_loader: ps.append(F.softmax(m(xb),1)[:,1].numpy()); ys.append(yb.numpy())
        auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
        if auc>best: best=auc; best_state={k:v.clone() for k,v in m.state_dict().items()}
        if ep%5==0: print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.load_state_dict(best_state); m.eval(); ps=[]
    with torch.no_grad():
        for xb,_ in te_loader: ps.append(F.softmax(m(xb),1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[3] Train 3 seeds...", flush=True)
pte=[]
for sd in [42,56,70]:
    print(f"=== seed={sd} ===", flush=True)
    p,bes=run(sd); pte.append(p); print(f"  es_best={bes:.4f}", flush=True)

p=np.mean(pte,axis=0); auc=roc_auc_score(te_y,p)
print(f"\n★★ MultiScale CNN (1min×100 + 5min×100) TEST AUC = {auc:.4f}")
print(f"   LGBM ret+negw baseline: AUC = 0.5428")
print(f"   Single-scale CNN best:  AUC = 0.5411 (之前的 4ch)")
print(f"   {(auc-0.5428)*100:+.2f}pp vs LGBM")

# tpd sweep
print(f"\n[4] tpd sweep...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985,0.998,0.0005):
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100
    tpd=n/332; diff=acc-65; mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}")
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)
if best: print(f"\n★ tpd≈14.4: ACC={best[0]:.1f}% q={best[1]:.4f} tpd={best[2]:.1f}")

print(f"\nTOTAL: {time.time()-T0:.0f}s")
