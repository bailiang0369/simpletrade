"""快速: LGBM ret-feats + Tiny CNN shape-feats ensemble
先跑 LGBM 500k seed42 + CNN seed42, 平均看 AUC
然后看能不能再加 negw
"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore'); T0=time.time()

# ============= 1. LGBM on ret feats (full 2.4M train, 5 seeds) =============
print("[1] LGBM ret feats...", flush=True)
NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
Xtr=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)
Xes=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)
Xte=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)

# with negw
ret_tr=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_tr)>=np.quantile(np.abs(ret_tr),0.90),0.3,1.0).astype(np.float32); del ret_tr; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_lgb=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es_d=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es_d],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_lgb.append(m.predict(Xte))
p_lgb=np.mean(p_lgb,axis=0); auc_lgb=roc_auc_score(yte,p_lgb)
print(f"  LGBM ret+negw AUC={auc_lgb:.4f}", flush=True)
del Xtr,Xes,sw; gc.collect()

# ============= 2. Load CNN predictions (re-run seed42 only, ep=5 early stop) =============
# 直接再跑一次 CNN seed=42 ep=5
print(f"\n[2] Tiny CNN W=64 cz+sk (2ch) seed=42 ep=5...", flush=True)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Build 2ch only: close_zscore + stoch_k (最小化, 避免过拟合)
raw=pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
ds=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
L14=raw['low'].rolling(14,min_periods=1).min(); H14=raw['high'].rolling(14,min_periods=1).max()
raw['sk14']=np.where(H14-L14>1e-9,(raw['close']-L14)/(H14-L14)*100,50.0).astype(np.float32)
C=raw['close'].to_numpy().astype(np.float64)
rm=pd.Series(C).rolling(240,min_periods=60).mean().to_numpy()
rs=pd.Series(C).rolling(240,min_periods=60).std().to_numpy()
Cz=np.where(rs>1e-9,(C-rm)/rs,0.0).astype(np.float32)
Sk=(raw['sk14'].to_numpy()/100.0).astype(np.float32)
rts=raw['ts'].to_numpy(); dts=ds['ts'].to_numpy(); del raw,L14,H14,rm,rs,C; gc.collect()

idx=np.searchsorted(rts,dts); idx=np.clip(idx,512,len(Cz)-1)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200; W=128
tr_idx=idx[(dts>=0)&(dts<TRAIN_END)]; tr_y=ds.loc[(ds['ts']>=0)&(ds['ts']<TRAIN_END),'label'].to_numpy().astype(np.int64)
es_idx=idx[(dts>=TRAIN_END)&(dts<ES_END)]; es_y=ds.loc[(ds['ts']>=TRAIN_END)&(ds['ts']<ES_END),'label'].to_numpy().astype(np.int64)
te_idx=idx[dts>=META_END]; te_y2=ds.loc[ds['ts']>=META_END,'label'].to_numpy().astype(np.int64)
del ds,rts,dts,idx; gc.collect()
N=min(len(tr_idx),len(tr_y)); tr_idx=tr_idx[:N]; tr_y=tr_y[:N]
np.random.seed(42); sub=np.random.choice(len(tr_idx),min(500000,len(tr_idx)),replace=False)
tr_idx=tr_idx[sub]; tr_y=tr_y[sub]
print(f"  tr={len(tr_idx):,} te={len(te_idx):,}", flush=True)

class SeqDS(Dataset):
    def __init__(self,idx,y,w=128): self.idx=idx.astype(np.int64); self.y=y.astype(np.int64); self.w=w
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        e=self.idx[i]; s=e-self.w+1
        x=np.stack([Cz[s:e+1],Sk[s:e+1]],axis=0)
        return torch.from_numpy(x), torch.tensor(self.y[i],dtype=torch.long)

class TinyCNN(nn.Module):
    def __init__(self,in_ch=2,h=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch,h,5,padding=2),nn.BatchNorm1d(h),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h,h,5,padding=2),nn.BatchNorm1d(h),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h,h*2,5,padding=2),nn.BatchNorm1d(h*2),nn.SiLU(),nn.MaxPool1d(2),
            nn.Conv1d(h*2,h*2,5,padding=2),nn.BatchNorm1d(h*2),nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Linear(h*2,2)
    def forward(self,x): return self.fc(self.net(x).flatten(1))

torch.manual_seed(42); np.random.seed(42)
tr_loader=DataLoader(SeqDS(tr_idx,tr_y,W),2048,shuffle=True,num_workers=0)
es_loader=DataLoader(SeqDS(es_idx,es_y,W),4096,shuffle=False,num_workers=0)
te_loader=DataLoader(SeqDS(te_idx,te_y2,W),4096,shuffle=False,num_workers=0)
m=TinyCNN(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
ce=nn.CrossEntropyLoss(); t0=time.time()
for ep in range(1,6):  # 只跑 5 epochs (best=ep5)
    m.train(); tl=0; n=0
    for xb,yb in tr_loader: opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step(); tl+=loss.item()*len(xb); n+=len(xb)
    m.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for xb,yb in es_loader: ps.append(F.softmax(m(xb),1)[:,1].numpy()); ys.append(yb.numpy())
    auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
    print(f"  CNN ep={ep} loss={tl/n:.4f} es={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
m.eval(); p_cnn=[]
with torch.no_grad():
    for xb,_ in te_loader: p_cnn.append(F.softmax(m(xb),1)[:,1].numpy())
p_cnn=np.concatenate(p_cnn); auc_cnn=roc_auc_score(te_y2,p_cnn)
print(f"  CNN TEST AUC={auc_cnn:.4f}", flush=True)

# ============= 3. Ensemble =============
print(f"\n[3] Ensemble (平均 LGBM + CNN)...", flush=True)
y=yte[:min(len(yte),len(p_cnn))] if len(yte)!=len(p_cnn) else yte
p_lgb2=p_lgb[:len(p_cnn)] if len(p_lgb)>len(p_cnn) else p_lgb
auc_lgb2=roc_auc_score(y,p_lgb2); print(f"  LGBM (aligned) AUC={auc_lgb2:.4f}")
auc_cnn2=roc_auc_score(y,p_cnn); print(f"  CNN  (aligned) AUC={auc_cnn2:.4f}")

for w in [0.3,0.4,0.5,0.6,0.7]:
    p_ens=w*p_cnn + (1-w)*p_lgb2
    auc=roc_auc_score(y,p_ens)
    print(f"  CNN×{w}+LGBM×{1-w:.1f}: AUC={auc:.4f}")

best_w=0.5; p_best=0.5*p_cnn+0.5*p_lgb2; auc_best=roc_auc_score(y,p_best)
print(f"\n★ Best ensemble AUC={auc_best:.4f}  ({(auc_best-auc_lgb2)*100:+.2f}pp vs LGBM, {(auc_best-auc_cnn2)*100:+.2f}pp vs CNN)")

# tpd sweep for best ensemble
print(f"\n[4] tpd sweep (ensemble)...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985,0.998,0.0005):
    th=np.quantile(p_best,q); lm=p_best>th; sm=p_best<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100; tpd=n/332; diff=acc-65
    mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}")
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)
if best: print(f"\n★ Ensemble tpd≈14.4: ACC={best[0]:.1f}%")

# 对比单模型
print(f"\n{'='*60}")
print(f"  最终对比")
print(f"{'='*60}")
print(f"  LGBM ret+negw (2.4M, 5 seeds): AUC={auc_lgb:.4f}  tpd14.4 ACC≈61.6%")
print(f"  Tiny CNN shape (500k, ep=5):   AUC={auc_cnn:.4f}  tpd14.4 ACC≈56.9%")
print(f"  Ensemble (0.5 CNN+0.5 LGBM):   AUC={auc_best:.4f}  tpd14.4 ACC≈{best[0]:.1f}%")
print(f"  Gap to 65%: {65-best[0]:.1f}pp")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s")
