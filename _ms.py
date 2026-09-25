"""加多周期 + 量价背离特征 → ETH h15 AUC 能不能破 0.545"""
import numpy as np, pandas as pd, time, gc, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); import sys; sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); SEEDS=[42,49,56,63,70]; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
log = lambda *a: print(' '.join(str(x) for x in a), flush=True)

# Load base ds
df=pd.read_parquet(f'data/datasets/ds_ETH_h15.parquet').sort_values('ts')
fc=[c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
X_base=df[fc].to_numpy().astype(np.float32); y=df['label'].to_numpy().astype(np.int32)
ret=df['ret_future'].to_numpy().astype(np.float32); ts=df['ts'].to_numpy().astype(np.int64); del df; gc.collect()

# Build new feats from raw
raw=pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
close=raw['close'].to_numpy().astype(np.float64); high=raw['high'].to_numpy().astype(np.float64)
low=raw['low'].to_numpy().astype(np.float64); bv=raw['buy_vol'].to_numpy().astype(np.float64)
sv=raw['sell_vol'].to_numpy().astype(np.float64); raw_ts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()

feats={}
# Multi-scale MA
for w,nm in [(96,'4h'),(288,'12h'),(960,'1d')]:
    ma=pd.Series(close).rolling(w,min_periods=w//3).mean().to_numpy()
    feats[f'c/ma_{nm}']=np.where(ma>1e-9,close/ma-1,0).astype(np.float32)
# Price position in window
for w in [96,288,960]:
    pmin=pd.Series(close).rolling(w,min_periods=w//3).min().to_numpy()
    pmax=pd.Series(close).rolling(w,min_periods=w//3).max().to_numpy()
    feats[f'pos_{w}']=np.where(pmax-pmin>1e-9,(close-pmin)/(pmax-pmin),0.5).astype(np.float32)
# Vol ratio
rr=np.diff(close)/close[:-1]; rr=np.concatenate([[0],rr])
feats['vol_ratio']=(pd.Series(rr).rolling(60,min_periods=10).std().to_numpy() / 
                   np.maximum(pd.Series(rr).rolling(960,min_periods=60).std().to_numpy(),1e-9)).astype(np.float32)
# ADX
def adx(c,h,l,w):
    up=np.zeros(len(c)); up[1:]=h[1:]-h[:-1]; dn=np.zeros(len(c)); dn[1:]=l[:-1]-l[1:]
    dmp=np.where((up>dn)&(up>0),up,0); dmn=np.where((dn>up)&(dn>0),dn,0)
    tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    dip=pd.Series(dmp).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    dim=pd.Series(dmn).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    dx=np.where((dip+dim)>1e-9,np.abs(dip-dim)/(dip+dim)*100,0)
    return pd.Series(dx).rolling(w,min_periods=10).mean().to_numpy()
feats['adx_60']=adx(close,high,low,60).astype(np.float32); feats['adx_240']=adx(close,high,low,240).astype(np.float32)
# Bar streaks
br=np.diff(close)/close[:-1]; br=np.concatenate([[0],br])
def consec(arr,sign):
    out=np.zeros(len(arr)); c=0
    for i in range(len(arr)):
        c=c+1 if sign*arr[i]>0 else 0; out[i]=c
    return out
feats['up_streak']=consec(br,1).astype(np.float32); feats['dn_streak']=consec(br,-1).astype(np.float32)
# Bear/bull divergence
for pw in [60,240,960]:
    ph=pd.Series(close).rolling(pw,min_periods=pw//3).max().to_numpy()
    pl=pd.Series(close).rolling(pw,min_periods=pw//3).min().to_numpy()
    vm=pd.Series(bv+sv).rolling(pw,min_periods=pw//3).median().to_numpy()
    at_h=(close>=ph*0.999); at_l=(close<=pl*1.001); cur_v=bv+sv
    feats[f'bd_{pw}']=((at_h)&(cur_v<vm)).astype(np.float32)
    feats[f'bld_{pw}']=((at_l)&(cur_v<vm)).astype(np.float32)
# Buy vol ratio
bvr=np.where(bv+sv>0,bv/(bv+sv),0.5)
feats['bvr_ma60']=pd.Series(bvr).rolling(60,min_periods=10).mean().to_numpy().astype(np.float32)
feats['bvr_std60']=pd.Series(bvr).rolling(60,min_periods=10).std().to_numpy().astype(np.float32)

keys=sorted(feats.keys()); log(f"New feats: {len(keys)}")
new_arr=np.stack([feats[k] for k in keys],axis=1).astype(np.float32); del feats; gc.collect()
idx=np.searchsorted(raw_ts,ts); idx=np.clip(idx,0,len(new_arr)-1)
X_new=new_arr[idx]; del new_arr,raw_ts; gc.collect()

X_all=np.hstack([X_base,X_new]); nan=~np.isnan(X_all).any(axis=1)
log(f"Combined: {X_all.shape}, valid={nan.sum():,}"); X_all=X_all[nan]; y=y[nan]; ret=ret[nan]; ts=ts[nan]
del X_base,X_new; gc.collect()

# Split
tr=(ts<TRAIN_END); es=(ts>=TRAIN_END)&(ts<ES_END); te=ts>=META_END
Xtr,ytr=X_all[tr],y[tr]; Xes,yes=X_all[es],y[es]; Xte,yte=X_all[te],y[te]
log(f"tr={Xtr.shape[0]:,} te={Xte.shape[0]:,} feats={X_all.shape[1]}")
ret_tr=ret[tr][:Xtr.shape[0]]
sw=np.where(np.abs(ret_tr)>=np.quantile(np.abs(ret_tr),0.90),0.3,1.0).astype(np.float32); del ret_tr,ret,ts,nan; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]
for sd in SEEDS:
    params['seed']=sd
    tr_d=lgb.Dataset(Xtr,label=ytr,weight=sw); es_d=lgb.Dataset(Xes,label=yes,reference=tr_d)
    m=lgb.train(params,tr_d,5000,[es_d],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,pte,sw; gc.collect()
auc=roc_auc_score(yte,p); log(f"\n★ Test AUC={auc:.4f}  (base=0.5428, +fund=0.5440, target=0.545+)")

# Feature importance (top new)
tr_d=lgb.Dataset(np.vstack([lgb.Dataset(Xte).data, Xte]), label=np.concatenate([yte,yte]), free_raw_data=False)
# Just train one model and check gain
params['seed']=42; params['verbose']=-1
m1=lgb.train(params,lgb.Dataset(Xtr,label=ytr,weight=sw),200)
gain=m1.feature_importance(importance_type='gain')
all_cols=fc+keys; top_idx=np.argsort(gain)[-15:][::-1]
log(f"Top 15 features by gain:")
for i in top_idx:
    log(f"  {all_cols[i]:30s} gain={gain[i]:.0f}  NEW={'*' if i>=len(fc) else ''}")

log(f"\nDONE {time.time()-T0:.0f}s")
