"""极简: 只加 3 个最 promising 多周期特征, split 内 hstack 不 OOM"""
import numpy as np, pandas as pd, time, gc, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); import sys; sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
log=lambda *a: print(' '.join(str(x) for x in a), flush=True)

# Build 3 key multi-scale feats from raw
raw=pl.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort('ts')
c=raw['close'].to_numpy().astype(np.float64); h=raw['high'].to_numpy().astype(np.float64)
l=raw['low'].to_numpy().astype(np.float64); rts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()

# 4H MA ratio
ma4=pd.Series(c).rolling(96,min_periods=32).mean().to_numpy()
f1=np.where(ma4>1e-9, c/ma4-1, 0).astype(np.float32)

# 4H price position
pmin=pd.Series(c).rolling(96,min_periods=32).min().to_numpy()
pmax=pd.Series(c).rolling(96,min_periods=32).max().to_numpy()
f2=np.where(pmax-pmin>1e-9, (c-pmin)/(pmax-pmin), 0.5).astype(np.float32)

# ADX 240 (long-term trend)
up=np.zeros(len(c)); up[1:]=h[1:]-h[:-1]; dn=np.zeros(len(c)); dn[1:]=l[:-1]-l[1:]
dmp=np.where((up>dn)&(up>0),up,0); dmn=np.where((dn>up)&(dn>0),dn,0)
tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
dip=pd.Series(dmp).rolling(240,min_periods=20).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(240,min_periods=20).mean().to_numpy(),1e-9)
dim=pd.Series(dmn).rolling(240,min_periods=20).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(240,min_periods=20).mean().to_numpy(),1e-9)
dx=np.where((dip+dim)>1e-9, np.abs(dip-dim)/(dip+dim)*100, 0)
f3=pd.Series(dx).rolling(240,min_periods=20).mean().to_numpy().astype(np.float32)
del c,h,l,up,dn,dmp,dmn,tr,dip,dim,dx,ma4,pmin,pmax; gc.collect()

def get_feats_split(split, tlo, thi):
    idx=np.searchsorted(rts, np.load(f'{NPY}/ETH_h15_{split}_y.npy' if split=='train' else 'data/datasets/ds_ETH_h15.parquet', mmap_mode='r') if False else np.load(f'{NPY}/ETH_h15_{split}_X.npy', mmap_mode='r').shape[0]*0 + np.array([tlo]))
    # Simplified: just index using all raw ts matching split window
    m=(rts>=tlo)&(rts<thi)
    return np.stack([f1[m],f2[m],f3[m]],axis=1).astype(np.float32)

# Direct approach: load base splits, hstack, save
log("[1] Hstack + save splits...")
for split,tlo,thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),('test',META_END,10**18)]:
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    m=(rts>=tlo)&(rts<thi)
    Xm=np.stack([f1[m][:Xb.shape[0]],f2[m][:Xb.shape[0]],f3[m][:Xb.shape[0]]],axis=1).astype(np.float32)
    mr=min(Xb.shape[0],Xm.shape[0])
    Xout=np.hstack([Xb[:mr], Xm[:mr]])
    np.save(f'{NPY}/_ethms_{split}_X.npy', Xout); del Xb,Xm; gc.collect()
    log(f"  {split}: {Xout.shape}")
del f1,f2,f3,rts; gc.collect()

# Train
log("\n[2] Train base+ms (56+3=59 feats)...")
Xtr=np.load(f'{NPY}/_ethms_train_X.npy').astype(np.float32)
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xes=np.load(f'{NPY}/_ethms_early_stop_X.npy').astype(np.float32)
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xte=np.load(f'{NPY}/_ethms_test_X.npy').astype(np.float32)
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]
gc.collect()
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr.shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32); del ret_e; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,pte,sw; gc.collect()
auc=roc_auc_score(np.load(f'{NPY}/ETH_h15_test_y.npy')[:len(p)], p)
log(f"  ★ Test AUC={auc:.4f}  (base=0.5428, +fund=0.5440, +fund+bs=0.5442) ({time.time()-t0:.0f}s)")

# Now +funding too!
log("\n[3] Also add funding → base+ms+fund (59+12=71)...")
for split,tlo,thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),('test',META_END,10**18)]:
    X=np.load(f'{NPY}/_ethms_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    mr=min(X.shape[0],Xf.shape[0]); del X; gc.collect()
    Xb=np.load(f'{NPY}/_ethms_{split}_X.npy').astype(np.float32)[:mr]
    Xfb=Xf[:mr]; del Xf; gc.collect()
    np.save(f'{NPY}/_ethmsf_{split}_X.npy', np.hstack([Xb, Xfb])); del Xb,Xfb; gc.collect()

Xtr=np.load(f'{NPY}/_ethmsf_train_X.npy').astype(np.float32)
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xes=np.load(f'{NPY}/_ethmsf_early_stop_X.npy').astype(np.float32)
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xte=np.load(f'{NPY}/_ethmsf_test_X.npy').astype(np.float32)
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]
gc.collect()
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr.shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32); del ret_e; gc.collect()

pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p2=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,pte,sw; gc.collect()
auc2=roc_auc_score(np.load(f'{NPY}/ETH_h15_test_y.npy')[:len(p2)], p2)
log(f"  ★ base+ms+fund AUC={auc2:.4f}  (vs +fund=0.5440) ({time.time()-t0:.0f}s)")

# tpd sweep
y=np.load(f'{NPY}/ETH_h15_test_y.npy')[:len(p2)]
log(f"\n  tpd=14.4 vicinity:")
for q in np.arange(0.989,0.9955,0.0005):
    th=np.quantile(p2,q); lm=p2>th; sm=p2<(1-th); tm=lm|sm; n=tm.sum()
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    tpd=n/332
    if 12<=tpd<=18: log(f"    q={q:.4f} tpd={tpd:5.1f} ACC={acc:5.1f}%")

log(f"\nFINAL: +{auc2-0.5428:+.4f} vs base, +{auc2-0.5440:+.4f} vs +fund")
log(f"TOTAL: {time.time()-T0:.0f}s")
