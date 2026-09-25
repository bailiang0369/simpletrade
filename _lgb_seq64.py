"""LightGBM on 64-close + 64-stoch_k14 (1min raw, window=64=1h4min)
纯形态结构, 零 ret, 验证: 形态 > ret?
"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]; W=64
print(f"Window={W}, feats={W*2}")

# Build from raw
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
ds  = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future']).sort_values('ts')
L14=raw['low'].rolling(14,min_periods=1).min(); H14=raw['high'].rolling(14,min_periods=1).max()
raw['sk14']=np.where(H14-L14>1e-9,(raw['close']-L14)/(H14-L14)*100,50.0).astype(np.float32)
raw['close']=raw['close'].astype(np.float32)
rts=raw['ts'].to_numpy(); dts=ds['ts'].to_numpy(); C=raw['close'].to_numpy(); K=raw['sk14'].to_numpy()
del raw, L14, H14; gc.collect()

idx=np.searchsorted(rts,dts); idx=np.clip(idx,W*4,len(C)-1)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

def split(lo,hi):
    m=(dts>=lo)&(dts<hi); return idx[m], ds.loc[m,'label'].to_numpy().astype(np.int32)

def build(idx_arr):
    N=len(idx_arr); out=np.empty((N,W*2),dtype=np.float32)
    for i in range(N):
        e=idx_arr[i]; s=e-W+1
        c=C[s:e+1]; k=K[s:e+1]; c0=c[0]
        out[i,:W]=np.where(c0>1e-9,c/c0-1.0,0.0)
        out[i,W:]=k/100.0
    return out

print("[1] Build seq64 arrays...", flush=True)
tr_idx,tr_y=split(0,TRAIN_END); print(f"  tr={len(tr_idx):,}")
es_idx,es_y=split(TRAIN_END,ES_END); print(f"  es={len(es_idx):,}")
te_idx,te_y=split(META_END,10**18);  print(f"  te={len(te_idx):,}")

Xtr=build(tr_idx); Xes=build(es_idx); Xte=build(te_idx)
del tr_idx,es_idx,te_idx; gc.collect()
print(f"  Done. tr={Xtr.shape} te={Xte.shape}", flush=True)

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
print(f"\n[2] Train SEQ64 (LightGBM 5-seed)...", flush=True)
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=tr_y); es=lgb.Dataset(Xes,label=es_y,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p64=np.mean(pte,axis=0); del Xtr,Xes,pte; gc.collect()
auc64=roc_auc_score(te_y,p64); print(f"  ★ SEQ64 AUC={auc64:.4f} ({time.time()-t0:.0f}s)")

# Hybrid: SEQ64 + BASE ret feats
print(f"\n[3] SEQ64 + BASE ret feats (hybrid)...", flush=True)
for sn in ['train','early_stop','test']:
    Xb=np.load(f'{NPY}/ETH_h15_{sn}_X.npy').astype(np.float32)
    Xs=np.load(f'{NPY}/_seq64_{sn}_X.npy') if False else None  # not saved yet
# 手动构建 hybrid
print("  Skip: let's just check SEQ64 alone vs ret baseline")

# tpd sweep
print(f"\n[4] tpd sweep for SEQ64 (global q upper bound)...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985,0.998,0.0005):
    th=np.quantile(p64,q); lm=p64>th; sm=p64<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100
    tpd=n/332; diff=acc-65
    mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}")
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)

print(f"\n★ SEQ64 tpd≈14.4: ACC={best[0]:.1f}%  (ret baseline ≈61.6% oracle)")
print(f"\n{'='*60}")
print(f"  对比:")
print(f"  ret-based LightGBM (56 feats):     AUC=0.5428  tpd14.4 ACC≈61.6%")
print(f"  纯形态 SEQ64 (128 feats):          AUC={auc64:.4f}  tpd14.4 ACC≈{best[0]:.1f}%")
print(f"  {(auc64-0.5428)*100:+.2f}pp AUC  ({best[0]-61.6:+.1f}pp tail ACC)")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s")
