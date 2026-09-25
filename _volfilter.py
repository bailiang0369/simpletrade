"""关键: 模型信号 + 市场波动双重过滤
手动交易也是波动大时才下注 — 只在 ret_std 高的时段交易
"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200
SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Load ETH h15 pred (先快速 train)
def eth_comb(split):
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    mr=min(Xb.shape[0],Xf.shape[0])
    return np.hstack([Xb[:mr],Xf[:mr]]), np.load(f'{NPY}/ETH_h15_{split}_y.npy').astype(np.int32)[:mr]

Xtr,ytr=eth_comb('train'); Xes,yes=eth_comb('early_stop'); Xte,yte=eth_comb('test')
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr.shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32)
del ret_e; gc.collect()
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p15=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,pte,sw; gc.collect()

t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32)
ret=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts>=META_END; ts=ts[mm][:len(p15)]; y=y[mm][:len(p15)]; ret=ret[mm][:len(p15)]
days=(ts[-1]-ts[0])/SPD
log(f"[1] h15 AUC={roc_auc_score(y,p15):.4f}, {len(p15):,} rows, {days:.0f} days")

# ========== 1. Volatility filter ==========
log("\n[2] Build rolling volatility...")
raw=pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
close=raw['close'].to_numpy().astype(np.float64); raws_ts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()
rets=np.diff(close)/close[:-1]; rets=np.concatenate([[0],rets])
# rolling |ret| mean as volatility
vol20=pd.Series(np.abs(rets)).rolling(20,min_periods=5).mean().to_numpy()
vol60=pd.Series(np.abs(rets)).rolling(60,min_periods=10).mean().to_numpy()
del close,rets; gc.collect()

idx=np.searchsorted(raws_ts,ts)
v20=vol20[idx]; v60=vol60[idx]
del raws_ts,vol20,vol60; gc.collect()
log(f"  vol20 quantiles: 50%={np.quantile(v20,0.5):.6f} 80%={np.quantile(v20,0.8):.6f} 90%={np.quantile(v20,0.9):.6f}")

# ========== 2. Train h5 + h30 (OOM 不会再了, 分别训) ==========
log("\n[3] Train ETH h5 & h30...")
def train_h(h):
    df=pd.read_parquet(f'data/datasets/ds_ETH_h{h}.parquet').sort_values('ts')
    fc=[c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
    TE=1722556800; ES=1725148800
    Xtr=df[(df['ts']>=0)&(df['ts']<TE)][fc].to_numpy().astype(np.float32)
    ytr=df[(df['ts']>=0)&(df['ts']<TE)]['label'].to_numpy().astype(np.int32)
    Xes=df[(df['ts']>=TE)&(df['ts']<ES)][fc].to_numpy().astype(np.float32)
    yes=df[(df['ts']>=TE)&(df['ts']<ES)]['label'].to_numpy().astype(np.int32)
    Xte=df[df['ts']>=META_END][fc].to_numpy().astype(np.float32)
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
            'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte=[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte))
    return np.mean(pte,axis=0), df[df['ts']>=META_END]['ts'].to_numpy().astype(np.int64)

p_h5, ts_h5 = train_h(5); gc.collect()
p_h30, ts_h30 = train_h(30); gc.collect()

# Align
idx5=np.searchsorted(ts_h5, ts); idx5=np.clip(idx5,0,len(p_h5)-1)
idx30=np.searchsorted(ts_h30, ts); idx30=np.clip(idx30,0,len(p_h30)-1)
p_mens=np.mean([p15, p_h5[idx5], p_h30[idx30]], axis=0)
log(f"  h5 AUC={roc_auc_score(np.load(f'{NPY}/ETH_h5_test_y.npy')[:len(p_h5)], p_h5):.4f}")
log(f"  h30 AUC={roc_auc_score(np.load(f'{NPY}/ETH_h30_test_y.npy')[:len(p_h30)], p_h30):.4f}")
log(f"  Ens AUC={roc_auc_score(y,p_mens):.4f}")

# ========== 3. Comprehensive grid search: q × vol_filter ==========
log(f"\n{'='*60}")
log(f" GRID:  q × vol_filter  →  tpd≈14.4  看 ACC 上限")
log(f"{'='*60}")

q_range=np.arange(0.989, 0.995, 0.0005)
vol_q_range=[None, 0.7, 0.75, 0.8, 0.85, 0.9]  # vol 最低阈值 (None=不滤)
models=[('h15', p15), ('ens_h5h15h30', p_mens)]

results=[]
for mname, preds in models:
    for vq in vol_q_range:
        vmask = np.ones(len(preds),dtype=bool)
        if vq is not None:
            th=np.quantile(v20, vq)
            vmask = v20 >= th
        for q in q_range:
            th=np.quantile(preds,q); lm=preds>th; sm=preds<(1-th)
            sig=(lm|sm)&vmask
            n=sig.sum(); tpd=n/days
            if n<30: continue
            ss=sm[sig]; acc=(((~ss)&(y[sig]==1))|(ss&(y[sig]==0))).mean()*100
            results.append((mname,vq,q,tpd,acc,n))

# 只看 tpd 8-20
log(f"\n  model            vol_q    q     tpd    ACC     n")
log(f"  {'-'*55}")
best=None
for mname,vq,q,tpd,acc,n in sorted(results, key=lambda x: abs(x[3]-14.4)):
    if 10<=tpd<=20:
        vqs = "NONE" if vq is None else f"{vq:.2f}"
        marker = " ◀ BEST" if (best is None or acc>best[0]) else ""
        best = (acc, mname, vq, q, tpd, n) if acc>(best[0] if best else 0) else best
        log(f"  {mname:15s} {vqs:>5s} {q:.4f} {tpd:5.1f} {acc:5.1f}% {n:5d}{marker}")

if best:
    log(f"\n  ★ BEST: {best[1]} vol_q={'NONE' if best[2] is None else f'{best[2]:.2f}'} q={best[3]:.4f} tpd={best[4]:.1f} ACC={best[0]:.1f}%")

log(f"\nDONE {time.time()-T0:.0f}s")
