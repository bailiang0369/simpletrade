"""ETH h15: tpd=14.4 (1%) 时 ACC 到底多少 + regime filter + multi-horizon"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200; TRAIN_END=1722556800
SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== 1. Train ETH h15 base+fund 5-seed ==========
log("[1] Train ETH h15 base+fund 68 +negw...")
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

# Align
t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts_test=t.column('ts').to_numpy().astype(np.int64); y_test=t.column('label').to_numpy().astype(np.int32)
ret_test=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts_test>=META_END; ts_test=ts_test[mm][:len(p15)]; y_test=y_test[mm][:len(p15)]; ret_test=ret_test[mm][:len(p15)]
days=(ts_test[-1]-ts_test[0])/SPD
log(f"  Test: {len(p15):,} rows / {days:.0f} days = {len(p15)/days:.0f} rows/day")
log(f"  h15 AUC={roc_auc_score(y_test,p15):.4f}")

# ========== 2. Global q sweep — 找 tpd=14.4 附近 ==========
log(f"\n{'='*60}")
log(f" ETH h15  tpd 1-30 区间 ACC  (global q, upper bound)")
log(f"{'='*60}")
prev_acc=0
for q in np.arange(0.985, 0.999, 0.0005):
    th=np.quantile(p15,q)
    lm=p15>th; sm=p15<(1-th); tm=lm|sm
    n=tm.sum(); tpd=n/days; cov=n/len(p15)*100
    if tpd<0.5 or tpd>40: continue
    ss=sm[tm]; acc=(((~ss)&(y_test[tm]==1))|(ss&(y_test[tm]==0))).mean()*100
    marker = " ◀" if abs(tpd-14.4)<2 else ""
    log(f"  q={q:.4f}: n={n:5d} cov={cov:.2f}% tpd={tpd:5.1f} ACC={acc:5.1f}%{marker}")

# ========== 3. Regime features: ADX + vol ==========
log(f"\n{'='*60}")
log(f" REGIME FILTER (趋势市才交易) + q 扫")
log(f"{'='*60}")
raw=pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
close=raw['close'].to_numpy().astype(np.float64); high=raw['high'].to_numpy().astype(np.float64)
low=raw['low'].to_numpy().astype(np.float64); raws_ts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()

# Simplified ADX
def adx(c,h,l,w):
    up=np.zeros(len(c)); up[1:]=h[1:]-h[:-1]
    dn=np.zeros(len(c)); dn[1:]=l[:-1]-l[1:]
    dmp=np.where((up>dn)&(up>0),up,0); dmn=np.where((dn>up)&(dn>0),dn,0)
    tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    di_p=pd.Series(dmp).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    di_m=pd.Series(dmn).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    dx=np.where((di_p+di_m)>1e-9, np.abs(di_p-di_m)/(di_p+di_m)*100, 0)
    return pd.Series(dx).rolling(w,min_periods=10).mean().to_numpy()

# Volatility ratio
ret=np.diff(close)/close[:-1]; ret=np.concatenate([[0],ret])
vol=pd.Series(ret).rolling(60,min_periods=10).std().to_numpy()
vol_ma=pd.Series(vol).rolling(240,min_periods=30).mean().to_numpy()
vr=np.where(vol_ma>1e-12, vol/vol_ma, 1.0)

adx60=adx(close,high,low,60); del close,high,low; gc.collect()

# Align regime to test
idx=np.searchsorted(raws_ts, ts_test)
a_t=adx60[idx]; vr_t=vr[idx]
del raws_ts,adx60,vr; gc.collect()

log(f"  Regime dist in test: ADX60>20={(a_t>20).mean()*100:.1f}%  ADX60>25={(a_t>25).mean()*100:.1f}%")

for rname, rmask in [
    ('NO FILTER (baseline)', np.ones(len(p15),dtype=bool)),
    ('ADX60>20 trend', a_t>20),
    ('ADX60>25 strong trend', a_t>25),
    ('ADX60>30 very strong', a_t>30),
]:
    log(f"\n  {rname}:")
    for q in np.arange(0.985, 0.999, 0.0005):
        th=np.quantile(p15,q); lm=p15>th; sm=p15<(1-th); sig=(lm|sm)&rmask
        if sig.sum()<30: continue
        tpd=sig.sum()/days; cov=sig.sum()/len(p15)*100
        ss=sm[sig]; acc=(((~ss)&(y_test[sig]==1))|(ss&(y_test[sig]==0))).mean()*100
        if 8<=tpd<=25:
            marker=" ◀ TARGET" if abs(tpd-14.4)<2 else ""
            log(f"    q={q:.4f}: tpd={tpd:5.1f} ACC={acc:5.1f}% n={sig.sum():5d}{marker}")

# ========== 4. Multi-horizon ETH ensemble (h5+h15+h30) ==========
log(f"\n{'='*60}")
log(f" MULTI-HORIZON ETH ENSEMBLE  (h5+h15+h30)")
log(f"{'='*60}")

def train_h(h):
    df=pd.read_parquet(f'data/datasets/ds_ETH_h{h}.parquet').sort_values('ts')
    fc=[c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
    Xtr=df[(df['ts']>=0)&(df['ts']<TRAIN_END)][fc].to_numpy().astype(np.float32)
    ytr=df[(df['ts']>=0)&(df['ts']<TRAIN_END)]['label'].to_numpy().astype(np.int32)
    Xes=df[(df['ts']>=TRAIN_END)&(df['ts']<1725148800)][fc].to_numpy().astype(np.float32)
    yes=df[(df['ts']>=TRAIN_END)&(df['ts']<1725148800)]['label'].to_numpy().astype(np.int32)
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
log(f"  h5 AUC={roc_auc_score(np.load(f'{NPY}/ETH_h5_test_y.npy')[:len(p_h5)], p_h5):.4f}")
log(f"  h30 AUC={roc_auc_score(np.load(f'{NPY}/ETH_h30_test_y.npy')[:len(p_h30)], p_h30):.4f}")

# Align to h15 timestamps
idx5=np.searchsorted(ts_h5, ts_test); idx5=np.clip(idx5,0,len(p_h5)-1)
idx30=np.searchsorted(ts_h30, ts_test); idx30=np.clip(idx30,0,len(p_h30)-1)
p_mens=np.mean([p15, p_h5[idx5], p_h30[idx30]], axis=0)
log(f"  Ensemble AUC={roc_auc_score(y_test,p_mens):.4f}")

log(f"\n  Ensemble q sweep (tpd 8-25):")
for q in np.arange(0.985, 0.999, 0.0005):
    th=np.quantile(p_mens,q); lm=p_mens>th; sm=p_mens<(1-th); tm=lm|sm
    if tm.sum()<30: continue
    tpd=tm.sum()/days
    ss=sm[tm]; acc=(((~ss)&(y_test[tm]==1))|(ss&(y_test[tm]==0))).mean()*100
    if 8<=tpd<=25:
        marker=" ◀ TARGET" if abs(tpd-14.4)<2 else ""
        log(f"    q={q:.4f}: tpd={tpd:5.1f} ACC={acc:5.1f}% n={tm.sum():5d}{marker}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
