"""快速: ETH q 扫到 0.9995, 看 tpd=1 时 ACC"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200
SEEDS=[42,49,56,63,70]

# Load + train ETH h15 base+fund
Xb=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_train_X.npy').astype(np.float32)
mr=min(Xb.shape[0],Xf.shape[0]); Xtr=np.hstack([Xb[:mr],Xf[:mr]])
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xb=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_early_stop_X.npy').astype(np.float32)
Xes=np.hstack([Xb[:min(Xb.shape[0],Xf.shape[0])],Xf[:min(Xb.shape[0],Xf.shape[0])]])
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xb=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_test_X.npy').astype(np.float32)
Xte=np.hstack([Xb[:min(Xb.shape[0],Xf.shape[0])],Xf[:min(Xb.shape[0],Xf.shape[0])]])
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]
del Xb,Xf; gc.collect()

ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr.shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32); del ret_e; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p=np.mean(pte,axis=0); 
print(f"[1] ETH te={roc_auc_score(yte,p):.4f}", flush=True)

# Align timestamps
t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32)
ret=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts>=META_END; ts=ts[mm][:len(p)]; y=y[mm][:len(p)]; ret=ret[mm][:len(p)]

days = (ts[-1]-ts[0])/SPD
print(f"[2] Test: {len(p):,} rows, {days:.0f} days, {len(p)/days:.0f} rows/day", flush=True)
print(f"    p range: [{p.min():.4f}, {p.max():.4f}], p-0.5 max: {np.max(np.abs(p-0.5)):.4f}", flush=True)

# ========== Global q sweep (upper bound) ==========
print(f"\n{'='*55}")
print(f" ETH h15  COVERAGE 1%  →  q=?  ACC=?  tpd=?")
print(f"{'='*55}")
print(f"\n[3] Global q sweep (upper bound, no lookahead concern)", flush=True)
for q in [0.90, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999, 0.9993, 0.9995, 0.9997, 0.9999]:
    th=np.quantile(p,q)
    lm=p>th; sm=p<(1-th); tm=lm|sm
    n=tm.sum(); cov=n/len(p)*100; tpd=n/days
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    print(f"  q={q:.4f}: th={th:.4f} n={n:6,d} cov={cov:.3f}% tpd={tpd:6.1f} ACC={acc:5.1f}%")

# ========== Rolling q sweep ==========
print(f"\n[4] Rolling 30d no-lookahead q sweep (realistic)", flush=True)
for q in [0.99, 0.995, 0.999, 0.9995]:
    th=np.full(len(p),np.nan)
    for i in range(2000,len(p)):
        tp=ts[i]-30*SPD; m2=(ts[:i]>=tp)
        if m2.sum()>200: th[i]=np.quantile(p[:i][m2],q)
    gq=np.nanquantile(p,q); fv=np.where(~np.isnan(th))[0]
    th[:fv[0]]=gq; th=np.where(np.isnan(th),gq,th)
    lm=p>th; sm=p<(1-th); tm=lm|sm
    n=tm.sum(); cov=n/len(p)*100; tpd=n/days
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    print(f"  q={q:.4f}: n={n:6,d} cov={cov:.3f}% tpd={tpd:6.1f} ACC={acc:5.1f}%")

print(f"\nDONE {time.time()-T0:.0f}s")
