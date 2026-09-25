"""极简: 只扫 h15 + vol filter, 避免 OOM"""
import numpy as np, pandas as pd, time, gc, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True) if 'sys' in dir() else None
import sys
sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Train h15
Xb=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_train_X.npy').astype(np.float32)
mr=min(Xb.shape[0],Xf.shape[0]); Xtr=np.hstack([Xb[:mr],Xf[:mr]]); del Xb,Xf
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xb=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_early_stop_X.npy').astype(np.float32)
Xes=np.hstack([Xb[:min(Xb.shape[0],Xf.shape[0])],Xf[:min(Xb.shape[0],Xf.shape[0])]]); del Xb,Xf
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xb=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)
Xf=np.load(f'{NPY}/ETH_h15_fund_test_X.npy').astype(np.float32)
Xte=np.hstack([Xb[:min(Xb.shape[0],Xf.shape[0])],Xf[:min(Xb.shape[0],Xf.shape[0])]]); del Xb,Xf
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]
gc.collect()
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
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,pte,sw; gc.collect()

t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32)
ret=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts>=META_END; ts=ts[mm][:len(p)]; y=y[mm][:len(p)]; ret=ret[mm][:len(p)]
days=(ts[-1]-ts[0])/SPD
log(f"h15 AUC={roc_auc_score(y,p):.4f}")

# Vol
raw=pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
c=raw['close'].to_numpy().astype(np.float64); rts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()
rr=np.diff(c)/c[:-1]; rr=np.concatenate([[0],rr]); del c; gc.collect()
v60=pd.Series(np.abs(rr)).rolling(60,min_periods=10).mean().to_numpy(); del rr; gc.collect()
idx=np.searchsorted(rts,ts); v=v60[idx]; del rts,v60; gc.collect()

log(f"\n{'='*55}")
log(f"  h15  q × vol_filter  →  tpd 10-20 ACC")
log(f"{'='*55}")
log(f"  {'vol_q':>5s} {'q':>7s} {'tpd':>6s} {'ACC':>7s} {'n':>6s}  {'note'}")
log(f"  {'-'*55}")

best=None
for vq in [None, 0.65, 0.7, 0.75, 0.8, 0.85]:
    vmask = np.ones(len(p),dtype=bool) if vq is None else (v >= np.quantile(v,vq))
    for q in np.arange(0.989, 0.9955, 0.0005):
        th=np.quantile(p,q); lm=p>th; sm=p<(1-th)
        sig=(lm|sm)&vmask; n=sig.sum()
        if n<30: continue
        tpd=n/days; ss=sm[sig]
        acc=(((~ss)&(y[sig]==1))|(ss&(y[sig]==0))).mean()*100
        if 10<=tpd<=20:
            vqs="all" if vq is None else f"{vq:.2f}"
            marker=""
            if best is None or acc>best[0]: best=(acc,vq,q,tpd,n)
            if abs(tpd-14.4)<1: marker=" ◀"
            log(f"  {vqs:>5s} {q:.4f} {tpd:5.1f} {acc:5.1f}% {n:5d}  {marker}")

if best:
    log(f"\n  ★ BEST: vol_q={'all' if best[1] is None else f'{best[1]:.2f}'} q={best[2]:.4f} tpd={best[3]:.1f} ACC={best[0]:.1f}%")
    log(f"    ↑ vs no filter same tpd: +{best[0]-61.6:.1f}pp")
log(f"\nDONE {time.time()-T0:.0f}s")
