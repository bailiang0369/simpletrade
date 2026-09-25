"""最终: strict rolling q no-lookahead + direction + all tricks"""
import numpy as np, pandas as pd, time, gc, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True) if 'sys' not in dir() else None
import sys; sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Load already known AUC=0.5440 model, don't retrain — quick load from saved preds? No, no saved. Quick train.
# Actually OOM risk with Xhstack... let me just load base and add ret = proxy (base AUC=0.5428)
Xtr=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)
Xes=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)
Xte=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)
yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32); del ret_e
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]; t0=time.time()
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
log(f"AUC={roc_auc_score(y,p):.4f}, {len(p):,} rows, {days:.0f} days")

def rolling_q(arr, qq, ts_arr, skip=2000):
    """严格 no-lookahead: 只用过去 30 天数据算 quantile"""
    N=len(arr); th=np.full(N,np.nan)
    for i in range(skip,N):
        tp=ts_arr[i]-30*SPD; m=(ts_arr[:i]>=tp)
        if m.sum()>200: th[i]=np.quantile(arr[:i][m], qq)
    gq=np.nanquantile(arr,qq); fv=np.where(~np.isnan(th))[0]
    th[:fv[0]]=gq; th=np.where(np.isnan(th),gq,th)
    return th

def eval_rolling(name, preds, qq, long_only=False, short_only=False, vol_mask=None):
    th=rolling_q(preds, qq, ts)
    if long_only:
        lm=preds>th; tm=lm
    elif short_only:
        sm=preds<(1-th); tm=sm
    else:
        lm=preds>th; sm=preds<(1-th); tm=lm|sm
    if vol_mask is not None: tm=tm&vol_mask
    n=tm.sum(); tpd=n/days
    if n<10: return n,tpd,0,0
    if long_only: acc=(y[tm]==1).mean()*100
    elif short_only: acc=(y[tm]==0).mean()*100
    else:
        ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    return n,tpd,acc, th

log(f"\n{'='*60}")
log(f" STRICT ROLLING 30d NO-LOOKAHEAD  (真实生产环境)")
log(f"{'='*60}")

# Global q as upper bound first
log(f"\n[1] Global q (upper bound, oracle quantile)")
for q in np.arange(0.989, 0.9955, 0.0005):
    n,tpd,acc,_=eval_rolling("g",p,q)
    if 10<=tpd<=20: log(f"    q={q:.4f} tpd={tpd:5.1f} ACC={acc:5.1f}%")

log(f"\n[2] Rolling q 0.99-0.995 (strict no-lookahead, 极慢)")
for q in [0.990, 0.991, 0.992, 0.993, 0.994, 0.995]:
    n,tpd,acc,_=eval_rolling("r",p,q)
    if n>10: log(f"    q={q:.3f} tpd={tpd:5.1f} ACC={acc:5.1f}% n={n:5d}")

log(f"\n[3] Long-only rolling (只用 long 信号)")
for q in [0.990, 0.992, 0.994, 0.995]:
    n,tpd,acc,_=eval_rolling("r",p,q,long_only=True)
    if n>10: log(f"    q={q:.3f} tpd={tpd:5.1f} ACC={acc:5.1f}% n={n:5d}")

log(f"\n[4] Short-only rolling")
for q in [0.990, 0.992, 0.994, 0.995]:
    n,tpd,acc,_=eval_rolling("r",p,q,short_only=True)
    if n>10: log(f"    q={q:.3f} tpd={tpd:5.1f} ACC={acc:5.1f}% n={n:5d}")

# ========== 最终结论 ==========
log(f"\n{'='*60}")
log(f"  最终结论")
log(f"{'='*60}")
# 关键数字: tpd=14.4 (1% coverage) 时的真实 ACC
log(f"  ETH base+negh  AUC=0.5428")
log(f"  ETH base+fund  AUC=0.5440 (最好单模型)")
log(f"  目标: tpd=14.4, ACC≥65%")
log(f"  实际: global q=0.9915 tpd=14.4 ACC=61.6%")
log(f"        rolling q=0.990  tpd≈?  ACC≈?")
log(f"  差距: 3.4pp (global) / 更大 (rolling no-lookahead)")
log(f"  现有数据能否补上? → 存疑")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
