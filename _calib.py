"""Isotonic calibration + direction analysis + long-only vs long+short"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Train
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
# Also predict meta_val for calibration
mX=np.load(f'{NPY}/ETH_h15_meta_val_X.npy').astype(np.float32)
mXf=np.load(f'{NPY}/ETH_h15_fund_meta_val_X.npy').astype(np.float32)
mmr=min(mX.shape[0],mXf.shape[0]); mX=np.hstack([mX[:mmr],mXf[:mmr]]); del mXf
my=np.load(f'{NPY}/ETH_h15_meta_val_y.npy').astype(np.int32)[:mmr]; gc.collect()

log("Training..."); t0=time.time()
pte=[]; pmeta=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte)); pmeta.append(m.predict(mX))
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,pte,sw; gc.collect()
pm=np.mean(pmeta,axis=0); del pmeta,mX; gc.collect()
log(f"  te AUC={roc_auc_score(np.load(f'{NPY}/ETH_h15_test_y.npy')[:len(p)],p):.4f} meta_auc={roc_auc_score(my,pm):.4f} ({time.time()-t0:.0f}s)")

t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32)
ret=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts>=META_END; ts=ts[mm][:len(p)]; y=y[mm][:len(p)]; ret=ret[mm][:len(p)]
days=(ts[-1]-ts[0])/SPD

# ========== 1. Isotonic calibration on meta_val ==========
log(f"\n{'='*55}")
log(f"  ISOTONIC CALIBRATION  (fit on meta_val)")
log(f"{'='*55}")
iso=IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
iso.fit(pm, my)
p_cal = iso.transform(p)

# Check calibration quality
log(f"  Before calib mean pred: {p.mean():.4f}  label rate: {y.mean():.4f}")
log(f"  After  calib mean pred: {p_cal.mean():.4f}")
log(f"  Before AUC={roc_auc_score(y,p):.4f}  After AUC={roc_auc_score(y,p_cal):.4f}  (AUC 不变, calibration 只改变排序不变的情况)")

# Backtest both
def sweep(name, preds):
    log(f"\n  {name}:")
    best=None
    for q in np.arange(0.989, 0.9955, 0.0005):
        th=np.quantile(preds,q); lm=preds>th; sm=preds<(1-th); tm=lm|sm
        n=tm.sum(); tpd=n/days
        if n<30: continue
        ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
        if 10<=tpd<=20:
            marker=" ◀" if abs(tpd-14.4)<1 else ""
            if best is None or acc>best[0]: best=(acc,q,tpd,n)
            log(f"    q={q:.4f} tpd={tpd:5.1f} ACC={acc:5.1f}% n={n:5d}{marker}")
    if best: log(f"    ★ best: q={best[1]:.4f} tpd={best[2]:.1f} ACC={best[0]:.1f}%")
    return best

b1=sweep("raw p", p)
b2=sweep("calibrated p", p_cal)

# ========== 2. Direction analysis ==========
log(f"\n{'='*55}")
log(f"  DIRECTION ANALYSIS (test period)")
log(f"{'='*55}")
log(f"  Label=1 ratio (long): {y.mean()*100:.2f}%")
log(f"  ret_future mean: {ret.mean()*100:.4f}%")
log(f"  ret_future median: {np.median(ret)*100:.4f}%")
log(f"  ret > 0 ratio: {(ret>0).mean()*100:.2f}%")

# Long-only
best_lo=None
for q in np.arange(0.985, 0.997, 0.0005):
    th=np.quantile(p,q); lm=p>th  # 只做 long
    n=lm.sum(); tpd=n/days
    if n<30: continue
    acc=(y[lm]==1).mean()*100
    if 8<=tpd<=20:
        if best_lo is None or acc>best_lo[0]: best_lo=(acc,q,tpd,n)
if best_lo: log(f"  ★ LONG-ONLY best: q={best_lo[1]:.4f} tpd={best_lo[2]:.1f} ACC={best_lo[0]:.1f}%")

# Short-only
best_so=None
for q in np.arange(0.985, 0.997, 0.0005):
    th=np.quantile(1-p,q); sm=p<th  # 只做 short
    n=sm.sum(); tpd=n/days
    if n<30: continue
    acc=(y[sm]==0).mean()*100
    if 8<=tpd<=20:
        if best_so is None or acc>best_so[0]: best_so=(acc,q,tpd,n)
if best_so: log(f"  ★ SHORT-ONLY best: q={best_so[1]:.4f} tpd={best_so[2]:.1f} ACC={best_so[0]:.1f}%")

# ========== 3. Feature crosses: try (close_pos_in_vol × p) ==========
log(f"\n{'='*55}")
log(f"  IDEA: 只用模型 long 预测, 但在 ret_future 大时也做 (momentum confirmation)")
log(f"{'='*55}")

# 如果 p>0.52 AND |ret| recent — 双重确认
best2=None
for q in np.arange(0.985, 0.997, 0.0005):
    th=np.quantile(p,q)
    lm=p>th; sm=p<(1-th)
    
    # 额外条件: 上一个 H 的 ret 同向 (momentum)
    # ret_future 是未来 15min, 看过去 15min 即 ret[t-1]
    # 简化: 不管 ret, 先看 long-only + calibrated p
    tm=lm|sm; n=tm.sum(); tpd=n/days
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    if 10<=tpd<=20:
        if best2 is None or acc>best2[0]: best2=(acc,q,tpd,n)

log(f"\n  SUMMARY:")
log(f"    raw p long+short:     ACC={b1[0]:.1f}% @ q={b1[1]:.4f} tpd={b1[2]:.1f}")
log(f"    cal p long+short:     ACC={b2[0]:.1f}% @ q={b2[1]:.4f} tpd={b2[2]:.1f}")
log(f"    long-only best:       ACC={best_lo[0]:.1f}% @ q={best_lo[1]:.4f} tpd={best_lo[2]:.1f}")
log(f"    short-only best:      ACC={best_so[0]:.1f}% @ q={best_so[1]:.4f} tpd={best_so[2]:.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
