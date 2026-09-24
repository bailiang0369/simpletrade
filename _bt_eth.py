"""ETH h15 backtest: train first, then evaluate"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Load + train
def load_c(prefixes, split):
    Xs=[]; y=None
    for xp, yp in prefixes:
        X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
        if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
        Xs.append(X)
    mr=min(X.shape[0] for X in Xs)
    return np.hstack([X[:mr] for X in Xs]), y[:mr] if y is not None else None

Xtr,ytr = load_c([('',''),('_fund',None)],'train')
Xes,yes = load_c([('',''),('_fund',None)],'early_stop')
Xte,yte = load_c([('',''),('_fund',None)],'test')

ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_full[:Xtr.shape[0]])>=np.quantile(np.abs(ret_full[:Xtr.shape[0]]),0.90),0.3,1.0).astype(np.float32)

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte,pes=[],[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
ptest=np.mean(pte,axis=0); pes_m=np.mean(pes,axis=0)
etes=roc_auc_score(yes,pes_m); etest=roc_auc_score(yte,ptest)
log(f"[1] Train done: ES={etes:.4f} Test={etest:.4f}")
del Xtr,ytr,Xes,yes,Xte,yte,sw,pte,pes,ret_full; gc.collect()

# Get aligned test timestamps
import pyarrow.parquet as pq
TRAIN_END=1722556800; META_END=1759363200
t = pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts_all = t.column('ts').to_numpy().astype(np.int64)
y_all = t.column('label').to_numpy().astype(np.int32)
ret_all = t.column('ret_future').to_numpy().astype(np.float32)
del t; gc.collect()
m_test = ts_all >= META_END
ts = ts_all[m_test][:len(ptest)]
y = y_all[m_test][:len(ptest)]
ret = ret_all[m_test][:len(ptest)]

log(f"\n[2] Test period: {pd.to_datetime(ts[0],unit='s')} ~ {pd.to_datetime(ts[-1],unit='s')}")
log(f"    Rows: {len(ptest):,}")

# ========== No-lookahead rolling 30-day quantile ==========
log("\n[3] Rolling threshold no-lookahead backtest...")
N_DAYS=30; SPD=86400
thresh = np.full(len(ptest), np.nan)
for i in range(len(ptest)):
    if i < 2000: continue  # warmup
    t_prev = ts[i] - N_DAYS*SPD
    m = (ts[:i] >= t_prev)
    if m.sum() > 200:
        thresh[i] = np.quantile(ptest[:i][m], 0.90)

global_q = np.nanquantile(ptest, 0.90)
thresh[:np.where(~np.isnan(thresh))[0][0]] = global_q
thresh = np.where(np.isnan(thresh), global_q, thresh)

long_m = ptest > thresh
short_m = ptest < (1 - thresh)
trade_m = long_m | short_m
trade_y = y[trade_m]; trade_short = short_m[trade_m]; trade_ret = ret[trade_m]
corr = (~trade_short & (trade_y==1)) | (trade_short & (trade_y==0))
acc = corr.mean()
tn = pd.to_datetime(ts[trade_m],unit='s').date
td = pd.Series(tn).value_counts()

log(f"{'='*55}")
log(f" ETH h15 base+fund 68 +negw  |  No-Lookahead")
log(f"{'='*55}")
log(f" Test AUC:       {etest:.4f}")
log(f" Threshold:      rolling 30d q=0.90")
log(f" Total trades:   {trade_m.sum():,} ({trade_m.sum()/len(ptest)*100:.1f}%)")
log(f"  Long: {long_m.sum():,}  Short: {short_m.sum():,}")
log(f" Trade ACC:      {acc*100:.2f}%")
log(f" Win rate:       {(np.where(~trade_short,trade_ret,-trade_ret)>0).mean()*100:.2f}%")
log(f" Avg ret/trd:    {np.where(~trade_short,trade_ret,-trade_ret).mean()*100:.4f}%")
log(f" Avg tpd:        {td.mean():.1f}  Median: {td.median():.1f}  Min: {td.min()}")

# Global q sweep (not rolling - upper bound)
log(f"\n[4] Global q sweep (upper bound, comparison)...")
for q in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97]:
    th = np.quantile(ptest, q)
    lm = ptest > th; sm = ptest < (1-th); tm = lm | sm
    if tm.sum() < 30: continue
    c = (~sm[tm] & (y[tm]==1)) | (sm[tm] & (y[tm]==0))
    tdd = pd.Series(pd.to_datetime(ts[tm],unit='s').date).value_counts()
    log(f"  q={q:.2f} th={th:.4f}: n={tm.sum():,} acc={c.mean()*100:.2f}% tpd={tdd.mean():.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
