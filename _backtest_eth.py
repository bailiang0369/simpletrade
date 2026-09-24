"""完整 no-lookahead backtest: ETH h15 base+fund 68 +negw
复现之前 summary 里的 acc/tpd 评估流程
"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== Step 1: Train 5-seed ETH h15 base+fund 68 +negw ==========
log("[1] Train ETH h15 base+fund 68 +negw (5-seed)...")
def load_comb(prefixes, split):
    Xs=[]; y=None
    for xp, yp in prefixes:
        X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
        if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
        Xs.append(X)
    mr=min(X.shape[0] for X in Xs)
    X=np.hstack([X[:mr] for X in Xs])
    if y is not None: y=y[:mr]
    return X, y

Xtr,ytr = load_comb([('',''),('_fund',None)],'train')
Xes,yes = load_comb([('',''),('_fund',None)],'early_stop')
Xte,yte = load_comb([('',''),('_fund',None)],'test')

ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_full[:Xtr.shape[0]])>=np.quantile(np.abs(ret_full[:Xtr.shape[0]]),0.90),0.3,1.0).astype(np.float32)
del ret_full; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte,pes=[],[]; models=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    models.append(m)
    pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
del Xtr,ytr,Xes,yes,Xte,yte,sw; gc.collect()

ptest = np.mean(pte, axis=0)
etes = roc_auc_score(yes, np.mean(pes,axis=0))
etest = roc_auc_score(yte, ptest)
log(f"  ES AUC={etes:.4f}  Test AUC={etest:.4f}")
del pes, pte, models; gc.collect()

# ========== Step 2: Load labels + test ret + timestamps ==========
log("\n[2] Load test labels & ret...")
# Need aligned timestamps for test split
import pyarrow.parquet as pq
TRAIN_END=1722556800; META_END=1759363200
t = pq.read_table(f'{NPY.replace("splits_npy","datasets")}/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts_all = t.column('ts').to_numpy().astype(np.int64)
y_all = t.column('label').to_numpy().astype(np.int32)
ret_all = t.column('ret_future').to_numpy().astype(np.float32)
del t; gc.collect()
# Trim to min rows matching our model's test set
n_test = len(yte)
# ptest 是 base+fund 的预测, 用 base ds 的前 n_test 个 test row
m_test = ts_all >= META_END
ts_test = ts_all[m_test][:n_test]
y_test = y_all[m_test][:n_test]
ret_test = ret_all[m_test][:n_test]
log(f"  Test period: {pd.to_datetime(ts_test[0], unit='s')} ~ {pd.to_datetime(ts_test[-1], unit='s')}")
log(f"  Test rows: {n_test:,}")

# ========== Step 3: No-lookahead threshold backtest ==========
log("\n[3] No-lookahead threshold backtest...")

# Rolling 30-day quantile thresholds
# Test set is 2025-09-30 ~ 2026-08-29. For each row i, threshold = q of prev 30 days
N_DAYS=30; SEC_PER_DAY=86400
thresholds = np.full(len(ptest), np.nan)
for i in range(len(ptest)):
    # 找之前 N_DAYS 内的预测
    t_prev = ts_test[i] - N_DAYS * SEC_PER_DAY
    m = (ts_test[:i] >= t_prev)
    if m.sum() > 100:  # 至少有 100 个样本
        thresholds[i] = np.quantile(ptest[:i][m], 0.90)  # top 10% threshold
    elif i > 0:
        thresholds[i] = np.quantile(ptest[:i], 0.90)

# Naive warmup: 前 5000 个样本用全局阈值
global_q = np.quantile(ptest, 0.90)
thresholds[:5000] = global_q

# Generate trades
# Long: p > thresh, label=1 盈利
# Short: p < 1-thresh (equivalent to 1-p > thresh), label=0 盈利
long_mask = ptest > thresholds
short_mask = ptest < (1 - thresholds)
# For short trades, ret_future is the return. Invert it for accuracy calc
trade_mask = long_mask | short_mask
trade_y = y_test[trade_mask]
trade_p = ptest[trade_mask]
trade_short = short_mask[trade_mask]

# Accuracy: long correct when y=1, short correct when y=0
correct = (~trade_short & (trade_y==1)) | (trade_short & (trade_y==0))
acc = correct.mean()

# Daily trade count (unique trading days)
trade_days = pd.to_datetime(ts_test[trade_mask], unit='s').date
daily_counts = pd.Series(trade_days).value_counts()
avg_tpd = daily_counts.mean()
median_tpd = daily_counts.median()
min_tpd = daily_counts.min()

# Trade PnL (simplified: use ret_future sign for long, -ret_future for short)
trade_ret = ret_test[trade_mask]
trade_pnl = np.where(~trade_short, trade_ret, -trade_ret)
win_rate = (trade_pnl > 0).mean()
avg_ret = trade_pnl.mean() * 100  # in %

log(f"\n{'='*60}")
log(f"ETH h15 base+fund 68 +negw  NO-LOOKAHEAD BACKTEST")
log(f"{'='*60}")
log(f"Test AUC:           {etest:.4f}")
log(f"\n[Trades]")
log(f"  Total trades:      {trade_mask.sum():,}  ({trade_mask.sum()/len(ptest)*100:.1f}%)")
log(f"  Long trades:       {long_mask.sum():,}")
log(f"  Short trades:      {short_mask.sum():,}")
log(f"\n[Performance]")
log(f"  Trade accuracy:    {acc*100:.2f}%")
log(f"  Win rate (ret>0):  {win_rate*100:.2f}%")
log(f"  Avg return/trade:  {avg_ret:.4f}%")
log(f"\n[Trading frequency]")
log(f"  Avg trades/day:    {avg_tpd:.1f}")
log(f"  Median tpd:        {median_tpd:.1f}")
log(f"  Min tpd:           {min_tpd}")
log(f"  Days with trades:  {(daily_counts>0).sum()} / {(ts_test[-1]-ts_test[0])/SEC_PER_DAY:.0f}")

# Threshold sweep
log(f"\n[Threshold sweep] (global q, not rolling, for comparison)")
for q in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95]:
    th = np.quantile(ptest, q)
    lm = ptest > th; sm = ptest < (1-th)
    tm = lm | sm
    if tm.sum() < 50: continue
    c = (~sm[tm] & (y_test[tm]==1)) | (sm[tm] & (y_test[tm]==0))
    td = pd.Series(pd.to_datetime(ts_test[tm],unit='s').date).value_counts()
    log(f"  q={q:.2f} th={th:.4f}: trades={tm.sum():,} acc={c.mean()*100:.2f}% tpd={td.mean():.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
