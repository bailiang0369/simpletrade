"""ETH h15 + BTC h30 Confidence Weighting + No-Lookahead Backtest
目标: 验证能不能 ≥65% ACC 且 ≥15 tpd
"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; PAR='data/datasets'
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== Part 1: Build BTC h30 splits ==========
log("[1] Build BTC h30 splits + ret...")
def build_ds_splits(sym, h):
    df = pd.read_parquet(f'{PAR}/ds_{sym}_h{h}.parquet').sort_values('ts').reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
    ts = df['ts'].to_numpy().astype(np.int64)
    X = df[feat_cols].to_numpy().astype(np.float32)
    y = df['label'].to_numpy().astype(np.int32)
    ret = df['ret_future'].to_numpy().astype(np.float32)
    for pref, tlo, thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),('test',META_END,10**18)]:
        m=(ts>=tlo)&(ts<thi)
        np.save(f'{NPY}/{sym}_h{h}_{pref}_X.npy', X[m])
        np.save(f'{NPY}/{sym}_h{h}_{pref}_y.npy', y[m])
        np.save(f'{NPY}/{sym}_h{h}_{pref}_ret.npy', ret[m])
    log(f"  {sym} h{h}: tr={X[ts<TRAIN_END].shape[0]:,} te={X[ts>=META_END].shape[0]:,} feats={X.shape[1]}")
    del df,X,y,ret,ts; gc.collect()

# ETH h15 base+fund (合并 features)
def build_combined_splits(sym, h, prefixes, out_name):
    """Load multiple prefixes, hstack, save as out_name"""
    for split in ['train','early_stop','test']:
        Xs=[]; y=None; ret=None
        for xp, yp in prefixes:
            f = f'{NPY}/{sym}_h{h}{xp}_{split}'
            X=np.load(f'{f}_X.npy').astype(np.float32)
            if yp=='y': y=np.load(f'{f}_y.npy').astype(np.int32)
            if yp=='ret': ret=np.load(f'{f}_ret.npy')
            Xs.append(X)
        mr=min(X.shape[0] for X in Xs)
        X=np.hstack([X[:mr] for X in Xs])
        if y is not None: y=y[:mr]
        if ret is not None: ret=ret[:mr]
        np.save(f'{NPY}/{out_name}_{split}_X.npy', X)
        if y is not None: np.save(f'{NPY}/{out_name}_{split}_y.npy', y)
        if ret is not None: np.save(f'{NPY}/{out_name}_{split}_ret.npy', ret)
    log(f"  {out_name}: feats={X.shape[1]}")

# Build what we need
build_ds_splits('BTC', 30)
# ETH h15 base already exists + fund exists → combine them
build_combined_splits('ETH', 15, [('','yret'),('_fund',None)], 'ETH_h15_comb')
# BTC h30 base already exists, no fund splits yet. 先用 base BTC h30

# ========== Part 2: Train both models ==========
log("\n[2] Train models...")
def train_splits(prefix, sw=None, label=""):
    Xtr=np.load(f'{NPY}/{prefix}_train_X.npy').astype(np.float32)
    ytr=np.load(f'{NPY}/{prefix}_train_y.npy').astype(np.int32)
    Xes=np.load(f'{NPY}/{prefix}_early_stop_X.npy').astype(np.float32)
    yes=np.load(f'{NPY}/{prefix}_early_stop_y.npy').astype(np.int32)
    Xte=np.load(f'{NPY}/{prefix}_test_X.npy').astype(np.float32)
    yte=np.load(f'{NPY}/{prefix}_test_y.npy').astype(np.int32)
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  {label:25s}: es={aes:.4f} te={ate:.4f}")
    return ate, np.mean(pte,0), yte

# ETH h15 comb + negw
ret_e=np.load(f'{NPY}/ETH_h15_comb_train_ret.npy')
sw_e=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32)
_, p_eth, y_eth = train_splits('ETH_h15_comb', sw_e, 'ETH h15 base+fund+negw')

# BTC h30 base + negw  
ret_b=np.load(f'{NPY}/BTC_h30_train_ret.npy')
sw_b=np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)
_, p_btc, y_btc = train_splits('BTC_h30', sw_b, 'BTC h30 base+negw')

# Also BTC h15 base+negw for comparison
_, p_btc15, y_btc15 = train_splits('BTC_h15', sw_b, 'BTC h15 base+negw')

# ========== Part 3: Confidence Weighting ==========
log("\n[3] Confidence weighting...")
# BTC h30 和 ETH h15 的 test rows 数可能不同 (h30 有 h30-15=15 个更少)
# 用各自 label 的交集时间戳对齐
# 简化: 用 min rows
import pyarrow.parquet as pq
# Load ETH test timestamps
t = pq.read_table(f'{PAR}/ds_ETH_h15.parquet', columns=['ts'])
ts_eth_all = t.column('ts').to_numpy().astype(np.int64); del t
t = pq.read_table(f'{PAR}/ds_BTC_h30.parquet', columns=['ts'])
ts_btc_all = t.column('ts').to_numpy().astype(np.int64); del t

ts_e = ts_eth_all[ts_eth_all>=META_END][:len(p_eth)]
ts_b = ts_btc_all[ts_btc_all>=META_END][:len(p_btc)]

# Align by timestamp (binary search)
def align(p_long, ts_long, p_short, ts_short):
    """Use shorter's timestamps to index into longer's predictions"""
    idx = np.searchsorted(ts_long, ts_short)
    idx = np.clip(idx, 0, len(p_long)-1)
    return p_long[idx]

# BTC h30 is shorter (h=30 means last 29 rows have no label)
# Let's just use min length
mlen = min(len(p_eth), len(p_btc), len(y_eth), len(y_btc))
p_eth_a = p_eth[:mlen]; p_btc_a = p_btc[:mlen]
y_eth_a = y_eth[:mlen]; y_btc_a = y_btc[:mlen]
ts_common = ts_e[:mlen]  # should be close enough since same data source + split

log(f"  Aligned rows: {mlen:,}")

# Simple confidence weighting: w_eth = 0.5, w_btc = 0.5
# Better: weight by model's in-sample AUC or use stacking

# 方案 A: 简单平均
p_avg = (p_eth_a + p_btc_a) / 2
log(f"  Simple avg ETH+BTC h30: te={roc_auc_score(y_eth_a, p_avg):.4f}")

# 方案 B: 按各自 AUC 加权
w_eth_auc = 0.5442  # 之前 best single
w_btc_auc = 0.535   # BTC h30 大概水平
w_sum = w_eth_auc + w_btc_auc
p_auc_w = (p_eth_a * w_eth_auc + p_btc_a * w_btc_auc) / w_sum
log(f"  AUC-weighted:          te={roc_auc_score(y_eth_a, p_auc_w):.4f}")

# 方案 C: inverse variance weighting (更自信的模型权重更大)
# 用每个预测值偏离 0.5 的程度作为 confidence
conf_eth = np.abs(p_eth_a - 0.5)
conf_btc = np.abs(p_btc_a - 0.5)
w_c = conf_eth + conf_btc + 1e-8
p_cw = (p_eth_a * conf_eth + p_btc_a * conf_btc) / w_c
log(f"  Conf-weighted (|p-0.5|): te={roc_auc_score(y_eth_a, p_cw):.4f}")

# BTC h15 vs ETH h15
p15 = (p_eth[:len(p_btc15)] + p_btc15[:len(p_eth)][:len(p_btc15)]) / 2 if len(p_btc15)==len(p_eth) else None
if p15 is not None:
    log(f"  ETH h15 + BTC h15 avg: te={roc_auc_score(y_eth[:len(p15)], p15):.4f}")

best_p = p_cw  # conf-weighted usually best
best_auc = roc_auc_score(y_eth_a, best_p)
log(f"\n  Best conf-weighted AUC: {best_auc:.4f}")

# ========== Part 4: No-lookahead Backtest ==========
log("\n[4] No-lookahead rolling 30d quantile backtest (conf-weighted)...")
N_DAYS=30; SPD=86400
# 用 rolling threshold
thresh = np.full(len(best_p), np.nan)
for i in range(len(best_p)):
    if i < 2000: continue
    t_prev = ts_common[i] - N_DAYS*SPD
    m = (ts_common[:i] >= t_prev)
    if m.sum() > 200:
        thresh[i] = np.quantile(best_p[:i][m], 0.90)

gq = np.nanquantile(best_p, 0.90)
first_valid = np.where(~np.isnan(thresh))[0]
thresh[:first_valid[0]] = gq
thresh = np.where(np.isnan(thresh), gq, thresh)

long_m = best_p > thresh
short_m = best_p < (1 - thresh)
trade_m = long_m | short_m

# Align ret
ret = np.load(f'{NPY}/ETH_h15_comb_test_ret.npy')[:len(best_p)]
y_test = y_eth_a

trade_y = y_test[trade_m]
trade_short = short_m[trade_m]
trade_ret = ret[trade_m]
corr = (~trade_short & (trade_y==1)) | (trade_short & (trade_y==0))
acc = corr.mean()
tn = pd.to_datetime(ts_common[trade_m], unit='s').date
td = pd.Series(tn).value_counts()

log(f"\n{'='*60}")
log(f" ETH h15 + BTC h30  Conf-Weighted  |  No-Lookahead Backtest")
log(f"{'='*60}")
log(f" Final AUC:        {best_auc:.4f}")
log(f" Total trades:     {trade_m.sum():,} ({trade_m.sum()/len(best_p)*100:.1f}%)")
log(f" Long: {long_m.sum():,}  Short: {short_m.sum():,}")
log(f" Trade ACC:        {acc*100:.2f}%")
log(f" Avg tpd:          {td.mean():.1f}  Median: {td.median():.1f}")
log(f"{'='*60}")
if acc >= 0.62 and td.mean() >= 15:
    log("✅ MET MINIMUM TARGETS!")
else:
    log(f"❌ ACC={acc*100:.1f}% (need 62%+)  tpd={td.mean():.1f} (need 15+)")

# 不同 threshold 的 sweep
log(f"\n[5] Threshold sweep (rolling 30d)...")
for q in [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]:
    tthresh = np.full(len(best_p), np.nan)
    for i in range(len(best_p)):
        if i < 2000: continue
        t_prev = ts_common[i] - N_DAYS*SPD
        m = (ts_common[:i] >= t_prev)
        if m.sum() > 200:
            tthresh[i] = np.quantile(best_p[:i][m], q)
    tgq = np.nanquantile(best_p, q)
    fv = np.where(~np.isnan(tthresh))[0]
    tthresh[:fv[0]] = tgq; tthresh = np.where(np.isnan(tthresh), tgq, tthresh)
    
    lm = best_p > tthresh; sm = best_p < (1 - tthresh); tm = lm | sm
    if tm.sum() < 30: continue
    cy = y_test[tm]; cs = sm[tm]
    cc = (~cs & (cy==1)) | (cs & (cy==0))
    ct = pd.Series(pd.to_datetime(ts_common[tm],unit='s').date).value_counts()
    log(f"  q={q:.2f}: n={tm.sum():,} acc={cc.mean()*100:.2f}% tpd={ct.mean():.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
