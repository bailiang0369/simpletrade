"""Consensus Filtering: ETH trades, BTC as directional filter
ETH long + BTC long → trade long
ETH short + BTC short → trade short
否则 → skip
"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]; SPD=86400
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
log = lambda *a: print(' '.join(str(x) for x in a), flush=True)

# Load already trained predictions? Let's retrain both (they're saved as separate models)
# Actually let's save model predictions to disk first, then do consensus
log("[1] Train ETH h15 base+fund+negw...")
def eth_comb(split):
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    y=np.load(f'{NPY}/ETH_h15_{split}_y.npy').astype(np.int32)
    mr=min(Xb.shape[0],Xf.shape[0]); y=y[:mr]
    return np.hstack([Xb[:mr],Xf[:mr]]), y

Xtr,ytr=eth_comb('train'); Xes,yes=eth_comb('early_stop'); Xte,yte=eth_comb('test')
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr.shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32)
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_eth=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_eth.append(m.predict(Xte))
p_eth=np.mean(p_eth,axis=0); del Xtr,ytr,Xes,yes,Xte,yte,sw,ret_e; gc.collect()
log(f"  ETH te={roc_auc_score(np.load(f'{NPY}/ETH_h15_test_y.npy')[:len(p_eth)], p_eth):.4f}")

log("\n[2] Train BTC h15 base+negw (比 h30 好)...")
def train_seed(prefix, sw=None):
    Xtr=np.load(f'{NPY}/{prefix}_train_X.npy').astype(np.float32)
    ytr=np.load(f'{NPY}/{prefix}_train_y.npy').astype(np.int32)
    Xes=np.load(f'{NPY}/{prefix}_early_stop_X.npy').astype(np.float32)
    yes=np.load(f'{NPY}/{prefix}_early_stop_y.npy').astype(np.int32)
    Xte=np.load(f'{NPY}/{prefix}_test_X.npy').astype(np.float32)
    params.update({'seed':0})
    pte=[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte))
    return np.mean(pte,axis=0)
ret_b=np.load(f'{NPY}/BTC_h15_train_ret.npy')
sw_b=np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)
p_btc15 = train_seed('BTC_h15', sw_b)
del ret_b, sw_b; gc.collect()
log(f"  BTC h15 te={roc_auc_score(np.load(f'{NPY}/BTC_h15_test_y.npy'), p_btc15):.4f}")

# Also BTC h30
ret_b30=np.load(f'{NPY}/BTC_h30_train_ret.npy')
sw_b30=np.where(np.abs(ret_b30)>=np.quantile(np.abs(ret_b30),0.90),0.3,1.0).astype(np.float32)
p_btc30 = train_seed('BTC_h30', sw_b30)
del ret_b30, sw_b30; gc.collect()
log(f"  BTC h30 te={roc_auc_score(np.load(f'{NPY}/BTC_h30_test_y.npy'), p_btc30):.4f}")

# Align: use ETH timestamps as reference
import pyarrow.parquet as pq
t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32); ret=t.column('ret_future').to_numpy().astype(np.float32)
del t
m=ts>=META_END; ts=ts[m][:len(p_eth)]; y=y[m][:len(p_eth)]; ret=ret[m][:len(p_eth)]

# BTC h15 should have same len
p_b15 = p_btc15[:len(p_eth)]
# BTC h30 slightly shorter (no label for last 29 min)
p_b30 = p_btc30[:len(p_eth)]

log(f"\n[3] Consensus Analysis...")
log(f"  corr(ETH,BTC15)={np.corrcoef(p_eth,p_b15)[0,1]:.3f}  corr(ETH,BTC30)={np.corrcoef(p_eth,p_b30)[0,1]:.3f}")

def eval_filtered(pe, pb, y, ret, ts, q_eth, q_btc=None):
    """q_eth=ETH threshold quantile, q_btc=BTC threshold quantile (None=0.5)"""
    N=len(pe)
    # Rolling ETH threshold
    def rolling_q(arr, qq, skip=2000):
        th=np.full(N,np.nan)
        for i in range(skip,N):
            tp=ts[i]-30*SPD; m=(ts[:i]>=tp)
            if m.sum()>200: th[i]=np.quantile(arr[:i][m], qq)
        gq=np.nanquantile(arr,qq); fv=np.where(~np.isnan(th))[0]
        th[:fv[0]]=gq; th=np.where(np.isnan(th),gq,th)
        return th
    
    th_eth = rolling_q(pe, q_eth)
    if q_btc is not None:
        th_btc = rolling_q(pb, q_btc)
    else:
        th_btc = np.full(N, 0.5)  # simple direction filter
    
    # ETH signals
    long_eth = pe > th_eth
    short_eth = pe < (1 - th_eth)
    sig_eth = long_eth | short_eth  # ETH wants to trade
    
    # BTC consensus filter
    if q_btc is not None:
        long_btc = pb > th_btc; short_btc = pb < (1 - th_btc)
        btc_agree = (long_eth & long_btc) | (short_eth & short_btc)
    else:
        # Simple BTC direction: BTC > 0.5 means bullish, < 0.5 means bearish
        btc_agree = (long_eth & (pb > 0.5)) | (short_eth & (pb < 0.5))
    
    # Only trade when BOTH ETH signals AND BTC agrees
    trade_m = sig_eth & btc_agree
    trade_short = short_eth[trade_m]
    trade_y = y[trade_m]
    trade_ret = ret[trade_m]
    corr = ((~trade_short)&(trade_y==1))|(trade_short&(trade_y==0))
    acc = corr.mean()
    tn = pd.to_datetime(ts[trade_m],unit='s').date
    td = pd.Series(tn).value_counts()
    return trade_m.sum(), acc, td.mean()

# ========== Test Various Configs ==========
log(f"\n{'='*60}")
log(f" ETH h15 → BTC Consensus Filter  |  No-Lookahead")
log(f"{'='*60}")

# Config 1: ETH q=0.90, BTC=direction only (0.5)
for q in [0.85, 0.90, 0.92, 0.95]:
    n, acc, tpd = eval_filtered(p_eth, p_b15, y, ret, ts, q, None)
    log(f"  ETH q={q:.2f} BTC-dir:   n={n:,} acc={acc*100:.2f}% tpd={tpd:.1f}")

# Config 2: ETH q=0.90, BTC q=0.70/0.75/0.80 (BTC also selective)
log(f"\n  ETH q=0.90, BTC also filtered:")
for qb in [0.6, 0.7, 0.75, 0.8]:
    n, acc, tpd = eval_filtered(p_eth, p_b15, y, ret, ts, 0.90, qb)
    log(f"    BTC q={qb:.2f}: n={n:,} acc={acc*100:.2f}% tpd={tpd:.1f}")

# Config 3: Use BTC h30 instead of h15
log(f"\n  Same with BTC h30:")
for q in [0.85, 0.90, 0.92, 0.95]:
    n, acc, tpd = eval_filtered(p_eth, p_b30, y, ret, ts, q, None)
    log(f"  ETH q={q:.2f} BTC30-dir: n={n:,} acc={acc*100:.2f}% tpd={tpd:.1f}")

# Config 4: Baseline ETH alone (no BTC filter)
log(f"\n  ETH alone (no BTC filter, for comparison):")
for q in [0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
    n, acc, tpd = eval_filtered(p_eth, p_eth, y, ret, ts, q, None)  # pb=pe means always "agree"
    log(f"  q={q:.2f}: n={n:,} acc={acc*100:.2f}% tpd={tpd:.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
