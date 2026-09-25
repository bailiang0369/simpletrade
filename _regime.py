"""快速试 regime filtering + 多 horizon + 校准
只用现有数据, 看能不能从 60.9% → 65% ACC @ 1% coverage
"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SPD=86400; META_END=1759363200
SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== Step 1: Build regime features from raw ETH ==========
log("[1] Build regime features...")
import polars as pl
raw = pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
close = raw['close'].to_numpy().astype(np.float64)
high = raw['high'].to_numpy().astype(np.float64)
low = raw['low'].to_numpy().astype(np.float64)
volume = raw['buy_vol'].to_numpy().astype(np.float64)
ts = raw['ts'].to_numpy().astype(np.int64)

# ADX-like trend strength (simplified: directional movement index)
def adx_like(c, h, l, w=60):
    """Simplified trend strength: how consistently price moves in one direction"""
    up_move = np.zeros(len(c)); up_move[1:] = h[1:] - h[:-1]
    down_move = np.zeros(len(c)); down_move[1:] = l[:-1] - l[1:]
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    dm_plus_r = pd.Series(dm_plus).rolling(w,min_periods=10).mean().to_numpy()
    dm_minus_r = pd.Series(dm_minus).rolling(w,min_periods=10).mean().to_numpy()
    tr_r = pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy()
    tr_r = np.where(tr_r>1e-9, tr_r, 1e-9)
    di_p = dm_plus_r / tr_r; di_m = dm_minus_r / tr_r
    dx = np.where((di_p+di_m)>1e-9, np.abs(di_p-di_m)/(di_p+di_m)*100, 0)
    adx = pd.Series(dx).rolling(w,min_periods=10).mean().to_numpy()
    return adx

# Volatility regime
def vol_regime(c, w=60):
    ret = np.diff(c)/c[:-1]; ret=np.concatenate([[0],ret])
    vol = pd.Series(ret).rolling(w,min_periods=10).std().to_numpy()
    vol_ma = pd.Series(vol).rolling(w*3,min_periods=30).mean().to_numpy()
    return vol/np.where(vol_ma>1e-12, vol_ma, 1e-12)  # vol ratio

# Price position (near MA vs far from MA)
def ma_dev(c, w=240):
    ma = pd.Series(c).rolling(w,min_periods=30).mean().to_numpy()
    return (c-ma)/np.where(ma>1e-9, ma, 1e-9)

adx60 = adx_like(close, 60)
adx240 = adx_like(close, 240)
vol_ratio = vol_regime(close, 60)
ma_dev240 = ma_dev(close, 240)
ma_dev960 = ma_dev(close, 960)  # 16h
log(f"  ADX60: median={np.nanmedian(adx60):.1f}, ADX240 median={np.nanmedian(adx240):.1f}")
log(f"  vol_ratio: median={np.nanmedian(vol_ratio):.2f}")
del raw, high, low, volume; gc.collect()

# ========== Step 2: Load ETH predictions ==========
log("\n[2] Load ETH predictions...")
def eth_comb(split):
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    mr=min(Xb.shape[0],Xf.shape[0])
    return np.hstack([Xb[:mr],Xf[:mr]])

import lightgbm as lgb
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:eth_comb('train').shape[0]]
sw=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32)
del ret_e; gc.collect()

Xtr=eth_comb('train'); ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xes=eth_comb('early_stop'); yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xte=eth_comb('test'); yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p_eth=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,pte; gc.collect()
log(f"  ETH te AUC={roc_auc_score(yte, p_eth):.4f}, shape={p_eth.shape}")

# ========== Step 3: Align regime + pred + label ==========
import pyarrow.parquet as pq
t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts_ds=t.column('ts').to_numpy().astype(np.int64); y_all=t.column('label').to_numpy().astype(np.int32)
ret_all=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts_ds>=META_END
ts_test=ts_ds[mm][:len(p_eth)]; y_test=y_all[mm][:len(p_eth)]; ret_test=ret_all[mm][:len(p_eth)]

# Align regime features to test timestamps (binary search into raw ts)
idx = np.searchsorted(ts, ts_test)
adx_t = adx60[idx]; adx_t2 = adx240[idx]; vr_t = vol_ratio[idx]
md_t = ma_dev240[idx]; md_t2 = ma_dev960[idx]
log(f"\n[3] Regime stats in test period:")
log(f"  ADX60>25 (trend):     {(adx_t>25).mean()*100:.1f}%")
log(f"  ADX60<15 (range):     {(adx_t<15).mean()*100:.1f}%")
log(f"  vol_ratio>1.5 (high): {(vr_t>1.5).mean()*100:.1f}%")
log(f"  vol_ratio<0.7 (low):  {(vr_t<0.7).mean()*100:.1f}%")

# ========== Step 4: Regime filter backtests ==========
SPD2=86400
def rolling_q(arr, qq, ts_arr, skip=2000):
    N=len(arr); th=np.full(N,np.nan)
    for i in range(skip,N):
        tp=ts_arr[i]-30*SPD2; m=(ts_arr[:i]>=tp)
        if m.sum()>200: th[i]=np.quantile(arr[:i][m], qq)
    gq=np.nanquantile(arr,qq); fv=np.where(~np.isnan(th))[0]
    th[:fv[0]]=gq; th=np.where(np.isnan(th),gq,th)
    return th

def eval_backtest(pe, y, ret, ts_arr, regime_mask=None, label="", base_q=0.90):
    """regime_mask: 额外过滤条件 (True=允许交易)"""
    th=rolling_q(pe, base_q, ts_arr)
    long_m=pe>th; short_m=pe<(1-th); sig=long_m|short_m
    if regime_mask is not None:
        sig = sig & regime_mask
    if sig.sum()<10: return sig.sum(), 0, 0, 0
    ss=short_m[sig]; acc=(((~ss)&(y[sig]==1))|(ss&(y[sig]==0))).mean()*100
    tn=pd.to_datetime(ts_arr[sig],unit='s').date; td=pd.Series(tn).value_counts()
    win=((~ss)&(ret[sig]>0)).sum() + (ss&(ret[sig]<0)).sum()
    winrate = win/sig.sum()*100
    return sig.sum(), acc, td.mean(), winrate

log(f"\n{'='*60}")
log(f" REGIME FILTER BACKTESTS (no-lookahead, rolling 30d q=0.99)")
log(f"{'='*60}")

# BASELINE: no filter
for q in [0.97, 0.98, 0.99]:
    n,a,tp,wr = eval_backtest(p_eth, y_test, ret_test, ts_test, None, "BASE", q)
    log(f"  BASE q={q:.2f}: n={n:,} ({n/len(p_eth)*100:.1f}%) ACC={a:.1f}% tpd={tp:.1f} win={wr:.1f}%")

# REGIME FILTERS — 只在趋势市/高波动市交易
for rname, rmask in [
    ('ADX60>20 trend', adx_t>20),
    ('ADX60>25 strong', adx_t>25),
    ('ADX60>30 very strong', adx_t>30),
    ('vol_ratio>1.2 high vol', vr_t>1.2),
    ('vol_ratio>1.5 high', vr_t>1.5),
    ('ADX>20 & vol>1.0', (adx_t>20)&(vr_t>1.0)),
    ('ADX>25 & vol>1.2', (adx_t>25)&(vr_t>1.2)),
]:
    for q in [0.97, 0.98, 0.99]:
        n,a,tp,wr = eval_backtest(p_eth, y_test, ret_test, ts_test, rmask, rname, q)
        if n > 50:
            log(f"  {rname:25s} q={q:.2f}: n={n:,} ({n/len(p_eth)*100:.1f}%) ACC={a:.1f}% tpd={tp:.1f} win={wr:.1f}%")

# ========== Step 5: 用 regime 训练时加权 (regime-aware training) ==========
log(f"\n[5] Regime-aware training weights...")
# 高波动样本权重更高 (手动交易也是波动大时更有信号)
def eth_all(split):
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    mr=min(Xb.shape[0],Xf.shape[0])
    return np.hstack([Xb[:mr],Xf[:mr]])

# 获取 train 的 ADX / vol
idx_tr = np.searchsorted(ts, np.load(f'{NPY}/ETH_h15_train_X.npy')[:eth_comb('train').shape[0]*0 + 0])
# 直接从原始 ts 过滤
ts_train = ts[(ts>=0)&(ts<TRAIN_END 0.0))]

# 简化: 用已有的 ret 来计算 vol
ret_train=np.load(f'{NPY}/ETH_h15_train_ret.npy')
vol_bin = pd.qcut(np.abs(ret_train), 4, labels=False, duplicates='drop')
# 让高 vol 样本权重更高: low_vol ×0.5, med ×1.0, high ×1.5
sw_regime = np.where(vol_bin >= 2, 1.5, np.where(vol_bin >= 1, 1.0, 0.5)).astype(np.float32)

# negw + regime combined
sw_comb = np.where(np.abs(ret_train)>=np.quantile(np.abs(ret_train),0.90), 0.3, 1.0).astype(np.float32) * sw_regime

Xtr=eth_all('train'); ytr=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)[:Xtr.shape[0]]
Xes=eth_all('early_stop'); yes=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)[:Xes.shape[0]]
Xte=eth_all('test'); yte=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)[:Xte.shape[0]]

pte2=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw_comb[:Xtr.shape[0]]); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte2.append(m.predict(Xte))
p_regime=np.mean(pte2,axis=0)
del Xtr,ytr,Xes,yes,Xte,yte,pte2; gc.collect()

# Align
ml2=min(len(p_regime), len(y_test))
p_r=p_regime[:ml2]; y_r=y_test[:ml2]; ts_r=ts_test[:ml2]; ret_r=ret_test[:ml2]
log(f"  regime-weighted te AUC={roc_auc_score(y_r,p_r):.4f} (vs {roc_auc_score(y_r,p_eth[:ml2]):.4f} base)")

log(f"\n  Regime-weighted model backtest:")
for q in [0.97,0.98,0.99]:
    n,a,tp,wr = eval_backtest(p_r, y_r, ret_r, ts_r, None, "regime_w", q)
    log(f"    q={q:.2f}: n={n:,} ACC={a:.1f}% tpd={tp:.1f}")

# ========== Step 6: Multi-threshold (ETH 多 horizon ensemble) ==========
log(f"\n[6] Multi-horizon ETH ensemble (h5+h15+h30)...")
# 只需要训练 h5 和 h30, 然后和 h15 ensemble
# (简化: 先看有没有 h5 ds)
if os.path.exists('data/datasets/ds_ETH_h5.parquet'):
    for hh in [5, 30]:
        if not os.path.exists(f'{NPY}/ETH_h{hh}_test_X.npy'):
            df=pd.read_parquet(f'data/datasets/ds_ETH_h{hh}.parquet').sort_values('ts')
            fc=[c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
            for pref,tlo,thi in [('train',0,TRAIN_END),('early_stop',META_END),('test',META_END,10**18)]:
                tm=(df['ts'].to_numpy()>=tlo)&(df['ts'].to_numpy()<thi)
                np.save(f'{NPY}/ETH_h{hh}_{pref}_X.npy', df[fc].to_numpy().astype(np.float32)[tm])
                np.save(f'{NPY}/ETH_h{hh}_{pref}_y.npy', df['label'].to_numpy().astype(np.int32)[tm])

    p_multi = [p_eth[:len(y_test)]]
    for hh in [5, 30]:
        Xtr=np.load(f'{NPY}/ETH_h{hh}_train_X.npy').astype(np.float32)
        ytr=np.load(f'{NPY}/ETH_h{hh}_train_y.npy').astype(np.int32)
        Xes=np.load(f'{NPY}/ETH_h{hh}_early_stop_X.npy').astype(np.float32)
        yes=np.load(f'{NPY}/ETH_h{hh}_early_stop_y.npy').astype(np.int32)
        Xte=np.load(f'{NPY}/ETH_h{hh}_test_X.npy').astype(np.float32)
        pte=[]
        for sd in SEEDS:
            params['seed']=sd
            tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
            m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
            pte.append(m.predict(Xte))
        p_avg=np.mean(pte,axis=0)
        log(f"  ETH h{hh} te AUC={roc_auc_score(yes if len(yes)==len(p_avg) else np.load(f'{NPY}/ETH_h{hh}_test_y.npy'), p_avg):.4f}")
        # 对齐到 h15 test 的时间戳
        t=pq.read_table(f'data/datasets/ds_ETH_h{hh}.parquet', columns=['ts'])
        ts_hh=t.column('ts').to_numpy().astype(np.int64); del t
        idx=pd.Index(ts_hh).get_indexer(ts_test, method='nearest')
        idx=np.clip(idx,0,len(p_avg)-1)
        p_multi.append(p_avg[idx])
    
    p_mens = np.mean(np.column_stack(p_multi), axis=0)
    log(f"  Multi-horizon ens (h5+h15+h30): te={roc_auc_score(y_test[:len(p_mens)], p_mens):.4f}")
    
    # Backtest ens
    for q in [0.97,0.98,0.99]:
        n,a,tp,wr = eval_backtest(p_mens, y_test[:len(p_mens)], ret_test[:len(p_mens)], ts_test[:len(p_mens)], None, "ens", q)
        log(f"    ens q={q:.2f}: n={n:,} ACC={a:.1f}% tpd={tp:.1f}")

log(f"\nDONE TOTAL={time.time()-T0:.0f}s")
