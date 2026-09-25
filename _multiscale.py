"""加多周期 + 量价背离特征, 训练 ETH, 看能不能 AUC 破 0.546
手动交易看的是: 4H 方向 + 15min 形态 + 量价配合
"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import polars as pl
import pyarrow.parquet as pq
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]; TRAIN_END=1722556800; META_END=1759363200
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Load base ds feats (56)
df_tr=pd.read_parquet(f'data/datasets/ds_ETH_h15.parquet').sort_values('ts')
fc=[c for c in df_tr.columns if c not in ('ts','label','ret_future','soft_label')]
X_base=df_tr[fc].to_numpy().astype(np.float32)
y=df_tr['label'].to_numpy().astype(np.int32)
ret=df_tr['ret_future'].to_numpy().astype(np.float32)
ts=df_tr['ts'].to_numpy().astype(np.int64)
del df_tr; gc.collect()

# Load raw, compute multi-scale feats
log("[1] Build multi-scale features...")
raw=pl.read_parquet('data/datasets/raw_ETH.parquet').sort('ts')
close=raw['close'].to_numpy().astype(np.float64)
high=raw['high'].to_numpy().astype(np.float64)
low=raw['low'].to_numpy().astype(np.float64)
bv=raw['buy_vol'].to_numpy().astype(np.float64)
sv=raw['sell_vol'].to_numpy().astype(np.float64)
rts=raw['ts'].to_numpy().astype(np.int64); del raw; gc.collect()

feats={}

# --- 多周期均线 (4H=96根, 12H=288根, 1D=960根) ---
for w, name in [(96,'4h'), (288,'12h'), (960,'1d')]:
    ma=pd.Series(close).rolling(w,min_periods=w//3).mean().to_numpy()
    feats[f'close/ma_{name}'] = np.where(ma>1e-9, close/ma-1, 0).astype(np.float32)
    feats[f'ma_slope_{name}'] = (ma - np.roll(ma, w//4)) / np.where(ma>1e-9, ma, 1e-9) / (w//4) * 1000000
    feats[f'price_pos_{name}'] = ((close - pd.Series(close).rolling(w,min_periods=w//3).min().to_numpy()) /
                                  np.maximum(pd.Series(close).rolling(w,min_periods=w//3).max().to_numpy() - 
                                             pd.Series(close).rolling(w,min_periods=w//3).min().to_numpy(), 1e-9)).astype(np.float32)

# --- 量价背离 (price new high but vol not new high = bearish divergence) ---
for pw in [60, 240, 960]:  # price window
    vw = pw
    p_high = pd.Series(close).rolling(pw,min_periods=pw//3).max().to_numpy()
    p_low = pd.Series(close).rolling(pw,min_periods=pw//3).min().to_numpy()
    vol_sum = pd.Series(bv+sv).rolling(vw,min_periods=vw//3).sum().to_numpy()
    vol_high = pd.Series(bv+sv).rolling(vw,min_periods=vw//3).max().to_numpy()
    
    # Price at window high but vol below median → bearish
    at_ph = (close >= p_high * 0.999).astype(np.float64)
    vol_med = pd.Series(bv+sv).rolling(vw,min_periods=vw//3).median().to_numpy()
    feats[f'bear_div_{pw}'] = ((at_ph.astype(bool)) & ((bv+sv) < vol_med)).astype(np.float32)
    feats[f'bull_div_{pw}'] = (((close <= p_low * 1.001)).astype(bool)) & ((bv+sv) < vol_med)).astype(np.float32)

# --- 结构特征: bar 连续性 (连续 N 根同向) ---
def consec(arr, sign):
    out=np.zeros(len(arr),dtype=np.float32)
    c=0
    for i in range(len(arr)):
        if sign * arr[i] > 0: c+=1
        else: c=0
        out[i]=c
    return out

bar_ret=np.diff(close)/close[:-1]; bar_ret=np.concatenate([[0],bar_ret])
feats['up_streak'] = consec(bar_ret, 1).astype(np.float32)
feats['dn_streak'] = consec(bar_ret, -1).astype(np.float32)

# --- ADX/trend strength (60min + 240min) ---
def adx(c,h,l,w):
    up=np.zeros(len(c)); up[1:]=h[1:]-h[:-1]
    dn=np.zeros(len(c)); dn[1:]=l[:-1]-l[1:]
    dmp=np.where((up>dn)&(up>0),up,0); dmn=np.where((dn>up)&(dn>0),dn,0)
    tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    di_p=pd.Series(dmp).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    di_m=pd.Series(dmn).rolling(w,min_periods=10).mean().to_numpy()/np.maximum(pd.Series(tr).rolling(w,min_periods=10).mean().to_numpy(),1e-9)
    dx=np.where((di_p+di_m)>1e-9, np.abs(di_p-di_m)/(di_p+di_m)*100, 0)
    return pd.Series(dx).rolling(w,min_periods=10).mean().to_numpy()

feats['adx_60'] = adx(close,high,low,60).astype(np.float32)
feats['adx_240'] = adx(close,high,low,240).astype(np.float32)

# --- 波动率 ratio (recent vol / longer vol) ---
ret_all=np.diff(close)/close[:-1]; ret_all=np.concatenate([[0],ret_all])
vol60=pd.Series(ret_all).rolling(60,min_periods=10).std().to_numpy()
vol960=pd.Series(ret_all).rolling(960,min_periods=60).std().to_numpy()
feats['vol_ratio'] = np.where(vol960>1e-9, vol60/vol960, 1.0).astype(np.float32)

# --- 成交密集度 (buy_vol / total_vol zscore) ---
bv_ratio = np.where(bv+sv>0, bv/(bv+sv), 0.5)
feats['bv_ratio_ma60'] = pd.Series(bv_ratio).rolling(60,min_periods=10).mean().to_numpy().astype(np.float32)
feats['bv_ratio_std60'] = pd.Series(bv_ratio).rolling(60,min_periods=10).std().to_numpy().astype(np.float32)

del close,high,low,bv,sv,rts; gc.collect()

keys=sorted(feats.keys()); n_new=len(keys); log(f"  New feats: {n_new}")
new_feat_arr=np.stack([feats[k] for k in keys],axis=1).astype(np.float32); del feats; gc.collect()

# Align to ds timestamps
idx=np.searchsorted(pq.read_table('data/datasets/raw_ETH.parquet', columns=['ts']).column('ts').to_numpy().astype(np.int64) if False else None, ts)
# 直接用同一个 raw ts
raw_ts = pl.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts'])['ts'].to_numpy().astype(np.int64)
del pl; gc.collect()
raw_ts = pl.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts'])['ts'].to_numpy().astype(np.int64)
idx=np.searchsorted(raw_ts, ts); idx=np.clip(idx, 0, len(new_feat_arr)-1)
X_new = new_feat_arr[idx]
log(f"  Aligned: {X_new.shape}")
del new_feat_arr, raw_ts; gc.collect()

# Combine with base (drop nan)
X_all = np.hstack([X_base, X_new])
nan_mask = ~np.isnan(X_all).any(axis=1)
log(f"  Combined: {X_all.shape}, valid={nan_mask.sum():,} ({nan_mask.mean()*100:.1f}%)")
X_all = X_all[nan_mask]; y=y[nan_mask]; ret=ret[nan_mask]; ts=ts[nan_mask]
del X_base, X_new; gc.collect()

# Split
def split(X,y,ts):
    def get(m): return X[m], y[m]
    tr=(ts>=0)&(ts<TRAIN_END); es=(ts>=TRAIN_END)&(ts<1725148800); te=ts>=META_END
    return get(tr), get(es), get(te)

(Xtr,ytr),(Xes,yes),(Xte,yte) = split(X_all,y,ts)
log(f"  tr={Xtr.shape[0]:,} es={Xes.shape[0]:,} te={Xte.shape[0]:,} total feats={X_all.shape[1]}")

ret_tr = ret[(ts>=0)&(ts<TRAIN_END)][:Xtr.shape[0]]
sw = np.where(np.abs(ret_tr)>=np.quantile(np.abs(ret_tr),0.90),0.3,1.0).astype(np.float32); del ret_tr, ret, ts, nan_mask; gc.collect()

# Train 5-seed
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,pte,sw; gc.collect()
te_auc=roc_auc_score(yte,p)
log(f"\n  ★ TEST AUC = {te_auc:.4f}  (base=0.5428, +fund=0.5440) ({time.time()-t0:.0f}s)")

# Quick tpd sweep
days=(META_END+len(yte)*15*60 - META_END)/SPD  # 近似
log(f"\n  tp=14.4 时 ACC (global q, upper bound):")
for q in np.arange(0.989, 0.9955, 0.0005):
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm
    n=tm.sum(); tpd=n/332  # 已知 332 天
    if n<30: continue
    ss=sm[tm]; acc=(((~ss)&(yte[tm]==1))|(ss&(yte[tm]==0))).mean()*100
    if 12<=tpd<=18:
        log(f"    q={q:.4f} tpd={tpd:5.1f} ACC={acc:5.1f}%")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
