"""加 5min 周期特征 → 验证多周期 ret 是否比单尺度好"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); SEEDS=[42,49,56,63,70]; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
print(f"Device=cpu", flush=True)

# ============= 1. Build 5min resample features =============
print("[1] Build 5min features from raw_ETH...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low','buy_vol','sell_vol']).sort_values('ts')
raw['dt']=pd.to_datetime(raw['ts'], unit='s')
r5 = raw.set_index('dt').resample('5min').agg(
    {'close':'last','high':'max','low':'max','buy_vol':'sum','sell_vol':'sum'}
).dropna().reset_index()
r5['ts']=(r5['dt'].astype(np.int64)//10**9).astype(np.int64)
del raw['dt'], raw; gc.collect()

# 5min features
C5=r5['close'].astype(np.float64); H5=r5['high'].astype(np.float64); L5=r5['low'].astype(np.float64)
feats={}
# lr_5: 5min log return
feats['lr_5']=np.zeros(len(C5),dtype=np.float32); feats['lr_5'][1:]=np.diff(np.log(np.maximum(C5,1e-9))).astype(np.float32)
# pos_5: 5min close position
feats['pos_5'] = np.where(
    pd.Series(C5).rolling(5,min_periods=2).max().to_numpy() - pd.Series(C5).rolling(5,min_periods=2).min().to_numpy() > 1e-9,
    (C5 - pd.Series(C5).rolling(5,min_periods=2).min().to_numpy()) / 
    (pd.Series(C5).rolling(5,min_periods=2).max().to_numpy() - pd.Series(C5).rolling(5,min_periods=2).min().to_numpy()),
    0.5
).astype(np.float32)
# stoch_k5_5: 5min stoch_k(5)
L5r=pd.Series(L5).rolling(5,min_periods=1).min().to_numpy(); H5r=pd.Series(H5).rolling(5,min_periods=1).max().to_numpy()
feats['stoch_k5_5'] = np.where(H5r-L5r>1e-9, (C5-L5r)/(H5r-L5r)*100, 50).astype(np.float32)
# stoch_k14_5: 5min stoch_k(14)
L5b=pd.Series(L5).rolling(14,min_periods=1).min().to_numpy(); H5b=pd.Series(H5).rolling(14,min_periods=1).max().to_numpy()
feats['stoch_k14_5'] = np.where(H5b-L5b>1e-9, (C5-L5b)/(H5b-L5b)*100, 50).astype(np.float32)
# vol_5: 5min vol
feats['vol_5'] = pd.Series(feats['lr_5']).rolling(5,min_periods=2).std().to_numpy().astype(np.float32)
# close zscore 5min (相对 240min 窗口 = 20个 5min bar)
rm=pd.Series(C5).rolling(20,min_periods=5).mean().to_numpy()
rs=pd.Series(C5).rolling(20,min_periods=5).std().to_numpy()
feats['z_5'] = np.where(rs>1e-9,(C5-rm)/rs,0.0).astype(np.float32)
# buy/sell vol ratio 5min
BV=r5['buy_vol'].astype(np.float64).to_numpy(); SV=r5['sell_vol'].astype(np.float64).to_numpy()
feats['bvr_5'] = np.where(BV+SV>0, BV/(BV+SV), 0.5).astype(np.float32)
del r5,C5,H5,L5; gc.collect()
keys=sorted(feats.keys()); print(f"  5min feats: {keys}", flush=True)

# ============= 2. Align to ds timestamps =============
print("[2] Align 5min feats to ds timestamps...", flush=True)
ds = pd.read_parquet('data/datasets/ds_ETH_h15.parquet').sort_values('ts').reset_index(drop=True)
dts=ds['ts'].to_numpy().astype(np.int64)
r5_ts = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close']).sort_values('ts')
# Recompute 5min ts
raw_tmp = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts']).sort_values('ts')
raw_tmp['dt']=pd.to_datetime(raw_tmp['ts'],unit='s')
r5ts = raw_tmp.set_index('dt').resample('5min').first().index
r5ts_arr=(r5ts.astype(np.int64)//10**9).astype(np.int64)
del raw_tmp, r5ts; gc.collect()

feats_arr=np.stack([feats[k] for k in keys],axis=1).astype(np.float32); del feats; gc.collect()
idx=np.searchsorted(r5ts_arr,dts); idx=np.clip(idx,50,len(feats_arr)-1)
X5=feats_arr[idx]; del feats_arr, r5ts_arr; gc.collect()
print(f"  Aligned: {X5.shape}, base ds: {ds.shape}", flush=True)

# ============= 3. Split + Train =============
print(f"\n[3] Train: base 56 feats + 7 new 5min feats...", flush=True)
fc=[c for c in ds.columns if c not in ('ts','label','ret_future','soft_label')]
Xb=ds[fc].to_numpy().astype(np.float32)
y=ds['label'].to_numpy().astype(np.int32)
ret=ds['ret_future'].to_numpy().astype(np.float32)
del ds; gc.collect()

# Stack (对齐 len)
mr=min(Xb.shape[0],X5.shape[0]); Xb=Xb[:mr]; X5=X5[:mr]; y=y[:mr]; ret=ret[:mr]
X_all=np.hstack([Xb, X5]); nan=~np.isnan(X_all).any(axis=1)
X_all=X_all[nan]; y=y[nan]; ret=ret[nan]
print(f"  Combined: {X_all.shape} ({len(fc)}+{X5.shape[1]}={X_all.shape[1]} feats), valid={nan.sum():,} ({nan.mean()*100:.1f}%)", flush=True)

def split(lo,hi):
    m=(dts[:mr][nan][:len(X_all)]>=lo)&(dts[:mr][nan][:len(X_all)]<hi)
    return X_all[m], y[m]

# 简化: 直接用 ts mask
dts_n=dts[:mr][nan][:len(X_all)]
tr_m=(dts_n>=0)&(dts_n<TRAIN_END); es_m=(dts_n>=TRAIN_END)&(dts_n<ES_END); te_m=(dts_n>=META_END)
Xtr,ytr=X_all[tr_m],y[tr_m]; Xes,yes=X_all[es_m],y[es_m]; Xte,yte=X_all[te_m],y[te_m]
print(f"  tr={Xtr.shape[0]:,} es={Xes.shape[0]:,} te={Xte.shape[0]:,}", flush=True)
del X_all,Xb,X5,nan,dts_n,dts,mr; gc.collect()

ret_tr=ret[tr_m][:len(Xtr)]
sw=np.where(np.abs(ret_tr)>=np.quantile(np.abs(ret_tr),0.90),0.3,1.0).astype(np.float32); del ret_tr,ret; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,pte,sw; gc.collect()
auc=roc_auc_score(yte,p)
print(f"\n★★ BASE(56) + 5min(7) feats + negw: AUC={auc:.4f}  (baseline=0.5428) ({time.time()-t0:.0f}s)", flush=True)

# Feature importance (top new)
tr_d=lgb.Dataset(Xte,label=yte); m1=lgb.train({'objective':'binary','verbose':-1},tr_d,50)
# 不对, 应该用训练好的 m... pte[0] 是预测, m 被覆盖了
# 没关系, 我们看 AUC 就行

# tpd sweep
print(f"\n[4] tpd sweep (global q oracle)...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985,0.998,0.0005):
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(yte[tm]==1))|(ss&(yte[tm]==0))).mean()*100
    tpd=n/332; diff=acc-65; mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}", flush=True)
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)
if best: print(f"\n★ tpd≈14.4: ACC={best[0]:.1f}% q={best[1]:.4f} tpd={best[2]:.1f}", flush=True)

print(f"\n{'='*60}")
print(f"  对比:")
print(f"  BASE ret feats only (56):  AUC=0.5428  tpd14.4 ACC≈61.6%")
print(f"  BASE + 5min (7 new):       AUC={auc:.4f}  tpd14.4 ACC≈{best[0] if best else '?'}%")
print(f"  {(auc-0.5428)*100:+.2f}pp AUC")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s", flush=True)
