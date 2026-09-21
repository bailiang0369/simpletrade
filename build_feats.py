import os
"""Build features once, save to npz - memory friendly"""
import warnings; warnings.filterwarnings('ignore')
import gc, datetime, numpy as np, pandas as pd, sys
sys.path.insert(0,'/workspace'); import config

print('loading raw...',flush=True)
raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
ts = raw_e['ts'].values.astype(np.int64)
close_e = raw_e['close'].values.astype(np.float64)
del raw_e; gc.collect()

raw_b = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
close_b = pd.Series(raw_b['close'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
buy_e = raw_e['buy_vol'].values.astype(np.float64) if False else pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet',columns=['buy_vol'])['buy_vol'].values.astype(np.float64)
sell_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet',columns=['sell_vol'])['sell_vol'].values.astype(np.float64)
fund_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet',columns=['funding'])['funding'].values.astype(np.float64)
buy_b = pd.Series(pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['buy_vol'])['buy_vol'].values,index=pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['ts'])['ts'].values).reindex(ts).values.astype(np.float64)
sell_b = pd.Series(pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['sell_vol'])['sell_vol'].values,index=pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['ts'])['ts'].values).reindex(ts).values.astype(np.float64)
fund_b = pd.Series(pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['funding'])['funding'].values,index=pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet',columns=['ts'])['ts'].values).reindex(ts).values.astype(np.float64)
del raw_b; gc.collect()
print('arrays ready',flush=True)

close_es = pd.Series(close_e); close_bs = pd.Series(close_b)
lr_e = close_es.pct_change(); lr_b = close_bs.pct_change()
hr = pd.to_datetime(ts,unit='s',utc=True).hour.values

feats = {}
# ETH returns - sparse but covering critical horizons
for w in [3,5,10,15,20,30,60,120,240,480,960]:
    feats[f'lr_e_{w}'] = close_es.pct_change(w).astype(np.float32).values
# BTC cross - fewer windows
for w in [5,15,30,60,120,240]:
    feats[f'lr_b_{w}'] = close_bs.pct_change(w).astype(np.float32).values
# Z-scores
for w in [15,30,60,120]:
    s=feats[f'lr_e_{w}']; ser=pd.Series(s)
    feats[f'z_e_{w}'] = ((ser-ser.rolling(w).mean())/(ser.rolling(w).std()+1e-9)).astype(np.float32).values
# Volatility
for w in [15,30,60,120,240]:
    feats[f'rvol_e_{w}'] = lr_e.rolling(w).std().astype(np.float32).values
    feats[f'rvol_b_{w}'] = lr_b.rolling(w).std().astype(np.float32).values
# CVD + Vol
for w in [15,30,60,120]:
    feats[f'cvd_e_{w}'] = pd.Series(buy_e-sell_e).rolling(w).mean().astype(np.float32).values
    feats[f'vol_e_{w}'] = pd.Series(buy_e+sell_e).rolling(w).mean().astype(np.float32).values
# Funding
for w in [5,30,60,120]:
    feats[f'fund_e_{w}'] = pd.Series(fund_e).rolling(w).mean().astype(np.float32).values
    feats[f'fund_b_{w}'] = pd.Series(fund_b).rolling(w).mean().astype(np.float32).values
# Skewness
for w in [60,240]:
    feats[f'skew_{w}'] = lr_e.rolling(w).skew().astype(np.float32).values
# Ratio
ratio = close_e/(close_b+1e-12)
for w in [15,30,60]:
    feats[f'ratio_lr_{w}'] = pd.Series(ratio).pct_change(w).astype(np.float32).values
# Position in range
for w in [60,240]:
    hi=pd.Series(close_e).rolling(w).max(); lo=pd.Series(close_e).rolling(w).min()
    feats[f'pos_{w}'] = ((close_e-lo)/(hi-lo+1e-12)).astype(np.float32).values
# Hour cyclical + interactions
feats['hour_sin']=np.sin(2*np.pi*hr/24).astype(np.float32)
feats['hour_cos']=np.cos(2*np.pi*hr/24).astype(np.float32)
feats['vol_x_hour_sin']=(feats['rvol_e_60']*feats['hour_sin']).astype(np.float32)
feats['vol_x_hour_cos']=(feats['rvol_e_60']*feats['hour_cos']).astype(np.float32)
# Cross interactions
feats['vol_x_mom30']=(feats['rvol_e_60']*feats['lr_e_30']).astype(np.float32)
feats['fund_x_vol']=(feats['fund_e_30']*feats['rvol_e_60']).astype(np.float32)
feats['cvd_x_vol']=(feats['cvd_e_60']*feats['rvol_e_60']).astype(np.float32)
# streaks
ls=np.sign(lr_e.values); us=np.zeros(len(ts),np.int16); ds=np.zeros(len(ts),np.int16)
for i in range(1,len(ts)):
    us[i]=us[i-1]+1 if ls[i]>0 else 0
    ds[i]=ds[i-1]+1 if ls[i]<0 else 0
feats['streak_diff']=(us-ds).astype(np.float32)
del us,ds,ls,buy_e,sell_e,fund_e,buy_b,sell_b,fund_b,ratio; gc.collect()

FEAT_NAMES=sorted(feats.keys())
X_all = np.stack([feats[f] for f in FEAT_NAMES],axis=1).astype(np.float32)
del feats, lr_e, lr_b, close_e, close_b, close_es, close_bs; gc.collect()
print(f'X_all: {X_all.shape} mem={X_all.nbytes/1e6:.0f}MB feats={len(FEAT_NAMES)}',flush=True)

# Build labels
labels={}
for h in [3,5,15,30,60]:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_e.values[h:]/close_e.values[:-h]-1).astype(np.float32) if False else None  # need close_e
# oh we deleted close_e. Reload quickly
close_e2 = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet',columns=['close'])['close'].values.astype(np.float64)
for h in [3,5,15,30,60]:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_e2[h:]/close_e2[:-h]-1).astype(np.float32)
    labels[f'ret_{h}']=rh
    labels[f'y_{h}']=(rh>0).astype(np.int8)
del close_e2; gc.collect()

# Time splits
def ts_mask(s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_m = ts_mask('2020-01-01','2024-06-30')
es_m = ts_mask('2024-06-30','2024-09-30')
te_m = ts_mask('2025-09-30','2026-08-29')

# Clean NaN
vm = ~np.isnan(labels['ret_30']) & ~np.isnan(X_all).any(axis=1)
vm = vm & np.isfinite(X_all).all(axis=1)
vi = np.where(vm)[0]
X = X_all[vi]; del X_all; gc.collect()
labels_clean = {}
for k,v in labels.items(): labels_clean[k]=v[vi]
del labels; gc.collect()

print(f'Saving: X={X.shape} vi={vi.shape} ts={ts.shape} hr={hr.shape}',flush=True)
os.makedirs('/workspace/models',exist_ok=True)
np.savez('/workspace/models/eth_data.npz', X=X, vi=vi, ts=ts, hr=hr,
         ret_3=labels_clean['ret_3'],ret_5=labels_clean['ret_5'],ret_15=labels_clean['ret_15'],ret_30=labels_clean['ret_30'],ret_60=labels_clean['ret_60'],
         y_3=labels_clean['y_3'],y_5=labels_clean['y_5'],y_15=labels_clean['y_15'],y_30=labels_clean['y_30'],y_60=labels_clean['y_60'],
         feat_names=np.array(FEAT_NAMES),
         tr_mask=tr_m[vi], es_mask=es_m[vi], te_mask=te_m[vi])
print('saved /workspace/models/eth_data.npz',flush=True)
