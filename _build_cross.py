import numpy as np, pandas as pd, gc, os
import polars as pl
PAR='data/datasets'; NPY='data/splits_npy'
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

print("[1] Load raw ETH/BTC...", flush=True)
eth = pl.read_parquet(f'{PAR}/raw_ETH.parquet').sort('ts').to_pandas()
btc = pl.read_parquet(f'{PAR}/raw_BTC.parquet').sort('ts').to_pandas()
print(f"  ETH={len(eth)} BTC={len(btc)}", flush=True)

# Build ETH cross features
lc_e = np.log(eth['close'].to_numpy())
ob_e = eth['buy_vol'].to_numpy().astype(np.float64)
os_e = eth['sell_vol'].to_numpy().astype(np.float64)

cf = {}
for k in (5,15,30,60,120,240,480,960):
    r=np.full(len(lc_e),np.nan); r[k:]=lc_e[k:]-lc_e[:-k]
    cf[f'ETH_lr_{k}']=r.astype(np.float32)
sr=pd.Series(lc_e)
for w in (30,60,120,240,480):
    mu=sr.rolling(w).mean().to_numpy(); sd=sr.rolling(w).std().to_numpy()
    cf[f'ETH_z_{w}']=np.where(sd>1e-9,(lc_e-mu)/sd,0.0).astype(np.float32)
lr1=np.full(len(lc_e),np.nan); lr1[1:]=lc_e[1:]-lc_e[:-1]
for w in (60,240):
    cf[f'ETH_rvol_{w}']=pd.Series(lr1).rolling(w).std().to_numpy().astype(np.float32)*100
d=pd.Series(ob_e-os_e); tt=pd.Series(ob_e+os_e)
for w in (30,60):
    cf[f'ETH_cvd_{w}']=(d.rolling(w).sum()/(tt.rolling(w).sum()+1e-12)).to_numpy().astype(np.float32)
print(f"[2] Built {len(cf)} ETH cross feats", flush=True)
del eth, lc_e, ob_e, os_e, sr, d, tt, lr1; gc.collect()

# Align ETH→BTC
eth_ts = pl.read_parquet(f'{PAR}/raw_ETH.parquet',columns=['ts']).sort('ts').to_numpy().flatten()
btc_ts = btc['ts'].to_numpy().astype(np.int64)
idx = np.searchsorted(eth_ts, btc_ts, side='right') - 1
idx = np.clip(idx, 0, len(eth_ts)-1)
del eth_ts, btc; gc.collect()
aligned = {k:v[idx] for k,v in cf.items()}
del cf; gc.collect()

# Load BTC ds
print("[3] Load BTC ds + merge cross...", flush=True)
df = pd.read_parquet(f'{PAR}/ds_BTC_h15.parquet').sort_values('ts').reset_index(drop=True)
ts_ds = df['ts'].to_numpy().astype(np.int64)
idx2 = np.searchsorted(btc_ts, ts_ds, side='right') - 1
idx2 = np.clip(idx2, 0, len(btc_ts)-1)
del btc_ts; gc.collect()

for fname, arr in aligned.items():
    df[fname] = arr[idx2]
del aligned; gc.collect()

all_feats = [c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
null_mask = df[all_feats].isnull().any(axis=1).to_numpy()
df = df[~null_mask].reset_index(drop=True)
print(f"  After null drop: {len(df):,} rows, {len(all_feats)} feats", flush=True)

# Save
print("[4] Save splits...", flush=True)
ts_new = df['ts'].to_numpy().astype(np.int64)
y_new = df['label'].to_numpy().astype(np.int32)
X_new = df[all_feats].to_numpy().astype(np.float32)
ret_new = df['ret_future'].to_numpy().astype(np.float32)
del df; gc.collect()

for split,tlo,thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),
                       ('meta_val',ES_END,META_END),('test',META_END,10**18)]:
    m=(ts_new>=tlo)&(ts_new<thi)
    np.save(f'{NPY}/BTC_h15_cross_{split}_X.npy',X_new[m])
    np.save(f'{NPY}/BTC_h15_cross_{split}_y.npy',y_new[m])
    np.save(f'{NPY}/BTC_h15_cross_{split}_ret.npy',ret_new[m])
    print(f"  cross {split}: {m.sum():,}", flush=True)
print("DONE ✓", flush=True)
