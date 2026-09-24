import numpy as np, pandas as pd, gc, os
import polars as pl
PAR='data/datasets'; NPY='data/splits_npy'
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

print("[1] Load raw ETH/BTC...", flush=True)
eth_pl = pl.read_parquet(f'{PAR}/raw_ETH.parquet').sort('ts')
btc_pl = pl.read_parquet(f'{PAR}/raw_BTC.parquet').sort('ts')
eth = eth_pl.to_pandas(); btc_ts = btc_pl['ts'].to_numpy().astype(np.int64)
del eth_pl, btc_pl; gc.collect()

# ETH cross feats
lc_e = np.log(eth['close'].to_numpy())
cf = {}
for k in (5,15,30,60,120,240,480,960):
    r=np.full(len(lc_e),np.nan); r[k:]=lc_e[k:]-lc_e[:-k]; cf[f'ETH_lr_{k}']=r.astype(np.float32)
sr=pd.Series(lc_e)
for w in (30,60,120,240,480):
    mu=sr.rolling(w).mean().to_numpy(); sd=sr.rolling(w).std().to_numpy()
    cf[f'ETH_z_{w}']=np.where(sd>1e-9,(lc_e-mu)/sd,0.0).astype(np.float32)
lr1=np.full(len(lc_e),np.nan); lr1[1:]=lc_e[1:]-lc_e[:-1]
for w in (60,240):
    cf[f'ETH_rvol_{w}']=pd.Series(lr1).rolling(w).std().to_numpy().astype(np.float32)*100
ob_e=eth['buy_vol'].to_numpy(); os_e=eth['sell_vol'].to_numpy()
d=pd.Series(ob_e-os_e); tt=pd.Series(ob_e+os_e)
for w in (30,60):
    cf[f'ETH_cvd_{w}']=(d.rolling(w).sum()/(tt.rolling(w).sum()+1e-12)).to_numpy().astype(np.float32)
print(f"[2] Built {len(cf)} ETH cross feats, RAM→", flush=True)
del lc_e, sr, d, tt, ob_e, os_e, lr1; gc.collect()

# Align
eth_ts=eth['ts'].to_numpy().astype(np.int64); del eth
idx=np.searchsorted(eth_ts,btc_ts,side='right')-1; idx=np.clip(idx,0,len(eth_ts)-1)
del eth_ts; gc.collect()
print(f"[3] Aligned, loading ds...", flush=True)
aligned={k:v[idx] for k,v in cf.items()}; del cf, idx; gc.collect()

# Stream from ds parquet
import pyarrow.parquet as pq
t = pq.read_table(f'{PAR}/ds_BTC_h15.parquet')
cols_all = t.column_names
feat_cols = [c for c in cols_all if c not in ('ts','label','ret_future','soft_label')]
print(f"[4] ds has {len(feat_cols)} base feats + {len(aligned)} cross = {len(feat_cols)+len(aligned)}", flush=True)

ts_ds = t.column('ts').to_numpy().astype(np.int64)
idx2 = np.searchsorted(btc_ts, ts_ds, side='right') - 1
idx2 = np.clip(idx2, 0, len(btc_ts)-1)
del btc_ts; gc.collect()

# Pre-add cross feats to ds table
for fname, arr in aligned.items():
    t = t.append_column(pl.lit(arr[idx2]).alias(fname))
del aligned; gc.collect()

# Null-drop using polars
import polars as pl
feat_all = feat_cols + list(pq.read_schema(f'{PAR}/ds_BTC_h15.parquet').names[-1:])  # approx
# Actually better: do it in polars
pdf = t.to_pandas()  # will be big but needed
print(f"[5] Converted to pandas {pdf.shape}", flush=True)
all_f = [c for c in pdf.columns if c not in ('ts','label','ret_future','soft_label')]
mask = pdf[all_f].notna().all(axis=1).to_numpy()
pdf = pdf[mask].reset_index(drop=True)
del mask; gc.collect()
print(f"  After null drop: {len(pdf):,}", flush=True)

ts = pdf['ts'].to_numpy().astype(np.int64)
y = pdf['label'].to_numpy().astype(np.int32)
ret = pdf['ret_future'].to_numpy().astype(np.float32)

del pdf, t; gc.collect()

# Save one split at a time
for split,tlo,thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),
                       ('meta_val',ES_END,META_END),('test',META_END,10**18)]:
    m=(ts>=tlo)&(ts<thi)
    print(f"[6] Saving {split} ({m.sum():,})...", flush=True)
    # Read ds again to get features one chunk
    X_chunk = np.load(f'{NPY}/BTC_h15_train_X.npy')[:0]  # placeholder - wrong approach
    # Actually load from parquet with filter
    sub_ts = ts[m]
    del m
    gc.collect()
    # Build X by slicing pandas... too much RAM
    # Use streaming approach instead: load all feats once from pdf chunk before
    print(f"  → need rewrite")
    break
print("ABORT - need different approach")
