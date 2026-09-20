"""Add BTC cross-asset features — clean rewrite."""
import numpy as np, pandas as pd, gc

print('[1] load...')
eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
btc = pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')
ts_eth = eth['ts'].values.astype(np.int64)

# Align BTC to ETH
btc_idx = pd.Index(btc['ts'].values)
eth_idx = pd.Index(ts_eth)
reorder = btc_idx.get_indexer(eth_idx)
reorder = np.where(reorder < 0, np.arange(len(reorder)), reorder)

N = len(eth)
btc_c = btc['close'].values.astype(np.float64)[reorder]
btc_h = btc['high'].values.astype(np.float64)[reorder]
btc_l = btc['low'].values.astype(np.float64)[reorder]
btc_o = btc['open'].values.astype(np.float64)[reorder]
eth_c = eth['close'].values.astype(np.float64)

print(f'  aligned N={N:,}')

print('\n[2] build feats...')
F = {}
# BTC returns
for h in [1,3,5,10,15,30,60,120,240]:
    r = np.full(N, np.nan, np.float32)
    r[h:] = (btc_c[h:]/btc_c[:-h] - 1).astype(np.float32)
    F[f'btc_ret_{h}'] = r
# BTC vol
br1 = np.full(N, np.nan, np.float64)
br1[1:] = btc_c[1:]/btc_c[:-1] - 1
for w in [5,15,30,60,120]:
    s = pd.Series(br1).rolling(w, min_periods=w).std().values
    F[f'btc_vol_{w}'] = s.astype(np.float32)
# BTC range/body
F['btc_range'] = ((btc_h - btc_l) / np.maximum(btc_c, 1e-9)).astype(np.float32)
F['btc_body'] = ((btc_c - btc_o) / np.maximum(btc_c, 1e-9)).astype(np.float32)
# BTC streak
sign = np.sign(br1)
st = np.zeros(N, np.float32)
for i in range(2, N):
    if abs(st[i-1]) > 0 and sign[i-1] == st[i-1] / abs(st[i-1]):
        st[i] = st[i-1] + sign[i-1]
    else:
        st[i] = sign[i-1]
F['btc_streak'] = st
# Rolling corr ETH/BTC returns
er1 = np.full(N, np.nan, np.float64)
er1[1:] = eth_c[1:]/eth_c[:-1] - 1
for w in [15, 30, 60, 120]:
    # Vectorized rolling corr
    es = pd.Series(er1); bs = pd.Series(br1); e_roll = es.rolling(w, min_periods=w)
    b_roll = bs.rolling(w, min_periods=w)
    cov = e_roll.cov(bs)
    ve = e_roll.var(); vb = b_roll.var()
    corr = (cov / np.sqrt(np.maximum(ve * vb, 1e-12))).values.astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    F[f'corr_{w}'] = corr
# ETH-BTC return spread
for h in [5,15,30,60]:
    er = np.full(N, np.nan, np.float64)
    er[h:] = eth_c[h:]/eth_c[:-h] - 1
    F[f'spread_{h}'] = (er - F[f'btc_ret_{h}'].astype(np.float64)).astype(np.float32)

print(f'  total feats: {len(F)}')
X_new = np.column_stack(list(F.values()))
names = list(F.keys())
del F; gc.collect()

# NaN → 0
print(f'  NaN before replace: {np.isnan(X_new).sum()}')
X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
print(f'  X_new={X_new.shape} mem={X_new.nbytes/1e9:.2f}GB')

# Save BTC feats
np.savez('/workspace/models/eth_btcfeats.npz', X=X_new, feat_names=np.array(names))
print('saved eth_btcfeats.npz')
del X_new; gc.collect()

print('\n[3] merge with existing...')
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X_old = D['X'].astype(np.float32)
old_names = list(D['feat_names'])
vi = D['vi'].astype(np.int64)

# Load BTC feats and align to vi
X_btc = np.load('/workspace/models/eth_btcfeats.npz', allow_pickle=True)['X'].astype(np.float32)
btc_names = list(np.load('/workspace/models/eth_btcfeats.npz', allow_pickle=True)['feat_names'])
X_btc_vi = X_btc[vi]
print(f'  X_old={X_old.shape}  X_btc_vi={X_btc_vi.shape}')

X_combo = np.column_stack([X_old, X_btc_vi])
print(f'  X_combo={X_combo.shape}  ({X_combo.shape[1]} feats)  mem={X_combo.nbytes/1e9:.2f}GB')

import os
out = '/workspace/models/eth_full.npz'
np.savez(out, X=X_combo, vi=D['vi'], ts=D['ts'],
         feat_names=np.array(old_names + btc_names),
         ret_3=D['ret_3'], ret_5=D['ret_5'], ret_15=D['ret_15'], ret_30=D['ret_30'], ret_60=D['ret_60'],
         y_3=D['y_3'], y_5=D['y_5'], y_15=D['y_15'], y_30=D['y_30'], y_60=D['y_60'])
print(f'  saved {out}  size={os.path.getsize(out)/1e9:.2f}GB')
del X_combo, X_old, X_btc, X_btc_vi; gc.collect()
print('DONE!')
