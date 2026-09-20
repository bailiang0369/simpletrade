"""Build multi-scale feats, save to disk."""
import numpy as np, pandas as pd, gc
eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
ts_1min = eth['ts'].values.astype(np.int64)
eth['ts'] = pd.to_datetime(eth['ts'], unit='s', utc=True)
eth = eth.set_index('ts')

def resample_ohlcv(df, mins):
    g = df.resample(f'{mins}min', label='right', closed='right')
    return g.agg({'open':'first','high':'max','low':'min','close':'last','buy_vol':'sum','sell_vol':'sum','funding':'last'}).dropna()

b5 = resample_ohlcv(eth, 5)
b15 = resample_ohlcv(eth, 15)
print(f'5min={len(b5):,}  15min={len(b15):,}')

def bar_feats(bars):
    c = bars['close'].values.astype(np.float32)
    h = bars['high'].values.astype(np.float32)
    l = bars['low'].values.astype(np.float32)
    o = bars['open'].values.astype(np.float32)
    bv = bars['buy_vol'].values.astype(np.float32)
    sv = bars['sell_vol'].values.astype(np.float32)
    fnd = bars['funding'].values.astype(np.float32)
    feats = {}
    for w in [1,2,3,5,10,20]:
        feats[f'ret_{w}'] = np.full(len(c), np.nan, np.float32); feats[f'ret_{w}'][w:] = c[w:]/c[:-w] - 1
    feats['range'] = (h - l) / np.maximum(c, 1e-9)
    feats['body'] = (c - o) / np.maximum(c, 1e-9)
    feats['ushadow'] = (h - np.maximum(o, c)) / np.maximum(c, 1e-9)
    feats['lshadow'] = (np.minimum(o, c) - l) / np.maximum(c, 1e-9)
    ret1 = c[1:]/c[:-1] - 1
    for w in [5,10,20,40]:
        s = pd.Series(ret1).rolling(w, min_periods=w).std().values
        feats[f'vol_{w}'] = np.full(len(c), np.nan, np.float32); feats[f'vol_{w}'][1:] = s.astype(np.float32)
    tot = bv + sv
    feats['bvr'] = bv / np.maximum(tot, 1e-9)
    feats['bvr_fast'] = pd.Series(feats['bvr']).rolling(5,min_periods=5).mean().values.astype(np.float32)
    feats['bvr_slow'] = pd.Series(feats['bvr']).rolling(20,min_periods=20).mean().values.astype(np.float32)
    for w in [5,10,20]:
        feats[f'cvd_{w}'] = pd.Series(bv-sv).rolling(w,min_periods=w).sum().values.astype(np.float32)
    for w in [1,5,10,30]:
        feats[f'fund_{w}'] = pd.Series(fnd).rolling(w,min_periods=1).mean().values.astype(np.float32)
    return pd.DataFrame(feats, index=bars.index)

f5 = bar_feats(b5); f15 = bar_feats(b15)
print(f'5min feats={f5.shape}  15min feats={f15.shape}')

# Align
ts_idx = eth.index
n5 = f5.reindex(ts_idx, method='ffill').bfill().values.astype(np.float32)
n15 = f15.reindex(ts_idx, method='ffill').bfill().values.astype(np.float32)
print(f'aligned: 5m={n5.shape}  15m={n15.shape}')

# Save combined new feats
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
vi = D['vi'].astype(np.int64)

# Concat only vi rows → saves memory
X_new = np.column_stack([n5[vi], n15[vi]])
names_new = [f'5m_{c}' for c in f5.columns] + [f'15m_{c}' for c in f15.columns]
print(f'X_new (vi rows)={X_new.shape}')
np.savez('/workspace/models/eth_newfeats.npz', X=X_new, feat_names=np.array(names_new))
print('saved eth_newfeats.npz')

del n5, n15, X_new; gc.collect()

# Now load old data, concat
X_old = D['X'].astype(np.float32)
old_names = list(D['feat_names'])

# Load new
N = np.load('/workspace/models/eth_newfeats.npz', allow_pickle=True)
X_combo = np.column_stack([X_old, N['X'].astype(np.float32)])
combo_names = old_names + list(N['feat_names'])
print(f'X_combo={X_combo.shape}  ({X_combo.shape[1]} feats)')

# Replace NaN → -999 (LGB missing)
print('checking NaN...')
nans = np.isnan(X_combo).any(axis=1)
print(f'rows with NaN: {nans.sum():,}')
X_combo = np.where(nans[:,None], -999, X_combo).astype(np.float32)
print(f'clean X_combo={X_combo.shape}  mem={X_combo.nbytes/1e9:.2f}GB')

# Save combined
out_path = '/workspace/models/eth_combined.npz'
np.savez_compressed(out_path, X=X_combo, vi=D['vi'], ts=D['ts'], 
                     feat_names=np.array(combo_names),
                     ret_3=D['ret_3'], ret_5=D['ret_5'], ret_15=D['ret_15'], ret_30=D['ret_30'], ret_60=D['ret_60'],
                     y_3=D['y_3'], y_5=D['y_5'], y_15=D['y_15'], y_30=D['y_30'], y_60=D['y_60'])
print(f'saved {out_path}')
del X_combo, X_old; gc.collect()
print('done!')
