"""Build combined dataset memory-efficiently."""
import numpy as np, gc

print('load old...')
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X_old = D['X'].astype(np.float32)
old_names = list(D['feat_names'])

print('load new...')
N = np.load('/workspace/models/eth_newfeats.npz', allow_pickle=True)
X_new = N['X'].astype(np.float32)
new_names = list(N['feat_names'])

print(f'old={X_old.shape}  new={X_new.shape}')

# Concat
X_combo = np.column_stack([X_old, X_new])
print(f'combo={X_combo.shape}  mem={X_combo.nbytes/1e9:.2f}GB')

# Check NaN (in-place — doesn't allocate full copy)
print('NaN check...')
nans = np.isnan(X_combo).any(axis=1)
print(f'  NaN rows: {nans.sum()}')

del nans, X_old, X_new; gc.collect()

# Save (don't use np.savez_compressed — it loads everything into memory for compression)
out = '/workspace/models/eth_combined.npz'
print(f'save to {out}...')
np.savez(out, X=X_combo, vi=D['vi'], ts=D['ts'],
         feat_names=np.array(old_names + new_names),
         ret_3=D['ret_3'], ret_5=D['ret_5'], ret_15=D['ret_15'], ret_30=D['ret_30'], ret_60=D['ret_60'],
         y_3=D['y_3'], y_5=D['y_5'], y_15=D['y_15'], y_30=D['y_30'], y_60=D['y_60'])
print(f'saved! size={__import__("os").path.getsize(out)/1e9:.2f}GB')
del X_combo; gc.collect()
print('done')
