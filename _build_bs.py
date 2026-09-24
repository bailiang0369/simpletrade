"""Build buy/sell volume splits for BTC h15"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import polars as pl
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
PAR='data/datasets'; NPY='data/splits_npy'

raw = pl.read_parquet(f'{PAR}/raw_BTC.parquet', columns=['ts','buy_vol','sell_vol','close']).sort('ts')
bv = raw['buy_vol'].to_numpy().astype(np.float64)
sv = raw['sell_vol'].to_numpy().astype(np.float64)
cv = raw['close'].to_numpy().astype(np.float64)
ts_all = raw['ts'].to_numpy().astype(np.int64)
print(f"[1] buy_vol/sell_vol quality:")
print(f"  buy_vol: nan={np.isnan(bv).sum():,}, zero={(bv==0).sum():,}, min={np.nanmin(bv):.2f}")
print(f"  sell_vol: nan={np.isnan(sv).sum():,}, zero={(sv==0).sum():,}, min={np.nanmin(sv):.2f}")
print(f"  rows with bv+sv=0: {((bv+sv)==0).sum():,}")

total_vol = np.where(bv + sv > 0, bv + sv, np.nan)
imb = (bv - sv) / total_vol
net_vol = bv - sv

fr_bv = pd.Series(bv); fr_sv = pd.Series(sv); fr_imb = pd.Series(imb); fr_net = pd.Series(net_vol)

cols = {}
cols['bs_imbalance'] = imb.astype(np.float32)
cols['buy_ratio'] = (bv / total_vol).astype(np.float32)
cols['buy_sell_log'] = np.log(np.where(sv > 0, bv/sv, np.nan)).astype(np.float32)

for w in (5, 15, 60, 240):
    cols[f'bs_imb_ma{w}'] = fr_imb.rolling(w, min_periods=3).mean().to_numpy().astype(np.float32)
    cols[f'bs_imb_std{w}'] = fr_imb.rolling(w, min_periods=3).std().to_numpy().astype(np.float32)
    imb_mean = fr_imb.rolling(w, min_periods=3).mean().to_numpy()
    imb_std = fr_imb.rolling(w, min_periods=3).std().to_numpy()
    cols[f'bs_imb_z{w}'] = np.where(imb_std > 1e-9, (imb - imb_mean) / imb_std, 0.0).astype(np.float32)

for w in (5, 15, 60):
    cols[f'net_vol_ma{w}'] = fr_net.rolling(w, min_periods=3).mean().to_numpy().astype(np.float32)
    cols[f'net_vol_d{w}'] = (net_vol - np.roll(net_vol, w)).astype(np.float32)

for w in (60, 240):
    bv_ma = fr_bv.rolling(w, min_periods=30).mean().to_numpy()
    sv_ma = fr_sv.rolling(w, min_periods=30).mean().to_numpy()
    cols[f'bv_spike{w}'] = np.where(bv_ma > 0, bv / bv_ma, 1.0).astype(np.float32)
    cols[f'sv_spike{w}'] = np.where(sv_ma > 0, sv / sv_ma, 1.0).astype(np.float32)

print(f"\n  buy/sell feats: {len(cols)}")

keys = sorted(cols.keys())
bs_arr = np.stack([cols[k] for k in keys], axis=1).astype(np.float32)
print(f"  bs_arr shape: {bs_arr.shape}")
print(f"  NaN counts per col: {[np.isnan(bs_arr[:,i]).sum() for i in range(len(keys))]}")
del cols, fr_bv, fr_sv, fr_imb, fr_net, imb, net_vol, total_vol, bv, sv, cv; gc.collect()

TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
for split, tlo, thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),
                        ('meta_val',ES_END,META_END),('test',META_END,10**18)]:
    m = (ts_all>=tlo)&(ts_all<thi)
    np.save(f'{NPY}/BTC_h15_bs_{split}_X.npy', bs_arr[m])
    print(f"  bs {split}: {bs_arr[m].shape}")

del bs_arr, ts_all; gc.collect()
print(f"\nBuy/sell splits saved ✓  feats={keys}")
