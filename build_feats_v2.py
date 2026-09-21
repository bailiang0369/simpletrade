"""Compact v2: ~100 key features, float16 stacking, then cast float32 on valid rows."""
import numpy as np, pandas as pd, time, gc, sys
import ta
sys.path.insert(0,'/workspace'); import config

t0 = time.time()
print('[1] load...', flush=True)
raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
raw_b = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
N = len(raw_e); ts = raw_e['ts'].values.astype(np.int64)
print(f'  N={N:,}  {time.time()-t0:.1f}s')

Ce = pd.Series(raw_e['close'].values.astype(np.float64), index=ts)
Cb = pd.Series(raw_b['close'].values.astype(np.float64), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')

def to_f16(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, -65504, 65504)
    return arr.astype(np.float16)

# Collect (name, f16_array) tuples
pairs = []

print('[2] ETH returns...', flush=True)
for w in [1,3,5,10,15,20,30,45,60,90,120,180,240,360,480,720,960,1440]:
    pairs.append((f'lr_e_{w}', to_f16((Ce/Ce.shift(w)-1).values)))

print('[3] BTC returns...', flush=True)
for w in [1,5,15,30,60,120,240,480,960,1440]:
    pairs.append((f'lr_b_{w}', to_f16((Cb/Cb.shift(w)-1).values)))
ratio = Ce / (Cb + 1e-10)
for w in [5,15,30,60,120,240]:
    pairs.append((f'lr_r_{w}', to_f16((ratio/ratio.shift(w)-1).values)))
del ratio, Cb; gc.collect()

print('[4] vol + z + pos...', flush=True)
lr1 = Ce.pct_change()
for w in [15,30,60,120,240,480]:
    pairs.append((f'rvol_e_{w}', to_f16(lr1.rolling(w).std().values)))
for w in [15,30,60,120]:
    z = pd.Series(lr1.rolling(w).std())
    zr = z.rolling(1440)
    pairs.append((f'rvol_z_{w}', to_f16(((z - zr.mean())/(zr.std()+1e-9)).values)))
for w in [15,30,60,120,240,480]:
    mu = Ce.rolling(w).mean(); sd = Ce.rolling(w).std()
    pairs.append((f'z_e_{w}', to_f16(((Ce-mu)/(sd+1e-9)).values)))
He = pd.Series(raw_e['high'].values.astype(np.float64), index=ts)
Le = pd.Series(raw_e['low'].values.astype(np.float64), index=ts)
for w in [30,60,120,240,480,960]:
    pairs.append((f'pos_{w}', to_f16(((Ce - Le.rolling(w).min()) / (He.rolling(w).max() - Le.rolling(w).min() + 1e-10)).values)))
    pairs.append((f'hh_{w}', to_f16(((Ce / He.rolling(w).max() - 1) * 100).values)))
    pairs.append((f'll_{w}', to_f16(((Ce / Le.rolling(w).min() - 1) * 100).values)))
del lr1; gc.collect()

print('[5] OHLC...', flush=True)
Oe = pd.Series(raw_e['open'].values.astype(np.float64), index=ts)
rng = (He - Le) + 1e-10
pairs.append(('body_pct', to_f16(((Ce - Oe) / rng).values)))
pairs.append(('body_abs_pct', to_f16(((Ce - Oe).abs() / rng).values)))
pairs.append(('up_wick', to_f16(((He - Ce.where(Ce > Oe, Oe)) / rng).values)))
pairs.append(('lo_wick', to_f16(((Ce.where(Ce < Oe, Oe) - Le) / rng).values)))
pairs.append(('gap', to_f16(((Oe / Ce.shift(1) - 1) * 100).values)))
pairs.append(('close_in_range', to_f16(((Ce - Le) / rng).values)))
del rng; gc.collect()

# streaks
sign_s = np.sign(Ce.pct_change().values).astype(np.float32)
streak = np.zeros(N, np.float32)
for i in range(1, N):
    streak[i] = streak[i-1] + sign_s[i] if sign_s[i] == sign_s[i-1] else sign_s[i]
pairs.append(('streak', to_f16(streak)))
for w in [10,30,60]:
    pairs.append((f'streak_sum_{w}', to_f16(pd.Series(streak).rolling(w).sum().values)))
del streak, sign_s; gc.collect()

print('[6] TA...', flush=True)
ce_arr = raw_e['close'].values.astype(np.float64)
he_arr = raw_e['high'].values.astype(np.float64)
le_arr = raw_e['low'].values.astype(np.float64)
for w in [7,14,28]:
    try: pairs.append((f'rsi_{w}', to_f16(ta.momentum.RSIIndicator(ce_arr, window=w).rsi().values/100.0)))
    except: pass
try: pairs.append(('macd_diff', to_f16(ta.trend.MACD(ce_arr).macd_diff().values)))
except: pass
try: pairs.append(('stoch_k', to_f16(ta.momentum.StochasticOscillator(he_arr, le_arr, ce_arr).stoch().values/100.0)))
except: pass
try:
    bb = ta.volatility.BollingerBands(ce_arr)
    pairs.append(('bb_pct', to_f16(bb.bollinger_pband().values)))
    pairs.append(('bb_w', to_f16(bb.bollinger_wband().values)))
except: pass
try:
    adx = ta.trend.ADXIndicator(he_arr, le_arr, ce_arr)
    pairs.append(('adx', to_f16(adx.adx().values/100.0)))
except: pass

print('[7] vol + funding...', flush=True)
bv = pd.Series(raw_e['buy_vol'].values.astype(np.float64), index=ts)
sv = pd.Series(raw_e['sell_vol'].values.astype(np.float64), index=ts)
for w in [5,15,30,60,120]:
    pairs.append((f'cvd_{w}', to_f16(((bv-sv).rolling(w).sum() / (bv+sv).rolling(w).sum() + 1e-10).values)))
    pairs.append((f'vol_e_{w}', to_f16(((bv+sv).rolling(w).mean() / (bv+sv).rolling(120).mean() + 1e-10).values)))
del bv, sv; gc.collect()
fund = pd.Series(raw_e['funding'].values.astype(np.float64), index=ts)
fund_b = pd.Series(raw_b['funding'].values.astype(np.float64), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
for w in [1,15,30,60,120,240]:
    pairs.append((f'fund_e_{w}', to_f16(fund.rolling(w).mean().values)))
    pairs.append((f'fund_b_{w}', to_f16(fund_b.rolling(w).mean().values)))
pairs.append(('fund_e_z', to_f16(((fund - fund.rolling(240).mean()) / (fund.rolling(240).std() + 1e-9)).values)))
pairs.append(('fund_e_slope', to_f16((fund.rolling(15).mean() - fund.rolling(60).mean()).values)))
pairs.append(('fund_diff', to_f16((fund.rolling(30).mean() - fund_b.rolling(30).mean()).values)))
del fund, fund_b, raw_e, raw_b; gc.collect()

print('[8] skew + hour...', flush=True)
lr_e = Ce.pct_change()
for w in [60,240]:
    pairs.append((f'skew_{w}', to_f16(lr_e.rolling(w).skew().values)))
    pairs.append((f'kurt_{w}', to_f16(lr_e.rolling(w).kurt().values)))
del Ce, lr_e; gc.collect()
hr_arr = pd.to_datetime(ts, unit='s', utc=True).hour.values.astype(np.float32)
dow_arr = pd.to_datetime(ts, unit='s', utc=True).dayofweek.values.astype(np.float32)
pairs.append(('hour_sin', np.sin(2*np.pi*hr_arr/24).astype(np.float16)))
pairs.append(('hour_cos', np.cos(2*np.pi*hr_arr/24).astype(np.float16)))
pairs.append(('dow_sin', np.sin(2*np.pi*dow_arr/7).astype(np.float16)))
pairs.append(('dow_cos', np.cos(2*np.pi*dow_arr/7).astype(np.float16)))

# Sort
pairs.sort(key=lambda x: x[0])
FEAT_NAMES = [p[0] for p in pairs]
feat_f16 = [p[1] for p in pairs]
del pairs; gc.collect()
print(f'  feats={len(FEAT_NAMES)}  {time.time()-t0:.1f}s', flush=True)

# ========= STACK =========
print('[9] stack float16...', flush=True)
t = time.time()
X_all_f16 = np.stack(feat_f16, axis=1)  # float16, ~2.2GB
del feat_f16; gc.collect()
print(f'  {time.time()-t:.1f}s  {X_all_f16.shape} dtype={X_all_f16.dtype}', flush=True)

# Valid mask
finite = np.isfinite(X_all_f16).all(axis=1)
warmup = np.arange(N) > 2000
vi = np.where(finite & warmup)[0]
print(f'  valid rows={len(vi):,}', flush=True)

# Slice valid rows → float32 for model
print('[10] slice → float32...', flush=True)
X = X_all_f16[vi].astype(np.float32)
del X_all_f16; gc.collect()
print(f'  X={X.shape}  mem={X.nbytes/1e9:.2f}GB  {time.time()-t:.1f}s', flush=True)

# ========= LABELS =========
print('[11] labels...', flush=True)
close_eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
labels = {}
for h in [1,3,5,10,15,30,60]:
    rh = np.full(N, np.nan, np.float32)
    rh[:-h] = (close_eth[h:] / close_eth[:-h] - 1).astype(np.float32)
    labels[f'ret_{h}'] = rh[vi]
    labels[f'y_{h}'] = (rh[vi] > 0).astype(np.int8)
del close_eth; gc.collect()

# ========= SAVE =========
print('[12] save...', flush=True)
save_data = {
    'X': X, 'vi': vi, 'ts': ts,
    'feat_names': np.array(FEAT_NAMES),
    'hr': hr_arr[vi].astype(np.float32),
    'dow': dow_arr[vi].astype(np.float32),
}
save_data.update(labels)
np.savez('/workspace/models/eth_data.npz', **save_data)
print(f'DONE! X={X.shape}  feats={len(FEAT_NAMES)}  {time.time()-t0:.0f}s')
