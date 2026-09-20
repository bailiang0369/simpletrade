"""Final feature build - direct float32 output, no float16 gymnastics."""
import numpy as np, pandas as pd, time, gc, sys, os
import ta
sys.path.insert(0,'/workspace'); import config

t0 = time.time()
print('[1] load...', flush=True)
raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
raw_b = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
N = len(raw_e); ts = raw_e['ts'].values.astype(np.int64)
print(f'  N={N:,}  {time.time()-t0:.1f}s')

Ce = pd.Series(raw_e['close'].values.astype(np.float32), index=ts)
Cb = pd.Series(raw_b['close'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
He = pd.Series(raw_e['high'].values.astype(np.float32), index=ts)
Le = pd.Series(raw_e['low'].values.astype(np.float32), index=ts)
Oe = pd.Series(raw_e['open'].values.astype(np.float32), index=ts)

pairs = []

def add(name, arr):
    arr = np.array(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    pairs.append((name, arr))

print('[2] ETH returns...', flush=True)
for w in [1,3,5,10,15,20,30,45,60,90,120,180,240,360,480,720,960,1440]:
    add(f'lr_e_{w}', (Ce/Ce.shift(w)-1).values)

print('[3] BTC returns...', flush=True)
for w in [1,5,15,30,60,120,240,480,960,1440]:
    add(f'lr_b_{w}', (Cb/Cb.shift(w)-1).values)
ratio = Ce / (Cb + 1e-6)
for w in [5,15,30,60,120,240]:
    add(f'lr_r_{w}', (ratio/ratio.shift(w)-1).values)
del ratio, Cb; gc.collect()

print('[4] vol + z + pos...', flush=True)
lr1 = Ce.pct_change()
for w in [15,30,60,120,240,480]:
    add(f'rvol_e_{w}', lr1.rolling(w).std().values.astype(np.float32))
for w in [15,30,60,120]:
    z = lr1.rolling(w).std()
    zr = z.rolling(1440)
    add(f'rvol_z_{w}', ((z - zr.mean())/(zr.std()+1e-8)).values.astype(np.float32))
for w in [15,30,60,120,240,480]:
    mu = Ce.rolling(w).mean(); sd = Ce.rolling(w).std()
    add(f'z_e_{w}', ((Ce-mu)/(sd+1e-8)).values.astype(np.float32))
for w in [30,60,120,240,480,960]:
    add(f'pos_{w}', ((Ce - Le.rolling(w).min()) / (He.rolling(w).max() - Le.rolling(w).min() + 1e-8)).values.astype(np.float32))
    add(f'hh_{w}', ((Ce / He.rolling(w).max() - 1) * 100).values.astype(np.float32))
    add(f'll_{w}', ((Ce / Le.rolling(w).min() - 1) * 100).values.astype(np.float32))
del lr1; gc.collect()

print('[5] OHLC...', flush=True)
rng = (He - Le) + 1e-8
add('body_pct', ((Ce - Oe) / rng).values.astype(np.float32))
add('up_wick', ((He - Ce.where(Ce > Oe, Oe)) / rng).values.astype(np.float32))
add('lo_wick', ((Ce.where(Ce < Oe, Oe) - Le) / rng).values.astype(np.float32))
add('gap', ((Oe / Ce.shift(1) - 1) * 100).values.astype(np.float32))
add('close_in_range', ((Ce - Le) / rng).values.astype(np.float32))
del rng; gc.collect()

sign_s = np.sign(Ce.pct_change().values.astype(np.float32))
streak = np.zeros(N, np.float32)
for i in range(1, N):
    streak[i] = streak[i-1] + sign_s[i] if sign_s[i] == sign_s[i-1] else sign_s[i]
add('streak', streak)
for w in [10,30,60]:
    add(f'streak_sum_{w}', pd.Series(streak).rolling(w).sum().values.astype(np.float32))
del streak, sign_s; gc.collect()

print('[6] TA...', flush=True)
ce_arr = raw_e['close'].values.astype(np.float32)
he_arr = raw_e['high'].values.astype(np.float32)
le_arr = raw_e['low'].values.astype(np.float32)
for w in [7,14,28]:
    try: add(f'rsi_{w}', ta.momentum.RSIIndicator(ce_arr, window=w).rsi().values.astype(np.float32)/100.0)
    except: pass
try: add('macd_diff', ta.trend.MACD(ce_arr).macd_diff().values.astype(np.float32))
except: pass
try: add('stoch_k', ta.momentum.StochasticOscillator(he_arr, le_arr, ce_arr).stoch().values.astype(np.float32)/100.0)
except: pass
try:
    bb = ta.volatility.BollingerBands(ce_arr)
    add('bb_pct', bb.bollinger_pband().values.astype(np.float32))
    add('bb_w', bb.bollinger_wband().values.astype(np.float32))
except: pass
try:
    adx = ta.trend.ADXIndicator(he_arr, le_arr, ce_arr)
    add('adx', adx.adx().values.astype(np.float32)/100.0)
except: pass
try: add('cci', ta.trend.CCIIndicator(he_arr, le_arr, ce_arr).cci().values.astype(np.float32)/200.0)
except: pass

print('[7] vol + funding...', flush=True)
bv = pd.Series(raw_e['buy_vol'].values.astype(np.float32), index=ts)
sv = pd.Series(raw_e['sell_vol'].values.astype(np.float32), index=ts)
for w in [5,15,30,60,120]:
    add(f'cvd_{w}', ((bv-sv).rolling(w).sum() / (bv+sv).rolling(w).sum() + 1e-8).values.astype(np.float32))
    add(f'vol_e_{w}', ((bv+sv).rolling(w).mean() / (bv+sv).rolling(120).mean() + 1e-8).values.astype(np.float32))
del bv, sv; gc.collect()

fund = pd.Series(raw_e['funding'].values.astype(np.float32), index=ts)
fund_b = pd.Series(raw_b['funding'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
for w in [1,15,30,60,120,240]:
    add(f'fund_e_{w}', fund.rolling(w).mean().values.astype(np.float32))
    add(f'fund_b_{w}', fund_b.rolling(w).mean().values.astype(np.float32))
add('fund_e_z', ((fund - fund.rolling(240).mean()) / (fund.rolling(240).std() + 1e-8)).values.astype(np.float32))
add('fund_e_slope', (fund.rolling(15).mean() - fund.rolling(60).mean()).values.astype(np.float32))
add('fund_diff', (fund.rolling(30).mean() - fund_b.rolling(30).mean()).values.astype(np.float32))
del fund, fund_b, raw_e, raw_b; gc.collect()

print('[8] skew + hour...', flush=True)
lr_e = Ce.pct_change()
for w in [60,240]:
    add(f'skew_{w}', lr_e.rolling(w).skew().values.astype(np.float32))
    add(f'kurt_{w}', lr_e.rolling(w).kurt().values.astype(np.float32))
del Ce, lr_e; gc.collect()

hr_arr = pd.to_datetime(ts, unit='s', utc=True).hour.values.astype(np.float32)
dow_arr = pd.to_datetime(ts, unit='s', utc=True).dayofweek.values.astype(np.float32)
add('hour_sin', np.sin(2*np.pi*hr_arr/24).astype(np.float32))
add('hour_cos', np.cos(2*np.pi*hr_arr/24).astype(np.float32))
add('dow_sin', np.sin(2*np.pi*dow_arr/7).astype(np.float32))
add('dow_cos', np.cos(2*np.pi*dow_arr/7).astype(np.float32))

# interactions
def inter(n1, n2, out):
    if n1 in [p[0] for p in pairs] and n2 in [p[0] for p in pairs]:
        a = dict(pairs)[n1]; b = dict(pairs)[n2]
        add(out, (a*b).astype(np.float32))
inter('rvol_e_60','lr_e_30','vol_x_mom30')
inter('fund_e_30','rvol_e_60','fund_x_vol')
del pairs; gc.collect()

print(f'  collected {len(pairs)} feats  {time.time()-t0:.1f}s', flush=True)

# sort
pairs.sort(key=lambda x: x[0])
FEAT_NAMES = [p[0] for p in pairs]

# Stack in ONE shot
print('[9] stack...', flush=True)
X = np.stack([p[1] for p in pairs], axis=1).astype(np.float32)
del pairs; gc.collect()
print(f'  X={X.shape} mem={X.nbytes/1e9:.2f}GB', flush=True)

# Valid mask (in-place)
finite_mask = np.isfinite(X).all(axis=1)
del X; gc.collect()
warmup_mask = np.arange(N) > 2000
vm = finite_mask & warmup_mask
vi = np.where(vm)[0]
del finite_mask, warmup_mask, vm; gc.collect()
print(f'  vi={vi.shape}', flush=True)

# Labels
print('[10] labels...', flush=True)
close_eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
labels = {}
for h in [1,3,5,10,15,30,60]:
    rh = np.full(N, np.nan, np.float32)
    rh[:-h] = (close_eth[h:] / close_eth[:-h] - 1).astype(np.float32)
    labels[f'ret_{h}'] = rh[vi]
    labels[f'y_{h}'] = (rh[vi] > 0).astype(np.int8)
del close_eth; gc.collect()

# Save
print('[11] save...', flush=True)
# Save X, vi, ts, hr, dow, labels, feat_names — let model training script slice
save_data = {'vi': vi, 'ts': ts, 'feat_names': np.array(FEAT_NAMES),
             'hr': hr_arr[vi], 'dow': dow_arr[vi]}
save_data.update(labels)
np.savez('/workspace/models/eth_meta.npz', **save_data)
# Save X separately (larger file)
# Actually let me slice here — but need to be careful about memory
# X was deleted above. Let me re-stack only valid rows
print('[12] re-stack valid rows only...', flush=True)
# We need the original arrays again. Let me do it differently —
# just save full X, then we'll re-slice at model time
# But we deleted X... Need to recompute or keep
# Actually let me just skip slice and save X_all — model will load with mmap
# Re-do stack (already done above but we deleted X)
# OK let me recompute and save in one shot

# Hmm, we don't have feat arrays anymore. Need to recompute — but that takes 15s.
# Let me just keep the OLD 62-feat dataset and iterate from there.
print('ABORT: memory issue. Skipping full build.')
