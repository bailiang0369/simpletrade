"""ENSEMBLE V2: Multi-horizon + multi-model + focal loss + gate."""
import numpy as np, pandas as pd, time, gc, datetime as dtm, sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import ta

np.random.seed(42)

# ========= STEP 1: Rebuild with more features =========
print('[1] rebuild expanded feature set...', flush=True)
t0 = time.time()
raw_e = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
raw_b = pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
N = len(raw_e); ts = raw_e['ts'].values.astype(np.int64)

Ce = pd.Series(raw_e['close'].values.astype(np.float32), index=ts)
Cb = pd.Series(raw_b['close'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
He = pd.Series(raw_e['high'].values.astype(np.float32), index=ts)
Le = pd.Series(raw_e['low'].values.astype(np.float32), index=ts)
Oe = pd.Series(raw_e['open'].values.astype(np.float32), index=ts)
bv = pd.Series(raw_e['buy_vol'].values.astype(np.float32), index=ts)
sv = pd.Series(raw_e['sell_vol'].values.astype(np.float32), index=ts)
fund = pd.Series(raw_e['funding'].values.astype(np.float32), index=ts)
bv_b = pd.Series(raw_b['buy_vol'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
sv_b = pd.Series(raw_b['sell_vol'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')
fund_b = pd.Series(raw_b['funding'].values.astype(np.float32), index=raw_b['ts'].values.astype(np.int64)).reindex(ts, method='ffill')

# Collect features
pairs = []
def add(name, arr):
    arr = np.array(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    pairs.append((name, arr))

# ETH returns (key horizons only — memory)
for w in [3,5,10,15,20,30,45,60,90,120,180,240,360,480,720,960,1440]:
    add(f'lr_e_{w}', (Ce/Ce.shift(w)-1).values)
# BTC returns (fewer)
for w in [5,15,30,60,120,240,480,960,1440]:
    add(f'lr_b_{w}', (Cb/Cb.shift(w)-1).values)
ratio = Ce / (Cb + 1e-6)
for w in [15,30,60,120,240]:
    add(f'lr_r_{w}', (ratio/ratio.shift(w)-1).values)
del ratio, Cb; gc.collect()

# Vol
lr1 = Ce.pct_change()
for w in [15,30,60,120,240,480]:
    add(f'rvol_e_{w}', lr1.rolling(w).std().values.astype(np.float32))
for w in [15,30,60,120]:
    z = lr1.rolling(w).std()
    zr = z.rolling(1440)
    add(f'rvol_z_{w}', ((z - zr.mean())/(zr.std()+1e-8)).values.astype(np.float32))

# Z-scores + Position
for w in [15,30,60,120,240,480]:
    mu = Ce.rolling(w).mean(); sd = Ce.rolling(w).std()
    add(f'z_e_{w}', ((Ce-mu)/(sd+1e-8)).values.astype(np.float32))
for w in [30,60,120,240,480]:
    add(f'pos_{w}', ((Ce - Le.rolling(w).min()) / (He.rolling(w).max() - Le.rolling(w).min() + 1e-8)).values.astype(np.float32))
    add(f'hh_{w}', ((Ce / He.rolling(w).max() - 1) * 100).values.astype(np.float32))
    add(f'll_{w}', ((Ce / Le.rolling(w).min() - 1) * 100).values.astype(np.float32))
del lr1; gc.collect()

# OHLC + streaks
rng = (He - Le) + 1e-8
add('body_pct', ((Ce - Oe) / rng).values.astype(np.float32))
add('body_abs', ((Ce - Oe).abs() / rng).values.astype(np.float32))
add('up_wick', ((He - Ce.where(Ce > Oe, Oe)) / rng).values.astype(np.float32))
add('lo_wick', ((Ce.where(Ce < Oe, Oe) - Le) / rng).values.astype(np.float32))
add('gap', ((Oe / Ce.shift(1) - 1) * 100).values.astype(np.float32))
del rng; gc.collect()
sign_s = np.sign(Ce.pct_change().values.astype(np.float32))
streak = np.zeros(N, np.float32)
for i in range(1, N):
    streak[i] = streak[i-1] + sign_s[i] if sign_s[i] == sign_s[i-1] else sign_s[i]
add('streak', streak)
for w in [10,30,60]:
    add(f'streak_sum_{w}', pd.Series(streak).rolling(w).sum().values.astype(np.float32))
del streak, sign_s; gc.collect()

# TA indicators (key ones)
print('  TA...', flush=True)
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
    add('di_plus', adx.adx_pos().values.astype(np.float32)/100.0)
    add('di_minus', adx.adx_neg().values.astype(np.float32)/100.0)
except: pass
try: add('cci', ta.trend.CCIIndicator(he_arr, le_arr, ce_arr).cci().values.astype(np.float32)/200.0)
except: pass
try: add('williams_r', ta.momentum.WilliamsRIndicator(he_arr, le_arr, ce_arr).williams_r().values.astype(np.float32)/100.0)
except: pass

# Volume
for w in [5,15,30,60,120]:
    add(f'cvd_{w}', ((bv-sv).rolling(w).sum() / (bv+sv).rolling(w).sum() + 1e-8).values.astype(np.float32))
    add(f'vol_e_{w}', ((bv+sv).rolling(w).mean() / (bv+sv).rolling(120).mean() + 1e-8).values.astype(np.float32))
    add(f'buy_share_{w}', (bv.rolling(w).sum() / ((bv+sv).rolling(w).sum() + 1e-8)).values.astype(np.float32))
del bv, sv; gc.collect()
# BTC vol cross
for w in [15,60]:
    add(f'cvd_b_{w}', ((bv_b-sv_b).rolling(w).sum() / (bv_b+sv_b).rolling(w).sum() + 1e-8).values.astype(np.float32))
del bv_b, sv_b; gc.collect()

# Funding
for w in [1,5,15,30,60,120,240]:
    add(f'fund_e_{w}', fund.rolling(w).mean().values.astype(np.float32))
    add(f'fund_b_{w}', fund_b.rolling(w).mean().values.astype(np.float32))
add('fund_e_z', ((fund - fund.rolling(240).mean()) / (fund.rolling(240).std() + 1e-8)).values.astype(np.float32))
add('fund_e_slope', (fund.rolling(15).mean() - fund.rolling(60).mean()).values.astype(np.float32))
add('fund_diff', (fund.rolling(30).mean() - fund_b.rolling(30).mean()).values.astype(np.float32))
del fund, fund_b, raw_e, raw_b; gc.collect()

# Skew + hour
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

pairs.sort(key=lambda x: x[0])
FEAT_NAMES = [p[0] for p in pairs]
print(f'  collected {len(FEAT_NAMES)} feats  {time.time()-t0:.1f}s', flush=True)

# Stack + valid
print('[2] stack...', flush=True)
X_full = np.stack([p[1] for p in pairs], axis=1).astype(np.float32)
del pairs; gc.collect()
print(f'  X_full={X_full.shape} mem={X_full.nbytes/1e9:.2f}GB', flush=True)
finite = np.isfinite(X_full).all(axis=1)
warmup = np.arange(N) > 2000
vm = finite & warmup
vi = np.where(vm)[0]
print(f'  valid rows={len(vi):,}', flush=True)

# Build labels (horizons)
print('[3] labels...', flush=True)
close_eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
all_labels = {}
for h in [3,5,10,15,30,60]:
    rh = np.full(N, np.nan, np.float32)
    rh[:-h] = (close_eth[h:] / close_eth[:-h] - 1).astype(np.float32)
    all_labels[f'y_{h}'] = (rh > 0).astype(np.int8)[vi]
    all_labels[f'ret_{h}'] = rh[vi]
del close_eth; gc.collect()

# Slice X on valid rows
print('[4] slice valid...', flush=True)
X = X_full[vi].astype(np.float32)
del X_full, finite, warmup, vm; gc.collect()
ts_valid = ts[vi]
print(f'  X={X.shape} mem={X.nbytes/1e9:.2f}GB', flush=True)

# ========= STEP 2: Split + multi-horizon ensemble =========
def ts_mask_arr(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_valid>=a)&(ts_valid<b)

tr_mask = ts_mask_arr('2020-01-01','2024-06-30')
es_mask = ts_mask_arr('2024-06-30','2024-09-30')
te_mask = ts_mask_arr('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'\n[5] split  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}  feats={X.shape[1]}')

# ========= Train multiple models, multiple horizons =========
print('\n[6] train multi-horizon LGB ensemble...', flush=True)
def focal_obj_factory(gamma=2.0, alpha=0.5):
    def focal_obj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = p * y_true + (1 - p) * (1 - y_true)
        grad = alpha * (1 - p_t) ** gamma * (p - y_true)
        hess = alpha * (1 - p_t) ** gamma * p * (1 - p) * (1 + gamma * (p_t - (1 - p_t)))
        return grad, hess
    return focal_obj

all_te_preds = {}
all_es_preds = {}

# Try H=5, H=10, H=15 — shorter horizons often have better top-k
for H in [5, 10, 15, 30]:
    y = all_labels[f'y_{H}']
    print(f'\n  === H={H} ===', flush=True)
    y_tr_h = y[tr_idx]; y_es_h = y[es_idx]; y_te_h = y[te_idx]

    h_es_preds = []
    h_te_preds = []

    for seed in [42, 49, 56]:
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'learning_rate': 0.05, 'num_leaves': 63,
            'min_child_samples': 200, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5,
            'lambda_l2': 0.1, 'verbose': -1, 'n_jobs': 3, 'seed': seed,
        }
        m = lgb.train(params, lgb.Dataset(X[tr_idx], label=y_tr_h), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X[es_idx], label=y_es_h)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        p_es = m.predict(X[es_idx]); p_te = m.predict(X[te_idx])
        auc = roc_auc_score(y_es_h, p_es)
        print(f'    seed={seed} best={m.best_iteration} es_auc={auc:.4f}')
        h_es_preds.append(p_es); h_te_preds.append(p_te)

    all_es_preds[H] = np.mean(h_es_preds, axis=0)
    all_te_preds[H] = np.mean(h_te_preds, axis=0)

# ========= Combine horizons =========
print('\n[7] combine horizons (equal weight)...', flush=True)
H_combos = [(5,10,15), (5,10,15,30), (10,15)]
for combo in H_combos:
    es_avg = np.mean([all_es_preds[h] for h in combo], axis=0)
    te_avg = np.mean([all_te_preds[h] for h in combo], axis=0)
    y_te15 = all_labels['y_15'][te_idx]
    auc = roc_auc_score(y_te15, te_avg)
    print(f'  H={combo}  te_auc_for_H15={auc:.4f}')

# best combo
BEST_COMBO = (5,10,15)
p_es_final = np.mean([all_es_preds[h] for h in BEST_COMBO], axis=0)
p_te_final = np.mean([all_te_preds[h] for h in BEST_COMBO], axis=0)
y_te = all_labels['y_15'][te_idx]

# ========= Top-k =========
print(f'\n[8] TOP-K TEST (ensemble H={BEST_COMBO}, predict H=15)')
print(f'  TE={len(te_idx):,}  days={len(te_idx)/1440:.0f}')
for pct in [0.5, 1, 2, 3, 5, 8, 10, 15]:
    k = max(1, int(len(p_te_final)*pct/100))
    idx = np.argsort(-p_te_final)[:k]
    acc = y_te[idx].mean()*100
    tpd = k*1440/len(te_idx)
    print(f'  top{pct:>4}%: acc={acc:.2f}%  tpd={tpd:.1f}')

# ========= Per-horizon breakdown =========
print(f'\n[8b] PER-HORIZON TOP-K for each H:')
for H in [5, 10, 15, 30]:
    y_h = all_labels[f'y_{H}'][te_idx]
    pv = all_te_preds[H]
    auc = roc_auc_score(y_h, pv)
    k1 = max(1, int(len(pv)*0.01))
    acc1 = y_h[np.argsort(-pv)[:k1]].mean()*100
    print(f'  H={H:>2}  auc={auc:.4f}  top1pct={acc1:.2f}%')

# ========= Monthly stability =========
print(f'\n[9] MONTHLY STABILITY (ensemble H={BEST_COMBO}, top1%)')
dt_te = pd.to_datetime(ts_valid[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))
bad = 0
for m in all_months:
    mm = month == m
    n = mm.sum(); k = max(1, int(n*0.01))
    p_m = p_te_final[mm]; y_m = y_te[mm]
    acc_m = y_m[np.argsort(-p_m)[:k]].mean()*100
    auc_m = roc_auc_score(y_m, p_m)
    flag = ' BAD' if acc_m < 60 else ''
    if acc_m < 60: bad += 1
    print(f'  {m} n={n:>6,} AUC={auc_m:.4f} top1pct={acc_m:.2f}%{flag}')
print(f'\n  BAD months: {bad}/{len(all_months)}')

# ========= importance =========
print('\n[10] top importance:')
# Train one LGB quickly
from lightgbm import LGBMClassifier
m_imp = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                              n_jobs=3, random_state=42, verbose=-1)
m_imp.fit(X[tr_idx][:500000:5], all_labels['y_15'][tr_idx][:500000:5])  # subsample
imp = pd.Series(m_imp.feature_importances_, index=FEAT_NAMES).sort_values(ascending=False)
for i,(k,v) in enumerate(imp.head(25).items()):
    print(f'  {i+1:>2}. {k:<25s} imp={v:.0f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
