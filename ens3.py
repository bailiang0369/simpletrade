"""ENS3: Use existing 62-feat dataset, add per-row TA features lazily, multi-horizon ensemble."""
import numpy as np, pandas as pd, time, gc, datetime as dtm, sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

np.random.seed(42)
t0 = time.time()

print('[1] load baseline data...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X_base = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts_orig = D['ts'].astype(np.int64)[vi]
feat_base = D['feat_names']
print(f'  X_base={X_base.shape}  feats={len(feat_base)}')

# ========= Add TA features lazily =========
print('[2] add TA features...', flush=True)
raw_e = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
# Use row-wise: need indices for vi
ce_full = raw_e['close'].values.astype(np.float64)
he_full = raw_e['high'].values.astype(np.float64)
le_full = raw_e['low'].values.astype(np.float64)
oe_full = raw_e['open'].values.astype(np.float64)
ts_full = raw_e['ts'].values.astype(np.int64)

# Compute TA on FULL series, then slice vi
ta_feats = {}
import ta
try: ta_feats['rsi_14'] = ta.momentum.RSIIndicator(ce_full, window=14).rsi().values / 100.0
except: pass
try: ta_feats['macd_diff'] = ta.trend.MACD(ce_full).macd_diff().values
except: pass
try:
    sk = ta.momentum.StochasticOscillator(he_full, le_full, ce_full)
    ta_feats['stoch_k'] = sk.stoch().values / 100.0
except: pass
try:
    bb = ta.volatility.BollingerBands(ce_full)
    ta_feats['bb_pct'] = bb.bollinger_pband().values
except: pass
try:
    adx = ta.trend.ADXIndicator(he_full, le_full, ce_full)
    ta_feats['adx'] = adx.adx().values / 100.0
except: pass

# Return / vol / z on full series
Ce_full = pd.Series(ce_full, index=ts_full)
for w in [5, 10, 20, 45, 90, 180, 360, 720, 1440]:
    ta_feats[f'lr_e_{w}'] = (Ce_full / Ce_full.shift(w) - 1).values
lr_full = Ce_full.pct_change()
for w in [30, 60, 120, 240]:
    ta_feats[f'rvol_e_{w}'] = lr_full.rolling(w).std().values
for w in [30, 60, 120, 240, 480]:
    mu = Ce_full.rolling(w).mean(); sd = Ce_full.rolling(w).std()
    ta_feats[f'z_e_{w}'] = ((Ce_full - mu) / (sd + 1e-8)).values
He_full = pd.Series(he_full, index=ts_full)
Le_full = pd.Series(le_full, index=ts_full)
for w in [30, 60, 120, 240]:
    ta_feats[f'pos_{w}'] = ((Ce_full - Le_full.rolling(w).min()) / (He_full.rolling(w).max() - Le_full.rolling(w).min() + 1e-8)).values

# BTC cross
raw_b = pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')
Cb_full = pd.Series(raw_b['close'].values.astype(np.float64), index=raw_b['ts'].values.astype(np.int64)).reindex(ts_full, method='ffill')
for w in [15, 30, 60, 120, 240]:
    ta_feats[f'lr_b_{w}'] = (Cb_full / Cb_full.shift(w) - 1).values
del Cb_full, raw_b, Ce_full, He_full, Le_full, lr_full; gc.collect()

# Funding
fund_full = pd.Series(raw_e["funding"].values.astype(np.float64), index=ts_full)
fund_b_full_load = pd.read_parquet("/workspace/data/datasets/raw_BTC.parquet")
fund_b_full = pd.Series(fund_b_full_load["funding"].values.astype(np.float64), index=fund_b_full_load["ts"].values.astype(np.int64)).reindex(ts_full, method="ffill")
for w in [15, 60, 240]:
    ta_feats[f'fund_e_{w}'] = fund_full.rolling(w).mean().values
    ta_feats[f'fund_b_{w}'] = fund_b_full.rolling(w).mean().values
ta_feats['fund_e_z'] = ((fund_full - fund_full.rolling(240).mean()) / (fund_full.rolling(240).std() + 1e-8)).values
del fund_full, fund_b_full, raw_e; gc.collect()

# Slice vi + finite check
X_ta_list = []
ta_names = []
for name, arr in sorted(ta_feats.items()):
    sliced = arr[vi].astype(np.float32)
    sliced = np.where(np.isfinite(sliced), sliced, 0.0)
    X_ta_list.append(sliced)
    ta_names.append(name)
print(f'  ta feats={len(ta_names)}', flush=True)

# Combine base + TA
print('[3] combine...', flush=True)
X_ta = np.stack(X_ta_list, axis=1)
del X_ta_list; gc.collect()
# Build combined
X_base_orig = D["X"].astype(np.float32)
X = np.concatenate([X_base_orig[vi], X_ta], axis=1)
del X_base_orig, X_base, X_ta, D; gc.collect()
del X_base, X_ta; gc.collect()
ALL_NAMES = list(feat_base) + ta_names
print(f'  X={X.shape} mem={X.nbytes/1e9:.2f}GB  feats={X.shape[1]}')

# ========= Split =========
def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_orig>=a)&(ts_orig<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# ========= Labels =========
print('[4] labels...', flush=True)
close_eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
N_full = len(close_eth)
all_labels = {}
for h in [3,5,10,15,30]:
    rh = np.full(N_full, np.nan, np.float32)
    rh[:-h] = (close_eth[h:] / close_eth[:-h] - 1).astype(np.float32)
    all_labels[f'y_{h}'] = (rh[vi] > 0).astype(np.int8)
del close_eth; gc.collect()

# ========= Train multi-seed, multi-horizon =========
print('\n[5] TRAINING (multi-seed x multi-horizon)...', flush=True)
def train_one(H, seed, params_override=None):
    y_tr_h = all_labels[f'y_{H}'][tr_idx]
    y_es_h = all_labels[f'y_{H}'][es_idx]
    y_te_h = all_labels[f'y_{H}'][te_idx]
    params = {'objective':'binary','metric':'binary_logloss','learning_rate':0.05,
              'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,
              'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
              'verbose':-1,'n_jobs':3,'seed':seed}
    if params_override: params.update(params_override)
    m = lgb.train(params, lgb.Dataset(X[tr_idx], label=y_tr_h), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X[es_idx], label=y_es_h)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return m.predict(X[es_idx]), m.predict(X[te_idx]), m.best_iteration

preds_es = {}; preds_te = {}
for H in [3, 5, 10, 15, 30]:
    es_list = []; te_list = []
    for seed in [42, 49, 56]:
        p_es, p_te, bi = train_one(H, seed)
        auc = roc_auc_score(all_labels[f'y_{H}'][es_idx], p_es)
        print(f'  H={H} seed={seed} best={bi} es_auc={auc:.4f}')
        es_list.append(p_es); te_list.append(p_te)
    preds_es[H] = np.mean(es_list, axis=0)
    preds_te[H] = np.mean(te_list, axis=0)

# ========= Combine via meta-LR on ES =========
print('\n[6] meta-learn on ES...', flush=True)
from sklearn.linear_model import LogisticRegression

# For each target H, combine predictions from multiple model Hs → predict target H
TARGET_H = 15
y_es_target = all_labels[f'y_{TARGET_H}'][es_idx]
y_te_target = all_labels[f'y_{TARGET_H}'][te_idx]

# Stack ES predictions as meta-features
X_meta_es = np.column_stack([preds_es[h] for h in preds_es])
X_meta_te = np.column_stack([preds_te[h] for h in preds_te])

meta_lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=3, solver='lbfgs')
meta_lr.fit(X_meta_es, y_es_target)
p_meta_es = meta_lr.predict_proba(X_meta_es)[:,1]
p_meta_te = meta_lr.predict_proba(X_meta_te)[:,1]
print(f'  meta weights: {dict(zip([f"H{h}" for h in preds_es], meta_lr.coef_[0].round(3)))}')
print(f'  meta_es_auc={roc_auc_score(y_es_target, p_meta_es):.4f}  meta_te_auc={roc_auc_score(y_te_target, p_meta_te):.4f}')

# Equal-weight baseline
p_eq_es = np.mean(list(preds_es.values()), axis=0)
p_eq_te = np.mean(list(preds_te.values()), axis=0)
print(f'  equal_es_auc={roc_auc_score(y_es_target, p_eq_es):.4f}  equal_te_auc={roc_auc_score(y_te_target, p_eq_te):.4f}')

# Use meta if better, else equal
if roc_auc_score(y_es_target, p_meta_es) > roc_auc_score(y_es_target, p_eq_es):
    p_te_final = p_meta_te; print('  → using meta-LR')
else:
    p_te_final = p_eq_te; print('  → using equal-weight')

p_es_final = p_eq_es  # for calibration below

# ========= Calibration =========
print('\n[7] isotonic calibration...', flush=True)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
iso.fit(p_es_final, y_es_target)
p_te_cal = iso.predict(p_te_final)

# ========= TOP-K =========
print(f'\n[8] TOP-K TEST (TARGET H={TARGET_H}, TE={len(te_idx):,})')
y_te_final = y_te_target
p_dict = {'raw_avg': p_eq_te, 'meta': p_meta_te, 'calibrated': p_te_cal}
for label, pv in p_dict.items():
    auc = roc_auc_score(y_te_final, pv)
    print(f'\n  --- {label}  auc={auc:.4f} ---')
    for pct in [0.5, 1, 2, 3, 5, 8, 10, 15]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te_final[idx].mean()*100
        tpd = k*1440/len(te_idx)
        print(f'  top{pct:>4}%: acc={acc:.2f}%  tpd={tpd:.1f}')

# ========= Per-H breakdown =========
print(f'\n[8b] PER-HORIZON top-1% (model H → predict same H):')
for H in [3, 5, 10, 15, 30]:
    y_h = all_labels[f'y_{H}'][te_idx]
    pv = preds_te[H]
    auc = roc_auc_score(y_h, pv)
    k1 = max(1, int(len(pv)*0.01))
    acc1 = y_h[np.argsort(-pv)[:k1]].mean()*100
    print(f'  model_H={H:>2}  predict_H={H:>2}  auc={auc:.4f}  top1pct={acc1:.2f}%  tpd={k1*1440/len(te_idx):.1f}')

# ========= Monthly stability =========
print(f'\n[9] MONTHLY STABILITY (best combos, top-1% per month)')
dt_te = pd.to_datetime(ts_orig[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))

# Compare different predictions
for label, pv in [('meta', p_meta_te), ('calibrated', p_te_cal), ('best_H10', preds_te[10]), ('best_H5', preds_te[5])]:
    print(f'\n  [{label}]  bad months:')
    bad = 0
    for m in all_months:
        mm = month == m
        n = mm.sum(); k = max(1, int(n*0.01))
        p_m = pv[mm]
        # Use appropriate y for this pv
        if label.startswith('best_H'):
            h_used = int(label.split('H')[1])
            y_m = all_labels[f'y_{h_used}'][te_idx][mm]
        else:
            y_m = y_te_final[mm]
        acc_m = y_m[np.argsort(-p_m)[:k]].mean()*100
        auc_m = roc_auc_score(y_m, p_m)
        flag = ' BAD' if acc_m < 60 else ''
        if acc_m < 60: bad += 1
        print(f'    {m} n={n:>6,} AUC={auc_m:.4f} top1pct={acc_m:.2f}%{flag}')
    print(f'    → BAD={bad}/{len(all_months)}')

# ========= Feature importance =========
print('\n[10] top 25 importance...')
m_imp = lgb.train({'objective':'binary','verbose':-1,'n_jobs':3,'seed':42},
                   lgb.Dataset(X[tr_idx][::10], label=all_labels[f'y_{TARGET_H}'][tr_idx][::10]),
                   num_boost_round=300, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                   valid_sets=[lgb.Dataset(X[es_idx], label=all_labels[f'y_{TARGET_H}'][es_idx])])
imp = pd.Series(m_imp.feature_importance(importance_type='gain'), index=ALL_NAMES).sort_values(ascending=False)
for i,(k,v) in enumerate(imp.head(25).items()):
    print(f'  {i+1:>2}. {k:<25s} gain={v:.0f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
