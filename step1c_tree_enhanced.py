"""Step 1C: ETH Tree-ENS with ENHANCED features (memory-efficient incremental rolling)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, datetime as dtm
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import config

t0 = time.time()

# ========== MEMORY-EFFICIENT ROLLING ==========
def rolling_sum(a, w):
    """Incremental rolling sum."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    s = np.cumsum(a, dtype=np.float64)
    out[w-1:] = (s[w-1:] - np.concatenate([[0], s[:-w]])).astype(np.float32)
    return out

def rolling_mean(a, w):
    s = np.full(len(a), np.nan, dtype=np.float32)
    cs = np.cumsum(a, dtype=np.float64)
    s[w-1:] = cs[w-1:] - np.concatenate([[0], cs[:-w]])
    return (s / w).astype(np.float32)

def rolling_std(a, w):
    """Rolling std via E[x^2] - E[x]^2."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    a64 = a.astype(np.float64)
    cs1 = np.cumsum(a64)
    cs2 = np.cumsum(a64 ** 2)
    s1 = cs1[w-1:] - np.concatenate([[0], cs1[:-w]])
    s2 = cs2[w-1:] - np.concatenate([[0], cs2[:-w]])
    mean = s1 / w
    var = (s2 / w) - mean ** 2
    out[w-1:] = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    return out

def rolling_min(a, w):
    """Sliding min using simple approach (ok for w<=240)."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    # For efficiency on 3.5M elements, use stride tricks only for small w
    if w <= 60:
        from numpy.lib.stride_tricks import sliding_window_view
        v = sliding_window_view(a.astype(np.float32), w)
        out[w-1:] = v.min(axis=1)
    else:
        # Incremental: run length encoded min
        for i in range(w-1, len(a), 100000):  # chunked
            lo = max(0, i - w + 1)
            hi = min(len(a), i + 100000)
            for j in range(lo + w - 1, hi):
                out[j] = np.min(a[j-w+1:j+1])
    return out

def rolling_max(a, w):
    out = np.full(len(a), np.nan, dtype=np.float32)
    if w <= 60:
        from numpy.lib.stride_tricks import sliding_window_view
        v = sliding_window_view(a.astype(np.float32), w)
        out[w-1:] = v.max(axis=1)
    else:
        for j in range(w-1, len(a)):
            out[j] = np.max(a[j-w+1:j+1])
    return out

# ========== LOAD RAW DATA ==========
print("Loading ETH raw...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
eth_ts = eth['ts'].to_numpy().astype(np.int64)
eth_C = eth['close'].to_numpy().astype(np.float32)
eth_BV = eth['buy_vol'].to_numpy().astype(np.float32)
eth_SV = eth['sell_vol'].to_numpy().astype(np.float32)
eth_F = eth['funding'].to_numpy().astype(np.float32)
del eth; gc.collect()

print("Loading BTC raw...", flush=True)
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
btc_ts = btc['ts'].to_numpy().astype(np.int64)
btc_C = btc['close'].to_numpy().astype(np.float32)
btc_BV = btc['buy_vol'].to_numpy().astype(np.float32)
btc_SV = btc['sell_vol'].to_numpy().astype(np.float32)
btc_F = btc['funding'].to_numpy().astype(np.float32)
del btc; gc.collect()

idx = np.searchsorted(btc_ts, eth_ts, side="right") - 1
idx = np.clip(idx, 0, len(btc_ts) - 1)
del btc_ts; gc.collect()
btc_C = btc_C[idx]; btc_BV = btc_BV[idx]; btc_SV = btc_SV[idx]; btc_F = btc_F[idx]

# ========== FEATURES ==========
print("Building features...", flush=True)
N = len(eth_ts)
feats = []
names = []

# Pre-compute returns
print("  Returns...", flush=True)
ret_1 = np.zeros(N, dtype=np.float64)
ret_1[1:] = np.log(np.maximum(eth_C[1:], 1e-8) / np.maximum(eth_C[:-1], 1e-8))
for h in [3, 5, 10, 15, 30, 60, 120, 240]:
    lr = np.zeros(N, dtype=np.float32)
    lr[h:] = np.log(np.maximum(eth_C[h:], 1e-8) / np.maximum(eth_C[:-h], 1e-8)).astype(np.float32)
    feats.append(lr); names.append(f'lr_{h}')

# Volatility
print("  Volatility...", flush=True)
abs_ret = np.abs(ret_1).astype(np.float64)
sq_ret = (ret_1 ** 2).astype(np.float64)
for w in [15, 30, 60, 120]:
    rv = np.sqrt(rolling_sum(abs_ret, w)).astype(np.float32)
    feats.append(rv); names.append(f'rvol_{w}')
for w in [30, 60, 120]:
    rv_a = np.sqrt(rolling_sum(abs_ret, w)) + 1e-8
    rv_b = np.sqrt(rolling_sum(sq_ret, w)) + 1e-8
    feats.append((rv_a / rv_b).astype(np.float32)); names.append(f'rvol_ratio_{w}')
del abs_ret, sq_ret, ret_1; gc.collect()

# Price position (z-score + percentile)
print("  Price position...", flush=True)
C64 = eth_C.astype(np.float64)
for w in [30, 60, 120, 240]:
    C_mean = rolling_mean(C64, w)
    C_std = rolling_std(C64, w) + 1e-6
    z = ((C64 - C_mean) / C_std).astype(np.float32)
    feats.append(z); names.append(f'z_{w}')
    # percentile: need min/max
    C_min = rolling_min(C64, w)
    C_max = rolling_max(C64, w)
    pos = ((C64 - C_min) / (C_max - C_min + 1e-6)).astype(np.float32)
    feats.append(pos); names.append(f'pos_{w}')
del C64; gc.collect()

# Orderflow
print("  Orderflow...", flush=True)
BV = eth_BV.astype(np.float64)
SV = eth_SV.astype(np.float64)
TV = BV + SV
TV_safe = np.where(TV > 0, TV, 1.0)
imb = ((BV - SV) / TV_safe).astype(np.float32)
feats.append(imb); names.append('imb_now')
for w in [10, 30, 60, 120]:
    imb_sum = rolling_sum(BV - SV, w)
    tv_sum = rolling_sum(TV, w) + 1e-6
    feats.append((imb_sum / tv_sum).astype(np.float32)); names.append(f'imb_{w}')
for w in [10, 30, 60]:
    tv_mean = rolling_mean(TV, w) + 1e-6
    feats.append((TV / tv_mean).astype(np.float32)); names.append(f'tv_ratio_{w}')
for w in [30, 60, 120]:
    cvd = rolling_sum(BV - SV, w)
    feats.append(cvd.astype(np.float32)); names.append(f'cvd_{w}')
for w in [30, 60]:
    cvd = rolling_sum(BV - SV, w)
    feats.append((cvd - np.roll(cvd, 1)).astype(np.float32)); names.append(f'cvd_d_{w}')
del BV, SV, TV, TV_safe; gc.collect()
del imb; gc.collect()

# Funding (ETH)
print("  Funding...", flush=True)
F = eth_F.astype(np.float64)
feats.append(F.astype(np.float32)); names.append('fund_now')
for w in [15, 30, 60, 120]:
    f_mean = rolling_mean(F, w)
    f_std = rolling_std(F, w) + 1e-6
    feats.append(f_mean.astype(np.float32)); names.append(f'fund_mean_{w}')
    feats.append(f_std.astype(np.float32)); names.append(f'fund_std_{w}')
    feats.append(((F - f_mean) / f_std).astype(np.float32)); names.append(f'fund_z_{w}')
F_d1 = np.zeros(N, dtype=np.float32)
F_d1[1:] = (F[1:] - F[:-1]).astype(np.float32)
feats.append(F_d1); names.append('fund_d1')
for w in [10, 30, 60]:
    fund_trend = rolling_sum(F_d1.astype(np.float64), w)
    feats.append(fund_trend.astype(np.float32)); names.append(f'fund_trend_{w}')
for w in [60, 120, 240]:
    f_min = rolling_min(F, w)
    f_max = rolling_max(F, w)
    feats.append(((F - f_min) / (f_max - f_min + 1e-6)).astype(np.float32)); names.append(f'fund_pos_{w}')
del F, F_d1; gc.collect()

# BTC cross-asset
print("  BTC cross-asset...", flush=True)
bC = btc_C.astype(np.float64)
for h in [5, 15, 60]:
    lr = np.zeros(N, dtype=np.float32)
    lr[h:] = np.log(np.maximum(btc_C[h:], 1e-8) / np.maximum(btc_C[:-h], 1e-8)).astype(np.float32)
    feats.append(lr); names.append(f'btc_lr_{h}')
del btc_C; gc.collect()

bF = btc_F.astype(np.float64)
feats.append(bF.astype(np.float32)); names.append('btc_fund_now')
for w in [30, 60, 120]:
    bf_mean = rolling_mean(bF, w)
    feats.append(bf_mean.astype(np.float32)); names.append(f'btc_fund_mean_{w}')
    bf_min = rolling_min(bF, w)
    bf_max = rolling_max(bF, w)
    feats.append(((bF - bf_min) / (bf_max - bf_min + 1e-6)).astype(np.float32)); names.append(f'btc_fund_pos_{w}')
del bF; gc.collect()

bBV = btc_BV.astype(np.float64)
bSV = btc_SV.astype(np.float64)
bTV = bBV + bSV
btc_imb = ((bBV - bSV) / np.where(bTV > 0, bTV, 1.0)).astype(np.float32)
feats.append(btc_imb); names.append('btc_imb_now')
for w in [30, 60]:
    imb_sum = rolling_sum(bBV - bSV, w)
    tv_sum = rolling_sum(bTV, w) + 1e-6
    feats.append((imb_sum / tv_sum).astype(np.float32)); names.append(f'btc_imb_{w}')
del bBV, bSV, bTV, btc_imb; gc.collect()

bC64 = bC
for w in [60, 120]:
    bC_mean = rolling_mean(bC64, w)
    bC_std = rolling_std(bC64, w) + 1e-6
    feats.append(((bC64 - bC_mean) / bC_std).astype(np.float32)); names.append(f'btc_z_{w}')
del bC64, btc_F, btc_C, btc_BV, btc_SV; gc.collect()
del eth_BV, eth_SV, eth_F; gc.collect()

# ========== ASSEMBLE ==========
print(f"\\nAssembling {len(feats)} features...", flush=True)
X_all = np.column_stack(feats).astype(np.float32)
del feats; gc.collect()
print(f"  X_all: {X_all.shape} ({X_all.nbytes/1e9:.1f}GB)", flush=True)

# ========== LABEL + SPLITS ==========
H = 15
label = (eth_C[H:] > eth_C[:-H]).astype(np.int64)
del eth_C; gc.collect()
ts_u = eth_ts[:-H]; del eth_ts; gc.collect()
X_all = X_all[:-H]

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_m = ts_u < tre
es_m = (ts_u >= tre) & (ts_u < es_end)
mv_m = (ts_u >= es_end) & (ts_u < meta_end)
te_m = ts_u >= meta_end
ts_te = ts_u[te_m]; del ts_u; gc.collect()

print("Splitting...", flush=True)
X_tr = X_all[tr_m].copy(); y_tr = label[tr_m]
X_es = X_all[es_m].copy(); y_es = label[es_m]
X_mv = X_all[mv_m].copy(); y_mv = label[mv_m]
X_te = X_all[te_m].copy(); y_te = label[te_m]
del X_all, label, tr_m, es_m, mv_m, te_m; gc.collect()

FT = X_tr.shape[1]
print(f"  tr={X_tr.shape} es={X_es.shape} mv={X_mv.shape} te={X_te.shape}", flush=True)

# ========== NORM ==========
print("Normalizing...", flush=True)
for j in range(FT):
    col = X_tr[:, j]; col[np.isnan(col)] = 0
    m, s = col.mean(), col.std() + 1e-6
    lo = np.percentile(col, 0.5); hi = np.percentile(col, 99.5)
    m2 = col[(col >= lo) & (col <= hi)].mean()
    s2 = col[(col >= lo) & (col <= hi)].std() + 1e-6
    X_tr[:, j] = np.clip((col - m2) / s2, -5, 5)
    X_es[:, j] = np.clip((np.nan_to_num(X_es[:, j], nan=0) - m2) / s2, -5, 5)
    X_mv[:, j] = np.clip((np.nan_to_num(X_mv[:, j], nan=0) - m2) / s2, -5, 5)
    X_te[:, j] = np.clip((np.nan_to_num(X_te[:, j], nan=0) - m2) / s2, -5, 5)
gc.collect()
print("  Done.", flush=True)

# ========== WEIGHTS ==========
pos = y_tr.mean()
sw = np.where(y_tr > 0.5, (1 - pos) / pos, pos / (1 - pos)).astype(np.float32)
del pos; gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

# ========== LGBM ==========
print("\nLGBM 5 seeds...", flush=True)
pvs_lgb_t = []; pvs_lgb_m = []
lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=3)
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    t0s = time.time()
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_lgb_t.append(bst.predict(X_te))
    pvs_lgb_m.append(bst.predict(X_mv))
    print(f"  s={s}: it={bst.best_iteration} [{time.time()-t0s:.0f}s]", flush=True)
    del bst, tr_ds, es_ds; gc.collect()

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_mv)
print(f"  LGBM: MV={roc_auc_score(y_mv, pv_lgb_mv):.4f} TE={roc_auc_score(y_te, pv_lgb_te):.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m; gc.collect()

# ========== CATBOOST ==========
print("\nCatBoost 5 seeds...", flush=True)
pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70]:
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3.0, random_seed=s, verbose=0,
                            early_stopping_rounds=200, eval_metric='AUC',
                            bagging_temperature=1.0, random_strength=0.5,
                            thread_count=3)
    t0s = time.time()
    cb.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], sample_weight=sw, use_best_model=True)
    pvs_cat_t.append(cb.predict_proba(X_te)[:, 1])
    pvs_cat_m.append(cb.predict_proba(X_mv)[:, 1])
    print(f"  s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_mv)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

# ========== TREE-ENS ==========
pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
print(f"\n{'='*50}")
print(f"★ TREE-ENS ({FT} enhanced feats):")
print(f"  MV AUC = {roc_auc_score(y_mv, pv_tree_mv):.4f}")
print(f"  TE AUC = {roc_auc_score(y_te, pv_tree_te):.4f}")
print(f"  Δ MV→TE = {(roc_auc_score(y_te, pv_tree_te) - roc_auc_score(y_mv, pv_tree_mv))*100:+.2f}%")
print(f"{'='*50}", flush=True)

# ========== QUICK EVAL ==========
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def rolling_eval(name, pv, y, ts, days):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400; day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1; window = 30
    print(f"\n  [{name}] AUC={roc_auc_score(y, pv):.4f}")
    print(f"    {'q':>6} {'acc':>8} {'tpd':>6} {'n':>7} {'flag':>10}")
    print(f"    {'-'*42}")
    for q in [96, 97, 98, 98.5, 99, 99.2, 99.5]:
        trades = []
        for d in range(window, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - window]; hist_hi = day_lo
            hist_mask = (ts_arr >= hist_lo) & (ts_arr < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100: continue
            today_mask = (ts_arr >= day_lo) & (ts_arr < day_hi)
            thr = np.percentile(hist_pv, q)
            pick = pv_arr[today_mask] >= thr
            if pick.sum() > 0:
                trades.extend(y_arr[today_mask][pick].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            flag = "★★★" if (tpd >= 14 and acc >= 65) else ("★★" if (tpd >= 14 and acc >= 60) else ("★" if (tpd >= 14 and acc >= 55) else ""))
            print(f"    {q:>6.1f} {acc:>7.1f}% {tpd:>6.1f} {len(trades):>7}  {flag}", flush=True)

rolling_eval("Tree-ENS ENHANCED", pv_tree_te, y_te, ts_te, DAYS)

# Global q=99
thr = np.percentile(pv_tree_te, 99)
pick = pv_tree_te >= thr
if pick.sum() > 0:
    g_acc = y_te[pick].mean() * 100; g_tpd = pick.sum() / DAYS
    print(f"\n  GLOBAL q=99: acc={g_acc:.1f}% tpd={g_tpd:.1f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/eth_tree_enhanced.npz',
         pv_tree_te=pv_tree_te, pv_tree_mv=pv_tree_mv,
         pv_lgb_te=pv_lgb_te, pv_cat_te=pv_cat_te,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te,
         n_features=FT)
print(f"\nSaved [{time.time()-t0:.0f}s]", flush=True)
