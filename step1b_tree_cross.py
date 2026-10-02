"""Step 1B: ETH Tree-ENS + FULL BTC cross-asset features (134 total)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, datetime as dtm
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import config, features as fe

t0 = time.time()

# Load ETH timestamps + close
print("Loading ETH...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats_eth = fe.build_features(eth)
ts_eth = eth['ts'].to_numpy().astype(np.int64)
C_eth = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

# Load BTC + features
print("Loading BTC features...", flush=True)
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
feats_btc_df = fe.build_features(btc)
ts_btc = btc['ts'].to_numpy().astype(np.int64)
del btc; gc.collect()

feats_eth_np = feats_eth.to_numpy().astype(np.float16)
del feats_eth; gc.collect()
feats_btc_np = feats_btc_df.to_numpy().astype(np.float16)
del feats_btc_df; gc.collect()

# Align BTC feats to ETH timestamps
idx = np.searchsorted(ts_btc, ts_eth, side="right") - 1
idx = np.clip(idx, 0, len(ts_btc) - 1)
del ts_btc; gc.collect()
feats_btc_aligned = feats_btc_np[idx]
del feats_btc_np, idx; gc.collect()

# Combine (keep only ETH timestamps)
X_all = np.concatenate([feats_eth_np, feats_btc_aligned], axis=1)
del feats_eth_np, feats_btc_aligned; gc.collect()
print(f"  Combined feats shape: {X_all.shape}", flush=True)

# Label
H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int64)
ret_future = (C_eth[H:] / C_eth[:-H] - 1).astype(np.float32)
del C_eth; gc.collect()
ts_u = ts_eth[:-H]
del ts_eth; gc.collect()
X_all = X_all[:-H]

# Splits
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_m = ts_u < tre
es_m = (ts_u >= tre) & (ts_u < es_end)
mv_m = (ts_u >= es_end) & (ts_u < meta_end)
te_m = ts_u >= meta_end
ts_te = ts_u[te_m]
del ts_u; gc.collect()

print("Splitting...", flush=True)
X_tr = X_all[tr_m].copy().astype(np.float32)
X_es = X_all[es_m].copy().astype(np.float32)
X_mv = X_all[mv_m].copy().astype(np.float32)
X_te = X_all[te_m].copy().astype(np.float32)
y_tr = label[tr_m]; y_es = label[es_m]
y_mv = label[mv_m]; y_te = label[te_m]
r_tr = ret_future[tr_m]
del X_all, label, ret_future, tr_m, es_m, mv_m, te_m; gc.collect()

FT = X_tr.shape[1]
print(f"  tr={X_tr.shape} es={X_es.shape} mv={X_mv.shape} te={X_te.shape}", flush=True)

# Norm
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

# Sample weights
pos = y_tr.mean()
pw = np.where(y_tr > 0.5, (1 - pos) / pos, pos / (1 - pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)
del r_tr, ret_abs, lo_q, hi_q, ext_mask, pw, rw; gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

# LGBM 5 seeds
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
    print(f"  LGBM s={s}: best_iter={bst.best_iteration} [{time.time()-t0s:.0f}s]", flush=True)
    del bst, tr_ds, es_ds; gc.collect()

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_m)
print(f"  LGBM: MV={roc_auc_score(y_mv, pv_lgb_mv):.4f} TE={roc_auc_score(y_te, pv_lgb_te):.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m; gc.collect()

# CatBoost 5 seeds
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
    print(f"  CatBoost s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

# Tree-ENS
pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
print(f"\n★ TREE-ENS (134 feats): MV={roc_auc_score(y_mv, pv_tree_mv):.4f} TE={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)

# Quick rolling eval
print("\nQuick rolling eval...", flush=True)
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
def rolling_eval(pv, y, ts, days, q_list=[97, 98, 98.5, 99, 99.5]):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400; day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1; window = 30
    for q in q_list:
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
            print(f"  q={q:>5.1f}: acc={acc:>5.1f}% tpd={tpd:>5.1f} n={len(trades):>7} {flag}", flush=True)

for name, pv in [('TREE-ENS 134f', pv_tree_te), ('LGBM 134f', pv_lgb_te), ('CatBoost 134f', pv_cat_te)]:
    print(f"  [{name}]", flush=True)
    rolling_eval(pv, y_te, ts_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/eth_tree_fullbtc.npz',
         pv_tree_te=pv_tree_te, pv_tree_mv=pv_tree_mv,
         pv_lgb_te=pv_lgb_te, pv_lgb_mv=pv_lgb_mv,
         pv_cat_te=pv_cat_te, pv_cat_mv=pv_cat_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved eth_tree_fullbtc.npz [{time.time()-t0:.0f}s]", flush=True)
