"""V5c-part1: 只跑 Tree-ENS 全量训练, 保存结果. 不跑 NN."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import datetime as dtm, config

torch_seed=42; np.random.seed(42)
t0_all = time.time()

print("Loading data...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)

btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts = btc["ts"].to_numpy().astype(np.int64)
BTC_C = btc["close"].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C)-1)
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[idx[1:]],1e-8)/np.maximum(BTC_C[idx[:-1]],1e-8))
del BTC_ts, BTC_C; gc.collect()

feats_np = feats.to_numpy().astype(np.float32)
del feats, eth; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
ts_all_used = ts_all[:-H]
del feats_np, B_lr1, C_all, ts_all; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
mv_mask = (ts_all_used >= es_end) & (ts_all_used < meta_end)
te_mask = ts_all_used >= meta_end

tr_idx_all = np.where(tr_mask)[0]
X_tr = X_all[tr_idx_all].copy()
X_es = X_all[es_mask].copy()
X_mv = X_all[mv_mask].copy()
X_te = X_all[te_mask].copy()
y_tr = label[tr_idx_all]; r_tr = ret_future[tr_idx_all]
y_es = label[es_mask]; y_mv = label[mv_mask]; y_te = label[te_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

FT = X_tr.shape[1]
for j in range(FT):
    col = X_tr[:,j]; col[np.isnan(col)] = 0
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    m, s = col.mean(), col.std() + 1e-6
    X_tr[:,j] = np.clip((col - m) / s, -5, 5)
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()
print(f"  TR={X_tr.shape} TE={X_te.shape}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pos = y_tr.mean()
pw = np.where(y_tr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)

# LGBM
print("\nLGBM 5-seed full train...", flush=True)
lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_lgb_t = []; pvs_lgb_m = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(2000)])
    pvs_lgb_t.append(bst.predict(X_te)); pvs_lgb_m.append(bst.predict(X_mv))
    del bst; gc.collect()

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_m)
print(f"  LGBM: MV={roc_auc_score(y_mv, pv_lgb_mv):.4f} TE={roc_auc_score(y_te, pv_lgb_te):.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m; gc.collect()

# CatBoost
print("\nCatBoost 5-seed full train...", flush=True)
pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70]:
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3.0, random_seed=s, verbose=0,
                            early_stopping_rounds=200, eval_metric='AUC',
                            bagging_temperature=1.0, random_strength=0.5)
    t0s = time.time()
    cb.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], sample_weight=sw, use_best_model=True)
    pvs_cat_t.append(cb.predict_proba(X_te)[:,1])
    pvs_cat_m.append(cb.predict_proba(X_mv)[:,1])
    print(f"  s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
print(f"\n  ★ TREE-ENS: MV={roc_auc_score(y_mv, pv_tree_mv):.4f} TE={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/v5c_tree.npz',
         pv_lgb_te=pv_lgb_te, pv_cat_te=pv_cat_te, pv_tree_te=pv_tree_te,
         pv_lgb_mv=pv_lgb_mv, pv_cat_mv=pv_cat_mv, pv_tree_mv=pv_tree_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved v5c_tree.npz [{time.time()-t0_all:.0f}s]", flush=True)
