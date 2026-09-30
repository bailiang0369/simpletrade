"""Part 1: Tree 模型训练 + 保存预测"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np, datetime as dtm
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import config, features as fe

t0 = time.time()

# 0. 加载数据
print("[0] 加载数据...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)

BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C_full = btc['close'].to_numpy().astype(np.float64)
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C_full)-1)
BTC_C_aligned = BTC_C_full[idx]
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C_aligned[1:],1e-8)/np.maximum(BTC_C_aligned[:-1],1e-8))
del BTC_C_full, BTC_C_aligned, BTC_ts, btc; gc.collect()

feats_np = feats.to_numpy().astype(np.float32)
del feats, eth; gc.collect()
X_all = np.concatenate([feats_np, B_lr1[:, np.newaxis].astype(np.float32)], axis=1)
del feats_np, B_lr1; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = X_all[:-H]
ts_all_used = ts_all[:-H]
del C_all, ts_all; gc.collect()

def ts_mask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all_used >= a) & (ts_all_used < b)

tr_idx = np.where(ts_mask(*config.SPLITS['train']))[0]
es_idx = np.where(ts_mask(*config.SPLITS['early_stop']))[0]
mv_idx = np.where(ts_mask(*config.SPLITS['meta_val']))[0]
te_idx = np.where(ts_mask(*config.SPLITS['test']))[0]

# 子采样 train → 1.5M
np.random.seed(42)
if len(tr_idx) > 1_500_000:
    tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)

X_tr = X_all[tr_idx].astype(np.float32)
y_tr = label[tr_idx]
r_tr = ret_future[tr_idx]
X_es = X_all[es_idx].astype(np.float32)
y_es = label[es_idx]
X_mv = X_all[mv_idx].astype(np.float32)
y_mv = label[mv_idx]
X_te = X_all[te_idx].astype(np.float32)
y_te = label[te_idx]
ts_te = ts_all_used[te_idx]
del X_all, label, ret_future, ts_all_used; gc.collect()

# Robust z-score
print("  z-score...", flush=True)
for j in range(X_tr.shape[1]):
    col = X_tr[:,j]
    lo, hi = np.percentile(col[~np.isnan(col)], 0.5), np.percentile(col[~np.isnan(col)], 99.5)
    m, s = np.nanmean(col), np.nanstd(col) + 1e-6
    X_tr[:,j] = np.nan_to_num((np.clip(col, lo, hi) - m) / s, nan=0.0)
    X_es[:,j] = np.nan_to_num((np.clip(X_es[:,j], lo, hi) - m) / s, nan=0.0)
    X_mv[:,j] = np.nan_to_num((np.clip(X_mv[:,j], lo, hi) - m) / s, nan=0.0)
    X_te[:,j] = np.nan_to_num((np.clip(X_te[:,j], lo, hi) - m) / s, nan=0.0)

print(f"  TR={X_tr.shape} TE={X_te.shape}", flush=True)
gc.collect()

# 1. Tree 训练
print("\n[1] Tree LGBM 5-seed + 负权重", flush=True)
abs_ret = np.abs(r_tr)
q90 = np.quantile(abs_ret, 0.90)
q10 = np.quantile(abs_ret, 0.10)
w_tree = np.where((abs_ret >= q90) | (abs_ret <= q10), 0.3, 1.0).astype(np.float32)
print(f"  负权重比例={((abs_ret >= q90) | (abs_ret <= q10)).mean()*100:.1f}%", flush=True)

lgb_params = dict(
    objective='binary', metric='auc', learning_rate=0.05,
    num_leaves=127, max_depth=-1, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
    lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,
    verbose=-1, n_jobs=-1,
)

pvs_tree_es = []; pvs_tree_mv = []; pvs_tree_te = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tree)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_tree_es.append(bst.predict(X_es))
    pvs_tree_mv.append(bst.predict(X_mv))
    pvs_tree_te.append(bst.predict(X_te))
    print(f"  seed{s}: TE AUC={roc_auc_score(y_te, pvs_tree_te[-1]):.4f} iters={bst.best_iteration}", flush=True)
    del bst, tr_ds, es_ds; gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree_es = rank_agg(pvs_tree_es)
pv_tree_mv = rank_agg(pvs_tree_mv)
pv_tree_te = rank_agg(pvs_tree_te)
print(f"\n  ★ TREE: MV AUC={roc_auc_score(y_mv, pv_tree_mv):.4f} TE AUC={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)
del pvs_tree_es, pvs_tree_mv, pvs_tree_te; gc.collect()

# 保存 Tree 预测 + 数据切分
print("\n[2] 保存 Tree 预测 + 数据", flush=True)
os.makedirs('/workspace/models_saved', exist_ok=True)
np.savez('/workspace/models_saved/tree_preds.npz',
    pv_tree_es=pv_tree_es, pv_tree_mv=pv_tree_mv, pv_tree_te=pv_tree_te,
    y_es=y_es, y_mv=y_mv, y_te=y_te, ts_te=ts_te,
    X_tr=X_tr, X_es=X_es, X_mv=X_mv, X_te=X_te,
    y_tr=y_tr)
print(f"  已保存 → /workspace/models_saved/tree_preds.npz", flush=True)

# 打印 Tree top-k 结果
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
    k = max(1, int(len(pv_tree_te)*pct/100))
    acc = y_te[np.argsort(-pv_tree_te)[:k]].mean()*100; tpd = k/DAYS
    print(f"  TREE top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

print(f"\n⏱ Tree 训练耗时: {time.time()-t0:.0f}s", flush=True)
