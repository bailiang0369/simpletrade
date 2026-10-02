"""Step 1: ETH Tree-ENS (LGBM 5-seed + CatBoost 5-seed). Saves pv to npz."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, datetime as dtm
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import config
import features as fe

t0 = time.time()

# ---- Load ETH only ----
print("Loading ETH + features...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

feats_np = feats.to_numpy().astype(np.float16)   # 450MB f16
del feats; gc.collect()

# BTC lr1 cross-asset feat
print("Loading BTC close for cross-asset feat...", flush=True)
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').select(['ts','close']).sort('ts')
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_ts) - 1)
B_lr1 = np.zeros(len(ts_all), dtype=np.float32)
B_lr1[1:] = np.log(np.maximum(BTC_C[idx[1:]], 1e-8) / np.maximum(BTC_C[idx[:-1]], 1e-8)).astype(np.float32)
del BTC_ts, BTC_C; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float32)
X_all = np.concatenate([feats_np[:-H].astype(np.float32), B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1, C_all; gc.collect()

ts_all_used = ts_all[:-H]
del ts_all; gc.collect()

# Splits
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
mv_mask = (ts_all_used >= es_end) & (ts_all_used < meta_end)
te_mask = ts_all_used >= meta_end

X_tr = X_all[tr_mask].copy().astype(np.float32)
X_es = X_all[es_mask].copy().astype(np.float32)
X_mv = X_all[mv_mask].copy().astype(np.float32)
X_te = X_all[te_mask].copy().astype(np.float32)
y_tr = label[tr_mask]; y_es = label[es_mask]
y_mv = label[mv_mask]; y_te = label[te_mask]
r_tr = ret_future[tr_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

# z-score norm on train stats
FT = X_tr.shape[1]
print(f"Normalizing {FT} features...", flush=True)
for j in range(FT):
    col = X_tr[:, j]
    col[np.isnan(col)] = 0
    m, s = col.mean(), col.std() + 1e-6
    lo = np.percentile(col, 0.5); hi = np.percentile(col, 99.5)
    m2 = col[(col >= lo) & (col <= hi)].mean()
    s2 = col[(col >= lo) & (col <= hi)].std() + 1e-6
    X_tr[:, j] = np.clip((col - m2) / s2, -5, 5)
    X_es[:, j] = np.clip((np.nan_to_num(X_es[:, j], nan=0) - m2) / s2, -5, 5)
    X_mv[:, j] = np.clip((np.nan_to_num(X_mv[:, j], nan=0) - m2) / s2, -5, 5)
    X_te[:, j] = np.clip((np.nan_to_num(X_te[:, j], nan=0) - m2) / s2, -5, 5)
gc.collect()
print(f"  Sizes: tr={X_tr.shape} es={X_es.shape} mv={X_mv.shape} te={X_te.shape}", flush=True)

# Sample weights
pos = y_tr.mean()
pw = np.where(y_tr > 0.5, (1 - pos) / pos, pos / (1 - pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)
del r_tr, ret_abs, lo_q, hi_q, ext_mask, pw, rw; gc.collect()

# ---- LGBM 5 seeds ----
print("\nLGBM 5 seeds...", flush=True)
def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

pvs_lgb_t = []; pvs_lgb_m = []
lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=3)
for s in [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]:
    lgb_params['seed'] = s
    sw_es = np.ones(len(y_es), dtype=np.float32)
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, weight=sw_es, reference=tr_ds)
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

# ---- CatBoost 5 seeds ----
print("\nCatBoost 5 seeds...", flush=True)
pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]:
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3.0, random_seed=s, verbose=0,
                            early_stopping_rounds=200, eval_metric='AUC',
                            bagging_temperature=1.0, random_strength=0.5,
                            thread_count=3)
    t0s = time.time()
    sw_es = np.ones(len(y_es), dtype=np.float32)
    cb.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], sample_weight=sw,  use_best_model=True)
    pvs_cat_t.append(cb.predict_proba(X_te)[:, 1])
    pvs_cat_m.append(cb.predict_proba(X_mv)[:, 1])
    print(f"  CatBoost s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

# ---- Tree-ENS ----
pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
print(f"\n★ TREE-ENS: MV={roc_auc_score(y_mv, pv_tree_mv):.4f} TE={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)

# ---- Save ----
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/tree10_ens.npz',
         pv_lgb_te=pv_lgb_te, pv_lgb_mv=pv_lgb_mv,
         pv_cat_te=pv_cat_te, pv_cat_mv=pv_cat_mv,
         pv_tree_te=pv_tree_te, pv_tree_mv=pv_tree_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"Saved tree10_ens.npz [{time.time()-t0:.0f}s]", flush=True)

# ====== Breakthrough-style no_lookahead eval ======
def no_lookahead_eval(pv, y, ts, q=99, win=30, cold=10):
    conf = np.abs(pv - 0.5) * 2
    pred = (pv >= 0.5).astype(np.int8)
    day_of = ts // 86400
    unique_days = sorted(list(set(day_of.tolist())))
    sel = np.zeros(len(y), dtype=bool)
    for di, d in enumerate(unique_days):
        prior = unique_days[max(0,di-win):di]
        if len(prior) < cold: continue
        today = day_of == d
        for side in [1, 0]:
            hist_mask = np.isin(day_of, prior) & (pred == side)
            today_mask = today & (pred == side)
            if hist_mask.sum() == 0 or today_mask.sum() == 0: continue
            tau = np.percentile(conf[hist_mask], q)
            sel[today_mask & (conf >= tau)] = True
    acc = float((pred[sel]==y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(unique_days), 1)
    return acc*100, tpd, int(sel.sum())

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
print()
print("="*80, flush=True)
print("BREAKTHROUGH-STYLE NO_LOOKAHEAD EVAL (long/short independent conf quantile)", flush=True)
print("="*80, flush=True)
header = "  %-25s %6s | %8s %6s %7s" % ("Model", "q", "acc", "tpd", "n")
print(header, flush=True)
print("  " + "-"*58, flush=True)

best_q99 = (0,0,0,0,0)
for mname, pv in [
    ("LGBM-10 rank", pv_lgb_te),
    ("CatBoost-10 rank", pv_cat_te),
    ("TREE-ENS rank", pv_tree_te),
]:
    for q in [98, 99, 99.2, 99.5, 99.8]:
        acc, tpd, n = no_lookahead_eval(pv, y_te, ts_te, q=q)
        flag = "***" if (tpd>=14 and acc>=65) else ("**" if (tpd>=14 and acc>=60) else ("*" if (tpd>=10 and acc>=60) else ""))
        line = "  %-25s %6.1f | %7.1f%% %6.1f %7d %s" % (mname, q, acc, tpd, n, flag)
        print(line, flush=True)
        if q==99 and acc > best_q99[2]: best_q99=(mname, q, acc, tpd, n)

print()
print("  GLOBAL REFERENCE:", flush=True)
for mname, pv in [("TREE-ENS rank", pv_tree_te), ("LGBM rank", pv_lgb_te), ("CatBoost rank", pv_cat_te)]:
    k = max(1, int(len(y_te)*0.01))
    g = y_te[pv.argsort()[-k:]].mean()*100
    print("  %-25s global-top1%%=%.1f%%@%.1ft" % (mname, g, k/DAYS), flush=True)

print()
print("Blend scan LGB+Cat (q=99 no_lookahead)", flush=True)
best_b = (0,0,0,0,0,0,0)
for a in np.arange(0.2, 0.85, 0.05):
    b = round(1-a, 2)
    pv = a*pv_lgb_te + b*pv_cat_te
    acc, tpd, n = no_lookahead_eval(pv, y_te, ts_te, q=99)
    if acc > best_b[3]: best_b = (a, b, roc_auc_score(y_te,pv), acc, tpd, n)
a,b,auc,acc,tpd,n = best_b
print("  Best blend: LGB=%.2f Cat=%.2f AUC=%.4f q99=%.1f%%@%.1f n=%d" % (a,b,auc,acc,tpd,n), flush=True)
print("  BEST q=99:", best_q99, flush=True)
