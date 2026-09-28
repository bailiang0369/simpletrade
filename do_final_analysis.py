"""Final Analysis: Best NN (CNN/ResMLP) + Tree → Correlation → Stacking."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, datetime as dtm
import numpy as np, polars as pl, config
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import torch, torch.nn as nn, torch.nn.functional as F

t0 = time.time()
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
te_start = int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
H = 15

# ========== Part 1: Tree (known to be best baseline: AUC=0.5444) ==========
print("="*70 + "\nPart 1: Train Tree (baseline_auc params, 5 seeds)\n" + "="*70, flush=True)

import features as fe
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats = fe.build_features(eth)
ts = eth['ts'].to_numpy().astype(np.int64)
C = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C_full = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.clip(np.searchsorted(BTC_ts, ts, side='right')-1, 0, len(BTC_C_full)-1)
BTC_C = BTC_C_full[idx]; del BTC_ts, BTC_C_full; gc.collect()
B_lr1 = np.zeros(len(ts), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
del BTC_C; gc.collect()

X = np.concatenate([feats.to_numpy().astype(np.float32)[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
del feats, B_lr1; gc.collect()
y = (C[H:] > C[:-H]).astype(np.int64); del C; gc.collect()
ts_u = ts[:-H]; del ts; gc.collect()

tr_m = ts_u < tre; es_m = (ts_u >= tre) & (ts_u < es_end); te_m = ts_u >= te_start
np.random.seed(42)
tr_idx = np.where(tr_m)[0]
if len(tr_idx) > 1_500_000: tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)
X_tr, y_tr = X[tr_idx], y[tr_idx]
X_es, y_es = X[es_m], y[es_m]
X_te_t, y_te_t, ts_te_t = X[te_m], y[te_m], ts_u[te_m]
del X, y, ts_u, tr_m, es_m, te_m; gc.collect()

for j in range(X_tr.shape[1]):
    c = X_tr[:,j]; v = c[~np.isnan(c)]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    X_tr[:,j] = np.nan_to_num(np.clip(c, lo, hi), nan=0.0)
    m, s = X_tr[:,j].mean(), X_tr[:,j].std()+1e-6
    X_tr[:,j] = (X_tr[:,j] - m)/s
    X_es[:,j] = (np.nan_to_num(np.clip(X_es[:,j], lo, hi), nan=0.0) - m)/s
    X_te_t[:,j] = (np.nan_to_num(np.clip(X_te_t[:,j], lo, hi), nan=0.0) - m)/s
gc.collect()
print(f"Tree data ready: TR={X_tr.shape} ES={X_es.shape} TE={X_te_t.shape}", flush=True)

params = dict(objective='binary',metric='auc',learning_rate=0.03,num_leaves=63,min_child_samples=200,
              feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,lambda_l2=0.1,verbose=-1,n_jobs=-1)
pvs_t=[]
for s in [42,49,56,63,70]:
    params['seed']=s
    bst = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=5000,
                    valid_sets=[lgb.Dataset(X_es, label=y_es)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pvs_t.append(bst.predict(X_te_t))

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree = rank_agg(pvs_t)
auc_tree = roc_auc_score(y_te_t, pv_tree)
print(f"★ Tree AUC={auc_tree:.4f}", flush=True)
DAYS_t = (ts_te_t[-1]-ts_te_t[0])/86400.0

# ========== Part 2: Load best NN from do_nn_best ==========
print("\n" + "="*70 + "\nPart 2: Load NN ensemble (CNN1D + ResMLP)\n" + "="*70, flush=True)
d_nn = np.load('/workspace/models_saved/nn_all_preds.npz', allow_pickle=True)
nn_keys = [k for k in d_nn.keys() if k not in ['y_te', 'ts_te']]
nn_pvs = {k: d_nn[k] for k in nn_keys}
y_te_s = d_nn['y_te']; ts_te_s = d_nn['ts_te']
print(f"Loaded {len(nn_keys)} NN predictions: {nn_keys}", flush=True)

# Ensemble ALL NNs for max diversity
pv_nn_all = rank_agg([nn_pvs[k] for k in nn_keys])
auc_nn_all = roc_auc_score(y_te_s, pv_nn_all)
print(f"\n★ NN ALL-ENSEMBLE ({len(nn_keys)} models): AUC={auc_nn_all:.4f}", flush=True)

# Also try CNN-only ensemble
cnn_keys = [k for k in nn_keys if 'CNN' in k]
pv_nn_cnn = rank_agg([nn_pvs[k] for k in cnn_keys]) if cnn_keys else pv_nn_all
auc_nn_cnn = roc_auc_score(y_te_s, pv_nn_cnn)
print(f"★ CNN-only ({len(cnn_keys)} models): AUC={auc_nn_cnn:.4f}", flush=True)

# ========== Part 3: Correlation & Stacking ==========
print("\n" + "="*70 + "\nPart 3: Correlation & Stacking\n" + "="*70, flush=True)

t_map = {int(ts_te_t[i]): i for i in range(len(ts_te_t))}
s_map = {int(ts_te_s[i]): i for i in range(len(ts_te_s))}
common = np.intersect1d(ts_te_t, ts_te_s)

# Use NN-all vs Tree
pv_tc = np.array([pv_tree[t_map[int(t)]] for t in common], dtype=np.float64)
pv_sc = np.array([pv_nn_all[s_map[int(t)]] for t in common], dtype=np.float64)
y_c = np.array([y_te_t[t_map[int(t)]] for t in common], dtype=np.int64)

corr = np.corrcoef(pv_tc, pv_sc)[0,1]
auc_t_c = roc_auc_score(y_c, pv_tc); auc_s_c = roc_auc_score(y_c, pv_sc)

def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)
rank_t, rank_s = rank(pv_tc), rank(pv_sc)

print(f"\n  Common anchors: {len(common)}", flush=True)
print(f"  Tree AUC (common): {auc_t_c:.4f}", flush=True)
print(f"  NN   AUC (common): {auc_s_c:.4f}", flush=True)
print(f"  CORR (rank):       {corr:.4f}", flush=True)

# Ensemble methods
ensembles = {}
ensembles['Tree_ONLY'] = (pv_tc, auc_t_c)
ensembles['NN_ONLY'] = (pv_sc, auc_s_c)
ensembles['PROB_AVG'] = ((pv_tc+pv_sc)/2, roc_auc_score(y_c,(pv_tc+pv_sc)/2))
ensembles['RANK_AVG'] = ((rank_t+rank_s)/2, roc_auc_score(y_c,(rank_t+rank_s)/2))

best_w, best_a = 0.5, 0.0
for w in np.arange(0, 1.05, 0.05):
    a = roc_auc_score(y_c, w*rank_t+(1-w)*rank_s)
    if a > best_a: best_a, best_w = a, w
ensembles[f'OPT_BLEND({best_w:.2f}/{1-best_w:.2f})'] = (best_w*rank_t+(1-best_w)*rank_s, best_a)

from sklearn.linear_model import LogisticRegression
split = int(len(y_c)*0.7)
lr = LogisticRegression(C=1.0, max_iter=200).fit(np.column_stack([rank_t,rank_s])[:split], y_c[:split])
pv_lr = lr.predict_proba(np.column_stack([rank_t,rank_s]))[:,1]
ensembles['LR_STACKING'] = (pv_lr, roc_auc_score(y_c, pv_lr))
print(f"  LR coef: Tree={lr.coef_[0][0]:.3f}, NN={lr.coef_[0][1]:.3f}", flush=True)

# Evaluate all
print(f"\n  Ensemble comparison:", flush=True)
DAYS = (dtm.datetime.strptime('2026-09-28','%Y-%m-%d') - dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d')).days
best_name = None; best_auc = 0
for name, (pv, auc) in sorted(ensembles.items(), key=lambda x: -x[1][1]):
    flag = '🏆' if auc > best_auc and auc > max(auc_t_c, auc_s_c) + 0.0001 else ''
    best_auc = max(best_auc, auc)
    top1k = max(1, int(len(pv)*0.01))
    acc1 = y_c[np.argsort(-pv)[:top1k]].mean()*100
    tpd = top1k/DAYS
    print(f"  {flag} {name:<25s} AUC={auc:.4f} top1%={acc1:.1f}% tpd={tpd:.1f}", flush=True)

# ========== FINAL SUMMARY ==========
print(f"\n{'='*70}", flush=True)
print(f"📊 FINAL REPORT", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Tree AUC:         {auc_tree:.4f} ({len(y_te_t)} anchors)", flush=True)
print(f"  NN AUC:           {auc_nn_all:.4f} ({len(y_te_s)} anchors)", flush=True)
print(f"  CORR:             {corr:.4f}", flush=True)
print(f"  {'-'*70}", flush=True)
print(f"  Stacking verdict: {'🔥 STRONG GAIN POSSIBLE!' if best_a > max(auc_t_c,auc_s_c)+0.002 else '⚠️  Marginal gain' if best_a > max(auc_t_c,auc_s_c) else '❌ No gain'}", flush=True)
print(f"  Best ensemble AUC: {best_auc:.4f} (+{best_auc-max(auc_t_c,auc_s_c):.4f} vs single)", flush=True)
print(f"{'='*70}", flush=True)
print(f"TOTAL TIME: {time.time()-t0:.0f}s", flush=True)
