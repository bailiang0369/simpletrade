"""Sanity check: LightGBM on flattened (C,W)→C*W seq features."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
d = np.load('/workspace/models_saved/seq_data_v4.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es = d['X_es'].astype(np.float32); y_es = d['y_es']
X_te = d['X_te'].astype(np.float32); y_te = d['y_te']; ts_te = d['ts_te']

# Flatten: (N, C, W) → (N, C*W)
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_f = X_es.reshape(len(X_es), -1)
X_te_f = X_te.reshape(len(X_te), -1)
FT = X_tr_f.shape[1]
print(f"Flattened feature dim: {FT}  TRAIN={X_tr_f.shape}  [{time.time()-t0:.1f}s]", flush=True)

# Train LightGBM
pos = y_tr.mean(); pw = np.where(y_tr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
sample_w = pw * rw

params = dict(
    objective='binary', metric='auc', learning_rate=0.05,
    num_leaves=63, min_child_samples=50, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=5, lambda_l1=0.0, lambda_l2=1.0,
    verbose=-1, seed=42, n_jobs=-1)

print(f"\n=== LightGBM on flattened seq features ===", flush=True)
tr_ds = lgb.Dataset(X_tr_f, label=y_tr, weight=sample_w)
es_ds = lgb.Dataset(X_es_f, label=y_es, reference=tr_ds)
bst = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)])
pv_te = bst.predict(X_te_f); pv_es = bst.predict(X_es_f)
auc_es = roc_auc_score(y_es, pv_es); auc_te = roc_auc_score(y_te, pv_te)
print(f"\n  ★ ES AUC={auc_es:.4f}  TE AUC={auc_te:.4f}  num_trees={bst.best_iteration}", flush=True)

# 5-seed ensemble
print(f"\n=== 5-seed rank ensemble ===", flush=True)
pvs_te = []; pvs_es = []
for s in [42, 49, 56, 63, 70]:
    params['seed'] = s
    tr_ds = lgb.Dataset(X_tr_f, label=y_tr, weight=sample_w)
    es_ds = lgb.Dataset(X_es_f, label=y_es, reference=tr_ds)
    bst = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_te.append(bst.predict(X_te_f)); pvs_es.append(bst.predict(X_es_f))

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(p)).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_ens_te = rank_agg(pvs_te); pv_ens_es = rank_agg(pvs_es)
auc_es2 = roc_auc_score(y_es, pv_ens_es); auc_te2 = roc_auc_score(y_te, pv_ens_te)
print(f"  ★ 5-seed ES={auc_es2:.4f}  TE={auc_te2:.4f}", flush=True)

# top-1% accuracy
DAYS = (ts_te[-1]-ts_te[0])/86400.0
for label, pv in [('single', pv_te), ('5-seed', pv_ens_te)]:
    print(f"\n  [{label}] top-% acc:", flush=True)
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y_te[np.argsort(-pv)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=60 else ''
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

print(f"\nTotal {time.time()-t0:.0f}s", flush=True)
