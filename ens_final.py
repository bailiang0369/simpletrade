"""ENS_FINAL: Use existing 62-feat dataset, multi-seed x multi-horizon, meta-learn, monthly stability."""
import numpy as np, pandas as pd, time, gc, datetime as dtm, sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
t0 = time.time()

print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts_orig = D['ts'].astype(np.int64)[vi]
feat_base = D['feat_names']
print(f'  X={X.shape}  feats={X.shape[1]}')

# Labels
print('[2] labels...', flush=True)
close_eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
N_full = len(close_eth)
all_labels = {}
for h in [3,5,10,15,30]:
    rh = np.full(N_full, np.nan, np.float32)
    rh[:-h] = (close_eth[h:] / close_eth[:-h] - 1).astype(np.float32)
    all_labels[f'y_{h}'] = (rh[vi] > 0).astype(np.int8)
del close_eth; gc.collect()

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_orig>=a)&(ts_orig<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# ========= Multi-seed multi-horizon training =========
print('\n[3] TRAINING (H x seed)...', flush=True)
preds_es = {}; preds_te = {}
for H in [3, 5, 10, 15, 30]:
    print(f'  H={H}...', flush=True)
    y_tr_h = all_labels[f'y_{H}'][tr_idx]
    y_es_h = all_labels[f'y_{H}'][es_idx]
    es_list = []; te_list = []
    for seed in [42, 49, 56, 63, 70]:
        params = {'objective':'binary','metric':'binary_logloss','learning_rate':0.05,
                  'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,
                  'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                  'verbose':-1,'n_jobs':3,'seed':seed}
        m = lgb.train(params, lgb.Dataset(X[tr_idx], label=y_tr_h), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X[es_idx], label=y_es_h)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        es_list.append(m.predict(X[es_idx]))
        te_list.append(m.predict(X[te_idx]))
    preds_es[H] = np.mean(es_list, axis=0)
    preds_te[H] = np.mean(te_list, axis=0)
    auc_es = roc_auc_score(y_es_h, preds_es[H])
    auc_te = roc_auc_score(all_labels[f'y_{H}'][te_idx], preds_te[H])
    print(f'    avg seeds: es_auc={auc_es:.4f}  te_auc={auc_te:.4f}')
del X; gc.collect()

# ========= Combine =========
print('\n[4] COMBINE (meta-LR on ES)...', flush=True)
TARGET_H = 15
y_es_target = all_labels[f'y_{TARGET_H}'][es_idx]
y_te_target = all_labels[f'y_{TARGET_H}'][te_idx]

# Meta-LR: use all horizon predictions to predict H=TARGET_H
X_meta_es = np.column_stack([preds_es[h] for h in sorted(preds_es)])
X_meta_te = np.column_stack([preds_te[h] for h in sorted(preds_te)])

meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs', n_jobs=3)
meta.fit(X_meta_es, y_es_target)
p_meta_es = meta.predict_proba(X_meta_es)[:,1]
p_meta_te = meta.predict_proba(X_meta_te)[:,1]
print(f'  meta coef: {dict(zip(sorted(preds_es), meta.coef_[0].round(3)))}')
print(f'  meta es_auc={roc_auc_score(y_es_target, p_meta_es):.4f}  te_auc={roc_auc_score(y_te_target, p_meta_te):.4f}')

# Equal weight
p_eq_es = np.mean([preds_es[h] for h in preds_es], axis=0)
p_eq_te = np.mean([preds_te[h] for h in preds_te], axis=0)
print(f'  equal es_auc={roc_auc_score(y_es_target, p_eq_es):.4f}  te_auc={roc_auc_score(y_te_target, p_eq_te):.4f}')

# Also individual best H
best_h_te_auc = {h: roc_auc_score(all_labels[f'y_{h}'][te_idx], preds_te[h]) for h in preds_te}
best_h = max(best_h_te_auc, key=best_h_te_auc.get)
print(f'  single best H={best_h} te_auc={best_h_te_auc[best_h]:.4f}')

# Choose best approach on ES
p_es_candidates = {'meta': p_meta_es, 'equal': p_eq_es}
best_label = max(p_es_candidates, key=lambda k: roc_auc_score(y_es_target, p_es_candidates[k]))
print(f'  → using: {best_label}')
if best_label == 'meta':
    p_te_final = p_meta_te; p_es_final = p_meta_es
else:
    p_te_final = p_eq_te; p_es_final = p_eq_es

# ========= Isotonic calibration =========
print('\n[5] isotonic calibration...', flush=True)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
iso.fit(p_es_final, y_es_target)
p_te_cal = iso.predict(p_te_final)
p_es_cal = iso.predict(p_es_final)
print(f'  cal es_auc={roc_auc_score(y_es_target, p_es_cal):.4f}  te_auc={roc_auc_score(y_te_target, p_te_cal):.4f}')

# ========= TOP-K =========
print(f'\n[6] TOP-K TEST (target H={TARGET_H})')
y_te = y_te_target
labels = ['meta', 'equal', 'calibrated', f'best_single_H{best_h}']
preds = [p_meta_te, p_eq_te, p_te_cal, preds_te[best_h]]
ys = [y_te, y_te, y_te, all_labels[f'y_{best_h}'][te_idx]]

daily = 1440
for label, pv, yv in zip(labels, preds, ys):
    auc = roc_auc_score(yv, pv)
    print(f'\n  --- {label}  auc={auc:.4f} ---')
    for pct in [0.5, 1, 2, 3, 5, 8, 10, 15]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = yv[idx].mean()*100
        tpd = k*daily/len(te_idx)
        print(f'  top{pct:>4}%: acc={acc:.2f}%  tpd={tpd:.1f}')

# ========= Monthly stability =========
print(f'\n[7] MONTHLY STABILITY (top-1% per month)')
dt_te = pd.to_datetime(ts_orig[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))

for label, pv, yv in zip(labels, preds, ys):
    print(f'\n  [{label}]')
    bad = 0
    for m in all_months:
        mm = month == m
        n = mm.sum(); k = max(1, int(n*0.01))
        acc_m = yv[mm][np.argsort(-pv[mm])[:k]].mean()*100
        auc_m = roc_auc_score(yv[mm], pv[mm])
        flag = ' BAD' if acc_m < 60 else ''
        if acc_m < 60: bad += 1
        print(f'    {m} n={n:>6,} AUC={auc_m:.4f} top1pct={acc_m:.2f}%{flag}')
    print(f'    BAD months: {bad}/{len(all_months)}')

# ========= Feature importance =========
print('\n[8] top 25 feat importance...')
m_imp = lgb.train({'objective':'binary','verbose':-1,'n_jobs':3,'seed':42},
                   lgb.Dataset(lgb.Dataset.load('/workspace/models/eth_data.npz', column='X').data[np.where(tr_mask)[0][::10]],
                               label=all_labels['y_15'][tr_mask][::10]),
                   num_boost_round=500, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                   valid_sets=[lgb.Dataset(lgb.Dataset.load('/workspace/models/eth_data.npz', column='X').data[te_idx[:10000]],
                                           label=all_labels['y_15'][te_idx[:10000]])])
