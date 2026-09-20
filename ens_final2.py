"""ENS_FINAL2: Focused — H=5,10,15 × LGB(3 seeds) + CatBoost, meta-learn, monthly."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

t0 = time.time()
print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)[vi]
print(f'  X={X.shape}')

# Labels
close_eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
N = len(close_eth)
y_all = {}
for h in [3,5,10,15,30]:
    rh = np.full(N, np.nan, np.float32)
    rh[:-h] = (close_eth[h:]/close_eth[:-h]-1).astype(np.float32)
    y_all[h] = (rh[vi] > 0).astype(np.int8)
del close_eth; gc.collect()

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# ===== TRAIN LGB =====
print('\n[2] LGB training...', flush=True)
preds_es = {}; preds_te = {}
for H in [3,5,10,15,30]:
    print(f'  H={H}...', flush=True)
    y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
    els = []; tls = []
    for seed in [42, 49, 56]:
        p = {'objective':'binary','metric':'binary_logloss','learning_rate':0.05,
             'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,
             'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
             'verbose':-1,'n_jobs':3,'seed':seed}
        m = lgb.train(p, lgb.Dataset(X[tr_idx], y_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X[es_idx], y_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        els.append(m.predict(X[es_idx]))
        tls.append(m.predict(X[te_idx]))
    preds_es[f'lgb_{H}'] = np.mean(els, axis=0)
    preds_te[f'lgb_{H}'] = np.mean(tls, axis=0)
    auc_e = roc_auc_score(y_es, preds_es[f'lgb_{H}'])
    auc_t = roc_auc_score(y_all[H][te_idx], preds_te[f'lgb_{H}'])
    print(f'    LGB avg  es_auc={auc_e:.4f}  te_auc={auc_t:.4f}  ({time.time()-t0:.0f}s)', flush=True)

# ===== CatBoost H=15 =====
print('\n[3] CatBoost H=15...', flush=True)
try:
    from catboost import CatBoostClassifier
    y15_tr = y_all[15][tr_idx]; y15_es = y_all[15][es_idx]; y15_te = y_all[15][te_idx]
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                             subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                             verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
    cb.fit(X[tr_idx], y15_tr, eval_set=(X[es_idx], y15_es), use_best_model=True)
    preds_es['cb_15'] = cb.predict_proba(X[es_idx])[:,1]
    preds_te['cb_15'] = cb.predict_proba(X[te_idx])[:,1]
    print(f'    CB best_iter={cb.get_best_iteration()}  te_auc={roc_auc_score(y15_te, preds_te["cb_15"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'    CB failed: {e}', flush=True)

del X; gc.collect()

# ===== COMBINE =====
print('\n[4] COMBINE...', flush=True)
TARGET_H = 15
y_es_t = y_all[TARGET_H][es_idx]
y_te_t = y_all[TARGET_H][te_idx]

# Meta-LR
keys_sorted = sorted(preds_es.keys())
X_meta_es = np.column_stack([preds_es[k] for k in keys_sorted])
X_meta_te = np.column_stack([preds_te[k] for k in keys_sorted])
meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs', n_jobs=3)
meta.fit(X_meta_es, y_es_t)
p_meta_es = meta.predict_proba(X_meta_es)[:,1]
p_meta_te = meta.predict_proba(X_meta_te)[:,1]
print(f'  meta coefs: {dict(zip(keys_sorted, meta.coef_[0].round(2)))}', flush=True)
print(f'  meta   auc  es={roc_auc_score(y_es_t, p_meta_es):.4f}  te={roc_auc_score(y_te_t, p_meta_te):.4f}')

# Equal weight
p_eq_es = np.mean(list(preds_es.values()), axis=0)
p_eq_te = np.mean(list(preds_te.values()), axis=0)
print(f'  equal  auc  es={roc_auc_score(y_es_t, p_eq_es):.4f}  te={roc_auc_score(y_te_t, p_eq_te):.4f}')

# Choose better on ES
if roc_auc_score(y_es_t, p_meta_es) > roc_auc_score(y_es_t, p_eq_es):
    label_best = 'meta'; p_es_best = p_meta_es; p_te_best = p_meta_te
else:
    label_best = 'equal'; p_es_best = p_eq_es; p_te_best = p_eq_te
print(f'  → best={label_best}')

# Isotonic calibration on ES
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
iso.fit(p_es_best, y_es_t)
p_te_cal = iso.predict(p_te_best)

# ===== TOP-K =====
print(f'\n[5] TOP-K (predict target H={TARGET_H}, TE={len(te_idx):,})')
daily = 1440
configs = [('meta', p_meta_te, y_te_t), ('equal', p_eq_te, y_te_t), ('calibrated', p_te_cal, y_te_t)]
# Also per-horizon best
for H in [3,5,10,15,30]:
    auc_h = roc_auc_score(y_all[H][te_idx], preds_te[f'lgb_{H}'])
    configs.append((f'lgb_avg_H{H}', preds_te[f'lgb_{H}'], y_all[H][te_idx]))

for label, pv, yv in configs:
    auc = roc_auc_score(yv, pv)
    line = f'\n  [{label:>15s}] auc={auc:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 10]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = yv[idx].mean()*100; tpd = k*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# ===== MONTHLY =====
print(f'\n[6] MONTHLY (top-1% per month, predict H=15)')
dt_te = pd.to_datetime(ts[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))

for label, pv, yv in [('meta', p_meta_te, y_te_t), ('equal', p_eq_te, y_te_t), ('best_lgb_H10', preds_te['lgb_10'], y_all[10][te_idx])]:
    print(f'\n  [{label}]')
    bad = 0
    for m in all_months:
        mm = month == m
        n = mm.sum(); k = max(1, int(n*0.01))
        acc_m = yv[mm][np.argsort(-pv[mm])[:k]].mean()*100
        auc_m = roc_auc_score(yv[mm], pv[mm])
        flag = 'BAD' if acc_m < 60 else '   '
        if acc_m < 60: bad += 1
        print(f'    {m} n={n:>6,} AUC={auc_m:.4f} top1%={acc_m:>5.1f}% {flag}')
    print(f'    → BAD={bad}/{len(all_months)}')

print(f'\n⏱ {time.time()-t0:.0f}s')
