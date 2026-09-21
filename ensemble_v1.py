"""ENSEMBLE: H=15 direction. LGB + XGB + CatBoost, calibrated, monthly stability."""
import numpy as np, pandas as pd, time, gc, datetime as dtm, sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
ts = D['ts'].astype(np.int64)[D['vi']]
vi = D['vi'].astype(np.int64)
feat_names = D['feat_names']

H = 15
y_H = D[f'y_{H}'].astype(np.int8)

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)

tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')

tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}  feats={len(feat_names)}')
print(f'  y_te ↑={y_H[te_idx].mean():.4f}')

X_tr = X[tr_idx]; y_tr = y_H[tr_idx]
X_es = X[es_idx]; y_es = y_H[es_idx]
X_te = X[te_idx]; y_te = y_H[te_idx]
del X; gc.collect()

# ========= LGB =========
print('\n[2] LightGBM training...', flush=True)
t0 = time.time()
lgb_params = [
    {'objective':'binary','metric':'binary_logloss','learning_rate':0.03,'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':3,'seed':42},
    {'objective':'binary','metric':'binary_logloss','learning_rate':0.03,'num_leaves':31,'min_child_samples':300,'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5,'lambda_l2':0.5,'verbose':-1,'n_jobs':3,'seed':49},
    {'objective':'binary','metric':'binary_logloss','learning_rate':0.05,'num_leaves':127,'min_child_samples':150,'feature_fraction':0.9,'bagging_fraction':0.9,'bagging_freq':3,'lambda_l2':0.05,'verbose':-1,'n_jobs':3,'seed':56},
]
lgb_preds = []
for i, params in enumerate(lgb_params):
    m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X_es, label=y_es)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p_es = m.predict(X_es); p_te = m.predict(X_te)
    lgb_preds.append((p_es, p_te))
    auc = roc_auc_score(y_es, p_es)
    print(f'  LGB#{i} seed={params["seed"]} best_iter={m.best_iteration} es_auc={auc:.4f}  {time.time()-t0:.0f}s', flush=True)

# ========= CatBoost =========
print('\n[3] CatBoost training...', flush=True)
try:
    from catboost import CatBoostClassifier, Pool
    cb_model = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8,
                                     l2_leaf_reg=5, subsample=0.8, colsample_bylevel=0.8,
                                     od_type='Iter', od_wait=100, verbose=0, random_seed=42,
                                     thread_count=3, loss_function='Logloss')
    cb_model.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
    cb_es = cb_model.predict_proba(X_es)[:,1]
    cb_te = cb_model.predict_proba(X_te)[:,1]
    cb_auc = roc_auc_score(y_es, cb_es)
    print(f'  CatBoost best_iter={cb_model.get_best_iteration()} es_auc={cb_auc:.4f}  {time.time()-t0:.0f}s')
    have_cb = True
except Exception as e:
    print(f'  CatBoost skipped: {e}')
    cb_es = None; cb_te = None; have_cb = False

# ========= Simple average (out-of-fold on ES) =========
print('\n[4] ensemble simple avg...', flush=True)
es_avg = np.mean([p[0] for p in lgb_preds], axis=0)
te_avg = np.mean([p[1] for p in lgb_preds], axis=0)
if have_cb:
    es_avg = (es_avg*3 + cb_es)/4
    te_avg = (te_avg*3 + cb_te)/4

# ========= Calibration on ES set =========
print('[5] isotonic calibration on ES...', flush=True)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
iso.fit(es_avg, y_es)
te_cal = iso.predict(te_avg)
print(f'  LGB_avg es_auc={roc_auc_score(y_es, es_avg):.4f}')
print(f'  LGB_avg te_auc={roc_auc_score(y_te, te_avg):.4f}')
print(f'  calibrated te_auc={roc_auc_score(y_te, te_cal):.4f}')

# ========= Top-k analysis =========
print(f'\n[6] TOP-K (test, TE={len(te_idx):,})')
daily = 1440
p_te = te_cal  # calibrated
# also uncalibrated for comparison
p_te_raw = te_avg

for label, pv in [('calibrated', p_te), ('raw_avg', p_te_raw)]:
    print(f'\n  --- {label} ---')
    for pct in [0.5, 1, 2, 3, 5, 8, 10, 15]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean()*100
        tpd = k*daily/len(te_idx)
        print(f'  top{pct:>4}%: acc={acc:.2f}%  tpd={tpd:.1f}')

# ========= Month stability =========
print('\n[7] MONTHLY STABILITY (calibrated, per-month top-1%)')
dt_te = pd.to_datetime(ts[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))
bad = 0
for m in all_months:
    mm = month == m
    n = mm.sum(); k = max(1, int(n*0.01))
    p_m = p_te[mm]; y_m = y_te[mm]
    acc_m = y_m[np.argsort(-p_m)[:k]].mean()*100
    auc_m = roc_auc_score(y_m, p_m)
    flag = ' BAD' if acc_m < 60 else ''
    if acc_m < 60: bad += 1
    print(f'  {m} n={n:>6,} AUC={auc_m:.4f} top1pct={acc_m:.2f}%{flag}')
print(f'\n  BAD months: {bad}/{len(all_months)}')

# ========= Feature importance =========
print('\n[8] top importance (LGB#0):')
imp = pd.Series(lgb_params[0], index=['x'*100]*len(feat_names)) if False else None
# Re-train a single LGB to get importance quickly
import lightgbm as lgb2
m_imp = lgb2.train(lgb_params[0], lgb2.Dataset(X_tr, label=y_tr), num_boost_round=200,
                   valid_sets=[lgb2.Dataset(X_es, label=y_es)],
                   callbacks=[lgb2.early_stopping(50), lgb2.log_evaluation(0)])
imp = pd.Series(m_imp.feature_importance(importance_type='gain'), index=feat_names).sort_values(ascending=False)
for i,(k,v) in enumerate(imp.head(20).items()):
    print(f'  {i+1:>2}. {k:<25s} gain={v:.0f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
