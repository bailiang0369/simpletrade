"""ETH 1min -> H=15 direction, LightGBM, 严格时间切分, 月稳定性."""
import sys, time, datetime as dtm, numpy as np, pandas as pd, gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)[vi]
feat_names = D['feat_names']
H = 15
y_H = (D[f'ret_{H}'].astype(np.float32) > 0).astype(np.int8)
del D; gc.collect()

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
print(f'  TR={tr_mask.sum():,} ES={es_mask.sum():,} TE={te_mask.sum():,}')

tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]

params = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 200,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l2': 0.1, 'verbose': -1, 'n_jobs': 3, 'seed': 42,
}

print('[2] train LGB (H=15)...', flush=True)
t0 = time.time()
model = lgb.train(
    params, lgb.Dataset(X[tr_idx], label=y_H[tr_idx]),
    num_boost_round=5000,
    valid_sets=[lgb.Dataset(X[es_idx], label=y_H[es_idx])],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)
print(f'  best_iter={model.best_iteration}  {time.time()-t0:.0f}s')

p_te = model.predict(X[te_idx]); y_te = y_H[te_idx]; ts_te = ts[te_idx]
auc = roc_auc_score(y_te, p_te)
print(f'\n[3] test AUC={auc:.4f}')

for pct in [0.5, 1, 2, 3, 5, 8, 10]:
    k = max(1, int(len(p_te)*pct/100))
    idx = np.argsort(-p_te)[:k]
    acc = y_te[idx].mean()*100
    tpd = k * 1440 / len(te_idx)
    print(f'  top{pct:>4}%: acc={acc:.2f}%  tpd={tpd:.1f}')

# monthly
print('\n[4] monthly (per-month top-1pct):')
dt_te = pd.to_datetime(ts_te, unit='s', utc=True)
month = dt_te.to_period('M').values
bad = 0
all_months = sorted(pd.PeriodIndex(np.unique(month)))
for m in all_months:
    mm = month == m
    n = mm.sum(); k = max(1, int(n*0.01))
    p_m = p_te[mm]; y_m = y_te[mm]
    acc_m = y_m[np.argsort(-p_m)[:k]].mean()*100
    auc_m = roc_auc_score(y_m, p_m)
    flag = ' BAD' if acc_m < 60 else ''
    if acc_m < 60: bad += 1
    print(f'  {m} n={n:>6,} AUC={auc_m:.4f} top1pct_acc={acc_m:.2f} top1pct_n={k}{flag}')

print(f'\n  bad months={bad}/{len(all_months)}')

# importance
print('\n[5] top20 importance:')
imp = pd.Series(model.feature_importance(importance_type='gain'), index=feat_names).sort_values(ascending=False)
for i,(k,v) in enumerate(imp.head(20).items()):
    print(f'  {i+1:>2}. {k:<25s} gain={v:.0f}')

print(f'\n{time.time()-t0:.0f}s')
