"""ENS_TRAIN: Use combined 110-feat dataset, LGB+XGB+CB, strict memory mgmt."""
import numpy as np, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

t0 = time.time()

# === Load (ONLY what's needed at each step) ===
print('[1] load dataset...', flush=True)
D = np.load('/workspace/models/eth_combined.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)[vi]
print(f'  X={X.shape}  mem={X.nbytes/1e9:.2f}GB')

# Labels
close_eth = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
N_full = len(np.load('/workspace/models/eth_data.npz', allow_pickle=True)['vi'])  # placeholder - not used
# Actually load close from raw
import pandas as pd
close_raw = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
y_all = {}
for h in [5,10,15,30]:
    rh = np.full(len(close_raw), np.nan, np.float32)
    rh[:-h] = (close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    y_all[h] = (rh[vi] > 0).astype(np.int8)
del close_raw; gc.collect()

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# === Pre-split X to save memory ===
print('\n[2] split...', flush=True)
X_tr = X[tr_idx]; X_es = X[es_idx]; X_te = X[te_idx]
del X; gc.collect()
print(f'  X_tr={X_tr.shape}  X_es={X_es.shape}  X_te={X_te.shape}')

# === TRAIN LGB (all horizons, then free) ===
print('\n[3] LGB training...', flush=True)
preds_es = {}; preds_te = {}
for H in [5,10,15,30]:
    print(f'  H={H}', flush=True)
    y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
    els=[]; tls=[]
    for seed in [42, 49, 56]:
        p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
             'verbose':-1,'n_jobs':3,'seed':seed}
        m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_es, y_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        els.append(m.predict(X_es))
        tls.append(m.predict(X_te))
        del m; gc.collect()
    preds_es[f'lgb_{H}'] = np.mean(els, axis=0)
    preds_te[f'lgb_{H}'] = np.mean(tls, axis=0)
    print(f'    te_auc={roc_auc_score(y_all[H][te_idx], preds_te[f"lgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# === XGB ===
print('\n[4] XGB training...', flush=True)
try:
    import xgboost as xgb
    for H in [15, 30]:
        print(f'  H={H}', flush=True)
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        els=[]; tls=[]
        for seed in [42, 56]:
            p = {'objective':'binary:logistic','eval_metric':'logloss','learning_rate':0.05,
                 'max_depth':8,'min_child_weight':200,'subsample':0.8,'colsample_bytree':0.8,
                 'reg_lambda':1.0,'tree_method':'hist','verbosity':0,'nthread':3,'seed':seed}
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_es, label=y_es)
            m = xgb.train(p, dtrain, num_boost_round=5000, evals=[(dval,'val')],
                          early_stopping_rounds=100, verbose_eval=False)
            els.append(m.predict(xgb.DMatrix(X_es)))
            tls.append(m.predict(xgb.DMatrix(X_te)))
            del m, dtrain, dval; gc.collect()
        preds_es[f'xgb_{H}'] = np.mean(els, axis=0)
        preds_te[f'xgb_{H}'] = np.mean(tls, axis=0)
        print(f'    te_auc={roc_auc_score(y_all[H][te_idx], preds_te[f"xgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'    XGB failed: {e}', flush=True)

# === CB ===
print('\n[5] CB training...', flush=True)
try:
    from catboost import CatBoostClassifier
    for H in [15, 30]:
        print(f'  H={H}', flush=True)
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                                 subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                                 verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
        cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
        preds_es[f'cb_{H}'] = cb.predict_proba(X_es)[:,1]
        preds_te[f'cb_{H}'] = cb.predict_proba(X_te)[:,1]
        del cb; gc.collect()
        print(f'    te_auc={roc_auc_score(y_all[H][te_idx], preds_te[f"cb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'    CB failed: {e}', flush=True)

# Free X
del X_tr, X_es, X_te; gc.collect()

# === COMBINE & EVAL ===
print('\n[6] COMBINE & EVAL...', flush=True)
keys = sorted(preds_es.keys())
X_meta_es = np.column_stack([preds_es[k] for k in keys])
X_meta_te = np.column_stack([preds_te[k] for k in keys])
p_eq_es = np.mean(list(preds_es.values()), axis=0)
p_eq_te = np.mean(list(preds_te.values()), axis=0)

for target_H in [15, 30]:
    y_es_t = y_all[target_H][es_idx]
    y_te_t = y_all[target_H][te_idx]
    
    meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs')
    meta.fit(X_meta_es, y_es_t)
    p_meta_te = meta.predict_proba(X_meta_te)[:,1]
    
    auc_m = roc_auc_score(y_te_t, p_meta_te)
    auc_e = roc_auc_score(y_te_t, p_eq_te)
    best_label = 'meta' if auc_m > auc_e else 'equal'
    p_best = p_meta_te if auc_m > auc_e else p_eq_te
    
    print(f'\n  === TARGET H={target_H}  [{best_label}] meta={auc_m:.4f} equal={auc_e:.4f} ===')
    daily = 1440
    line = f'  auc={roc_auc_score(y_te_t, p_best):.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        k = max(1, int(len(p_best)*pct/100))
        idx = np.argsort(-p_best)[:k]
        acc = y_te_t[idx].mean()*100; tpd = k*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)
    
    # Individual
    for k in sorted(preds_te):
        auc_k = roc_auc_score(y_te_t, preds_te[k])
        acc_k = y_te_t[np.argsort(-preds_te[k])[:max(1, int(len(preds_te[k])*0.01))]].mean()*100
        print(f'    {k:>12s}: auc={auc_k:.4f}  top1%={acc_k:.1f}%')
    
    # Monthly
    dt_te = pd.to_datetime(ts[te_idx], unit='s', utc=True)
    month = dt_te.to_period('M').values
    all_months = sorted(pd.PeriodIndex(np.unique(month)))
    bad = 0
    for m in all_months:
        mm = month == m
        n = mm.sum(); k = max(1, int(n*0.01))
        acc_m = y_te_t[mm][np.argsort(-p_best[mm])[:k]].mean()*100
        auc_mm = roc_auc_score(y_te_t[mm], p_best[mm])
        flag = 'BAD' if acc_m < 60 else '   '
        if acc_m < 60: bad += 1
        print(f'    {m} n={n:>6,} AUC={auc_mm:.4f} top1%={acc_m:>5.1f}% {flag}')
    print(f'    → BAD={bad}/{len(all_months)}')

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
