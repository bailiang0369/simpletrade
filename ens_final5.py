"""ENS_FINAL5: Careful memory management — one model at a time, free after each."""
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
feat_names = list(D['feat_names'])
print(f'  X={X.shape}')

close_raw = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
y_all = {}
for h in [5,10,15,30]:
    rh = np.full(len(close_raw), np.nan, np.float32)
    rh[:-h] = (close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    y_all[h] = (rh[vi] > 0).astype(np.int8)

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_idx = np.where(ts_mask('2020-01-01','2024-06-30'))[0]
es_idx = np.where(ts_mask('2024-06-30','2024-09-30'))[0]
te_idx = np.where(ts_mask('2025-09-30','2026-08-29'))[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# Split once
X_tr = X[tr_idx]; X_es = X[es_idx]; X_te = X[te_idx]
del X; gc.collect()
print(f'  split: X_tr={X_tr.shape} mem={X_tr.nbytes/1e9:.2f}GB')

# ===== LGB all H =====
print('\n[2] LGB 62 feats...', flush=True)
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
        els.append(m.predict(X_es)); tls.append(m.predict(X_te)); del m; gc.collect()
    preds_es[f'lgb_{H}'] = np.mean(els, axis=0)
    preds_te[f'lgb_{H}'] = np.mean(tls, axis=0)
    print(f'    te={roc_auc_score(y_all[H][te_idx], preds_te[f"lgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# ===== Feature importance =====
print('\n[3] top feats...', flush=True)
m_imp = lgb.train({'objective':'binary','verbose':-1,'n_jobs':3,'seed':42},
                   lgb.Dataset(X_tr, y_all[15][tr_idx]), num_boost_round=500,
                   valid_sets=[lgb.Dataset(X_es, y_all[15][es_idx])],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
imps = m_imp.feature_importance(importance_type='gain')
top30 = np.argsort(-imps)[:30]
print('Top 10:', [(feat_names[i], int(imps[i])) for i in top30[:10]])
del m_imp; gc.collect()

# ===== CB H=15, H=30 =====
print('\n[4] CB...', flush=True)
try:
    from catboost import CatBoostClassifier
    for H in [15, 30]:
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                                 subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                                 verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
        cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
        preds_es[f'cb_{H}'] = cb.predict_proba(X_es)[:,1]
        preds_te[f'cb_{H}'] = cb.predict_proba(X_te)[:,1]
        del cb; gc.collect()
        print(f'  CB H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"cb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'  CB failed: {e}')

# ===== Now free X_tr, do XGB one H at a time =====
print('\n[5] XGB...', flush=True)
try:
    import xgboost as xgb
    del X_tr; gc.collect()  # FREE training data for XGB
    # Load only what XGB needs: X_tr_30 subset (less mem)
    X_tr_30 = (D['X'][:, top30]).astype(np.float32)[tr_idx]
    X_es_30 = (D['X'][:, top30]).astype(np.float32)[es_idx]
    X_te_30 = (D['X'][:, top30]).astype(np.float32)[te_idx]
    print(f'  X_tr_30={X_tr_30.shape}  mem={X_tr_30.nbytes/1e9:.2f}GB')
    
    for H in [15, 30]:
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        els=[]; tls=[]
        for seed in [42, 56]:
            p = {'objective':'binary:logistic','eval_metric':'logloss','learning_rate':0.05,
                 'max_depth':8,'min_child_weight':200,'subsample':0.8,'colsample_bytree':1.0,
                 'reg_lambda':1.0,'tree_method':'hist','verbosity':0,'nthread':3,'seed':seed}
            dtrain = xgb.DMatrix(X_tr_30, label=y_tr)
            dval = xgb.DMatrix(X_es_30, label=y_es)
            m = xgb.train(p, dtrain, num_boost_round=5000, evals=[(dval,'val')],
                          early_stopping_rounds=100, verbose_eval=False)
            els.append(m.predict(xgb.DMatrix(X_es_30)))
            tls.append(m.predict(xgb.DMatrix(X_te_30)))
            del m, dtrain, dval; gc.collect()
        preds_es[f'xgb30_{H}'] = np.mean(els, axis=0)
        preds_te[f'xgb30_{H}'] = np.mean(tls, axis=0)
        print(f'  XGB30 H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"xgb30_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
    
    del X_tr_30, X_es_30, X_te_30; gc.collect()
except Exception as e:
    print(f'  XGB failed: {e}')

del X_es, X_te; gc.collect()

# ===== EVAL =====
print('\n[6] EVAL...', flush=True)
daily = 1440
for target_H in [15, 30]:
    y_te = y_all[target_H][te_idx]
    y_es = y_all[target_H][es_idx]
    relevant = sorted([k for k in preds_te if k.endswith(f'_{target_H}')])
    
    print(f'\n  === H={target_H}  keys={relevant} ===')
    
    for k in relevant:
        pv = preds_te[k]
        auc = roc_auc_score(y_te, pv)
        line = f'  [{k:>12s}] auc={auc:.4f}'
        for pct in [0.5, 1, 2, 3, 5, 8, 10]:
            kk = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:kk]
            acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
            line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
        print(line)
    
    # Equal avg
    pv_avg = np.mean([preds_te[k] for k in relevant], axis=0)
    auc_a = roc_auc_score(y_te, pv_avg)
    line = f'  [{"avg":>12s}] auc={auc_a:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_avg)*pct/100))
        idx = np.argsort(-pv_avg)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)
    
    # Meta-LR
    Xm_es = np.column_stack([preds_es[k] for k in relevant])
    Xm_te = np.column_stack([preds_te[k] for k in relevant])
    meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs')
    meta.fit(Xm_es, y_es)
    pv_m = meta.predict_proba(Xm_te)[:,1]
    auc_m = roc_auc_score(y_te, pv_m)
    line = f'  [{"meta":>12s}] auc={auc_m:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_m)*pct/100))
        idx = np.argsort(-pv_m)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# Save
np.savez('/workspace/models/predictions_v5.npz', te_idx=te_idx, ts_te=ts[te_idx],
         y15=y_all[15][te_idx], y30=y_all[30][te_idx], **{f'p_{k}': v for k,v in preds_te.items()})
print(f'\n⏱ {time.time()-t0:.0f}s')
