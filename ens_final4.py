"""ENS_FINAL4: 62 feats (proven) + LGB+XGB+CB + feat importance + per-hour models."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

t0 = time.time()
print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)[vi]
feat_names = list(D['feat_names'])
print(f'  X={X.shape}  ({X.shape[1]} feats)')

# Labels
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
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# Split
X_tr = X[tr_idx]; X_es = X[es_idx]; X_te = X[te_idx]
y15_tr = y_all[15][tr_idx]; y15_es = y_all[15][es_idx]; y15_te = y_all[15][te_idx]
y30_tr = y_all[30][tr_idx]; y30_es = y_all[30][es_idx]; y30_te = y_all[30][te_idx]

# ===== Feature importance (from LGB H=15 single seed) =====
print('\n[2] FEATURE IMPORTANCE...', flush=True)
m_imp = lgb.train({'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
                    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                    'verbose':-1,'n_jobs':3,'seed':42},
                   lgb.Dataset(X_tr, y15_tr), num_boost_round=500,
                   valid_sets=[lgb.Dataset(X_es, y15_es)],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
imps = m_imp.feature_importance(importance_type='gain')
top_order = np.argsort(-imps)
print('Top 30 feats by gain:')
for i in top_order[:30]:
    print(f'  {i:>2d} {feat_names[i]:>18s}  gain={imps[i]:>10.0f}')
top30 = top_order[:30]
X_tr_30 = X_tr[:, top30]; X_es_30 = X_es[:, top30]; X_te_30 = X_te[:, top30]
print(f'  → pruning to top-30 feats')

del m_imp; gc.collect()

# ===== Phase 1: Train on ALL 62 feats =====
print('\n[3] LGB on 62 feats...', flush=True)
all_preds_es = {}; all_preds_te = {}
for H_key, yt_tr, yt_es, yt_te in [('15', y15_tr, y15_es, y15_te), ('30', y30_tr, y30_es, y30_te)]:
    els=[]; tls=[]
    for seed in [42, 49, 56]:
        p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
             'verbose':-1,'n_jobs':3,'seed':seed}
        m = lgb.train(p, lgb.Dataset(X_tr, yt_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_es, yt_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        els.append(m.predict(X_es)); tls.append(m.predict(X_te)); del m; gc.collect()
    all_preds_es[f'lgb62_{H_key}'] = np.mean(els, axis=0)
    all_preds_te[f'lgb62_{H_key}'] = np.mean(tls, axis=0)
    print(f'  lgb62 H={H_key} te_auc={roc_auc_score(yt_te, all_preds_te[f"lgb62_{H_key}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# XGB
print('[4] XGB on 62 feats...', flush=True)
for H_key, yt_tr, yt_es, yt_te in [('15', y15_tr, y15_es, y15_te), ('30', y30_tr, y30_es, y30_te)]:
    els=[]; tls=[]
    for seed in [42, 56]:
        p = {'objective':'binary:logistic','eval_metric':'logloss','learning_rate':0.05,
             'max_depth':8,'min_child_weight':200,'subsample':0.8,'colsample_bytree':0.8,
             'reg_lambda':1.0,'tree_method':'hist','verbosity':0,'nthread':3,'seed':seed}
        dtrain = xgb.DMatrix(X_tr, label=yt_tr); dval = xgb.DMatrix(X_es, label=yt_es)
        m = xgb.train(p, dtrain, num_boost_round=5000, evals=[(dval,'val')],
                      early_stopping_rounds=100, verbose_eval=False)
        els.append(m.predict(xgb.DMatrix(X_es))); tls.append(m.predict(xgb.DMatrix(X_te)))
        del m, dtrain, dval; gc.collect()
    all_preds_es[f'xgb62_{H_key}'] = np.mean(els, axis=0)
    all_preds_te[f'xgb62_{H_key}'] = np.mean(tls, axis=0)
    print(f'  xgb62 H={H_key} te_auc={roc_auc_score(yt_te, all_preds_te[f"xgb62_{H_key}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# CB
print('[5] CB on 62 feats...', flush=True)
try:
    from catboost import CatBoostClassifier
    for H_key, yt_tr, yt_es, yt_te in [('15', y15_tr, y15_es, y15_te), ('30', y30_tr, y30_es, y30_te)]:
        cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                                 subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                                 verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
        cb.fit(X_tr, yt_tr, eval_set=(X_es, yt_es), use_best_model=True)
        all_preds_es[f'cb62_{H_key}'] = cb.predict_proba(X_es)[:,1]
        all_preds_te[f'cb62_{H_key}'] = cb.predict_proba(X_te)[:,1]
        del cb; gc.collect()
        print(f'  cb62 H={H_key} te_auc={roc_auc_score(yt_te, all_preds_te[f"cb62_{H_key}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'  CB failed: {e}')

# ===== Phase 2: Train on TOP-30 feats =====
print('\n[6] LGB on top-30 feats...', flush=True)
for H_key, yt_tr, yt_es, yt_te in [('15', y15_tr, y15_es, y15_te), ('30', y30_tr, y30_es, y30_te)]:
    els=[]; tls=[]
    for seed in [42, 49, 56]:
        p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
             'feature_fraction':1.0,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
             'verbose':-1,'n_jobs':3,'seed':seed}
        m = lgb.train(p, lgb.Dataset(X_tr_30, yt_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_es_30, yt_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        els.append(m.predict(X_es_30)); tls.append(m.predict(X_te_30)); del m; gc.collect()
    all_preds_es[f'lgb30_{H_key}'] = np.mean(els, axis=0)
    all_preds_te[f'lgb30_{H_key}'] = np.mean(tls, axis=0)
    print(f'  lgb30 H={H_key} te_auc={roc_auc_score(yt_te, all_preds_te[f"lgb30_{H_key}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

del X_tr, X_es, X_te, X_tr_30, X_es_30; gc.collect()

# ===== Phase 3: Per-hour-of-day models on TEST =====
print('\n[7] PER-HOUR ANALYSIS (LGB62 H=15)...', flush=True)
hr_te = (ts[te_idx] % 86400) // 3600  # 0..23
y15_perf = y15_te.copy()
pv = all_preds_te['lgb62_15']
print(f'  Overall AUC: {roc_auc_score(y15_perf, pv):.4f}')
for h in range(24):
    hmask = hr_te == h
    if hmask.sum() < 100: continue
    a = roc_auc_score(y15_perf[hmask], pv[hmask])
    k = max(1, int(hmask.sum()*0.01))
    acc = y15_perf[hmask][np.argsort(-pv[hmask])[:k]].mean()*100
    print(f'    h{h:02d}: n={hmask.sum():>5,} auc={a:.4f} top1%={acc:>5.1f}%')

# ===== EVAL ALL =====
print('\n[8] EVAL ALL CONFIGS...', flush=True)
daily = 1440
for target_H in [15, 30]:
    y_te = y_all[target_H][te_idx]
    y_es = y_all[target_H][es_idx]

    # Collect predictions for this target
    relevant_keys = [k for k in all_preds_te if k.endswith(f'_{target_H}')]
    print(f'\n  TARGET H={target_H}  (using keys: {relevant_keys})')

    for src in ['lgb62', 'xgb62', 'cb62', 'lgb30']:
        k = f'{src}_{target_H}'
        if k not in all_preds_te: continue
        pv = all_preds_te[k]
        auc = roc_auc_score(y_te, pv)
        line = f'    [{k:>12s}] auc={auc:.4f}'
        for pct in [0.5, 1, 2, 3, 5, 8, 10]:
            kk = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:kk]
            acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
            line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
        print(line)

    # Equal weight avg of all models
    pv_all = np.mean([all_preds_te[k] for k in relevant_keys], axis=0)
    auc_a = roc_auc_score(y_te, pv_all)
    line = f'    [{"avg_all":>12s}] auc={auc_a:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_all)*pct/100))
        idx = np.argsort(-pv_all)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

    # Meta-LR stacking
    X_meta_es = np.column_stack([all_preds_es[k] for k in relevant_keys])
    X_meta_te = np.column_stack([all_preds_te[k] for k in relevant_keys])
    meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs')
    meta.fit(X_meta_es, y_es)
    pv_meta = meta.predict_proba(X_meta_te)[:,1]
    auc_m = roc_auc_score(y_te, pv_meta)
    line = f'    [{"meta_LR":>12s}] auc={auc_m:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_meta)*pct/100))
        idx = np.argsort(-pv_meta)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# Save predictions for later use
np.savez('/workspace/models/predictions_final.npz',
         te_idx=te_idx, ts_te=ts[te_idx], hr_te=(ts[te_idx] % 86400) // 3600,
         y15_te=y15_te, y30_te=y30_te,
         **{f'p_{k}': v for k,v in all_preds_te.items()})
print(f'\nsaved predictions!')
print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
