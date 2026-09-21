"""TRAIN FINAL — load one split at a time."""
import numpy as np, time, gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

t0 = time.time()
print('[1] load splits...', flush=True)
ys = np.load('/workspace/models/y_split.npz', allow_pickle=True)

preds_es = {}; preds_te = {}

# ===== LGB 3-seed =====
print('\n[2] LGB...', flush=True)
for H in [15, 30]:
    print(f'  H={H}', flush=True)
    X_tr = np.load('/workspace/models/X_tr.npy').astype(np.float32)
    X_es = np.load('/workspace/models/X_es.npy').astype(np.float32)
    X_te = np.load('/workspace/models/X_te.npy').astype(np.float32)
    y_tr = ys[f'y{str(H)}_tr']; y_es = ys[f'y{str(H)}_es']

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

    auc = roc_auc_score(ys[f'y{str(H)}_te'], preds_te[f'lgb_{H}'])
    print(f'    te_auc={auc:.4f}  ({time.time()-t0:.0f}s)', flush=True)

    del X_tr, X_es, X_te; gc.collect()

# ===== CB =====
print('\n[3] CB...', flush=True)
for H in [15, 30]:
    print(f'  H={H}', flush=True)
    X_tr = np.load('/workspace/models/X_tr.npy').astype(np.float32)
    X_es = np.load('/workspace/models/X_es.npy').astype(np.float32)
    X_te = np.load('/workspace/models/X_te.npy').astype(np.float32)
    y_tr = ys[f'y{str(H)}_tr']; y_es = ys[f'y{str(H)}_es']

    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                             subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                             verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
    cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)

    preds_es[f'cb_{H}'] = cb.predict_proba(X_es)[:,1]
    preds_te[f'cb_{H}'] = cb.predict_proba(X_te)[:,1]

    auc = roc_auc_score(ys[f'y{str(H)}_te'], preds_te[f'cb_{H}'])
    print(f'    te_auc={auc:.4f}  ({time.time()-t0:.0f}s)', flush=True)

    del cb, X_tr, X_es, X_te; gc.collect()

# ===== SAVE =====
print('\n[4] SAVE predictions...', flush=True)
np.savez('/workspace/models/predictions_final.npz',
         tr_idx=ys['tr_idx'], es_idx=ys['es_idx'], te_idx=ys['te_idx'],
         ts_te=ys['ts_te'],
         y15=ys['y15_te'], y30=ys['y30_te'],
         **{f'p_{k}': v for k,v in preds_te.items()},
         **{f'p_{k}_es': v for k,v in preds_es.items()})
print('saved!')

# ===== EVAL =====
print('\n[5] EVAL...', flush=True)
daily = 1440
for target_H in [15, 30]:
    y_te = ys[f'y{str(target_H)}_te']
    print(f'\n  === H={target_H} ===')
    for k in [f'lgb_{target_H}', f'cb_{target_H}']:
        pv = preds_te[k]
        auc = roc_auc_score(y_te, pv)
        line = f'  [{k:>12s}] auc={auc:.4f}'
        for pct in [0.5, 1, 2, 3, 5, 8, 10]:
            kk = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:kk]
            acc = y_te[idx].mean()*100; tpd = kk*daily/len(ys['te_idx'])
            line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
        print(line)
    pv_b = (preds_te[f'lgb_{target_H}'] + preds_te[f'cb_{target_H}']) / 2.0
    auc_b = roc_auc_score(y_te, pv_b)
    line = f'  [{"blend":>12s}] auc={auc_b:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_b)*pct/100))
        idx = np.argsort(-pv_b)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(ys['te_idx'])
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
