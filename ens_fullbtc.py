"""ENS_FULLBTC: 87 feats (62 ETH + 25 BTC), LGB+CB, careful memory."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

t0 = time.time()
print('[1] load eth_full (87 feats)...', flush=True)
D = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)
feat_names = list(D['feat_names'])
print(f'  X={X.shape}  ({X.shape[1]} feats)  mem={X.nbytes/1e9:.2f}GB')

# Labels
y_all = {}
for h in [5,10,15,30]:
    y_all[h] = D[f'y_{h}'].astype(np.int8)

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
print(f'  X_tr={X_tr.shape}  mem={X_tr.nbytes/1e9:.2f}GB')

# ===== LGB =====
print('\n[2] LGB training...', flush=True)
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
    print(f'    te_auc={roc_auc_score(y_all[H][te_idx], preds_te[f"lgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# ===== CB =====
print('\n[3] CatBoost...', flush=True)
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

del X_tr, X_es, X_te; gc.collect()

# ===== FEATURE IMPORTANCE =====
print('\n[4] Top feats (H=30)...', flush=True)
m_demo = lgb.train({'objective':'binary','verbose':-1,'n_jobs':3,'seed':42},
                    lgb.Dataset(np.load('/workspace/models/eth_full.npz', allow_pickle=True)['X'][tr_idx], y_all[30][tr_idx]),
                    num_boost_round=500,
                    valid_sets=[lgb.Dataset(np.load('/workspace/models/eth_full.npz', allow_pickle=True)['X'][es_idx], y_all[30][es_idx])],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
imps = m_demo.feature_importance(importance_type='gain')
top = np.argsort(-imps)[:20]
print('  Top 20:')
for i in top:
    is_btc = 'BTC ' if 'btc' in feat_names[i] or 'corr' in feat_names[i] or 'spread' in feat_names[i] else '    '
    print(f'    {is_btc} {i:>2d} {feat_names[i]:>18s}  gain={imps[i]:>10.0f}')
del m_demo; gc.collect()

# ===== EVAL =====
print('\n[5] EVAL...', flush=True)
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

# Save predictions
np.savez('/workspace/models/predictions_final.npz', te_idx=te_idx, ts_te=ts[te_idx],
         y15=y_all[15][te_idx], y30=y_all[30][te_idx],
         **{f'p_{k}': v for k,v in preds_te.items()})
print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
