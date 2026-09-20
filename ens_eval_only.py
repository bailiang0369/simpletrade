"""EVAL ONLY — use single-seed LGB on subset to avoid OOM, then combine with CB."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('[1] load dataset...', flush=True)
D = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X_full = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)
del D; gc.collect()

eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
close = eth['close'].values.astype(np.float64)
N = len(close)
y_all = {}
for h in [15, 30]:
    r = np.full(N, np.nan, np.float32)
    r[:-h] = (close[h:]/close[:-h]-1).astype(np.float32)
    y_all[h] = (r[vi] > 0).astype(np.int8)
del close, eth; gc.collect()

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_idx = np.where(ts_mask('2020-01-01','2024-06-30'))[0]
es_idx = np.where(ts_mask('2024-06-30','2024-09-30'))[0]
te_idx = np.where(ts_mask('2025-09-30','2026-08-29'))[0]

# Use subset for training to avoid OOM — 1.5M rows is enough
tr_sub = tr_idx[:1500000]
X_tr = X_full[tr_sub]; X_es = X_full[es_idx]; X_te = X_full[te_idx]
del X_full; gc.collect()
print(f'  X_tr_sub={X_tr.shape}  X_es={X_es.shape}  X_te={X_te.shape}')

# ===== Train single-seed LGB on subset, 3-seed on ES =====
print('\n[2] LGB...', flush=True)
preds_es = {}; preds_te = {}

for H in [15, 30]:
    y_tr = y_all[H][tr_sub]; y_es = y_all[H][es_idx]
    # 3 seeds on subset
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
    print(f'  LGB H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"lgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# ===== CB =====
print('\n[3] CB...', flush=True)
from catboost import CatBoostClassifier
for H in [15, 30]:
    y_tr = y_all[H][tr_sub]; y_es = y_all[H][es_idx]
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                             subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                             verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
    cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
    preds_es[f'cb_{H}'] = cb.predict_proba(X_es)[:,1]
    preds_te[f'cb_{H}'] = cb.predict_proba(X_te)[:,1]
    del cb; gc.collect()
    print(f'  CB H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"cb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

del X_tr, X_es, X_te; gc.collect()

# ===== EVAL =====
print('\n[4] FINAL EVAL (87 feats = 62 ETH + 25 BTC)...', flush=True)
daily = 1440
best_result = None
best_label = None

for target_H in [15, 30]:
    y_te = y_all[target_H][te_idx]
    print(f'\n  === TARGET H={target_H} ===')
    
    for k in [f'lgb_{target_H}', f'cb_{target_H}']:
        pv = preds_te[k]
        auc = roc_auc_score(y_te, pv)
        line = f'  [{k:>12s}] auc={auc:.4f}'
        for pct in [0.5, 1, 2, 3, 5, 8, 10]:
            kk = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:kk]
            acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
            line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
        print(line)
    
    # Blend
    pv_b = (preds_te[f'lgb_{target_H}'] + preds_te[f'cb_{target_H}']) / 2.0
    auc_b = roc_auc_score(y_te, pv_b)
    line = f'  [{"blend":>12s}] auc={auc_b:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        kk = max(1, int(len(pv_b)*pct/100))
        idx = np.argsort(-pv_b)[:kk]
        acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# Overall best config
print('\n📊 SUMMARY (87 feats, 1.5M training sample):')
print('  Compared to 62 feats (no BTC) baseline:')
print('    62 feats best = LGB H=30, top-1%=60.2%, AUC=0.5435')
print(f'    87 feats best = see above')

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
