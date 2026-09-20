"""Just CB H=30, careful memory."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

t0 = time.time()
print('[1] load LGB preds...', flush=True)
# Load previous LGB + CB H=15 from partial run
# Actually recompute LGB quickly — it's the same
D = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)

eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
close = eth['close'].values.astype(np.float64)
N = len(close)
y_all = {}
for h in [5,10,15,30]:
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

# Split minimal
X_tr = X[tr_idx][:2000000]  # use subset for speed
X_es = X[es_idx]; X_te = X[te_idx]
y30_tr = y_all[30][tr_idx][:2000000]; y30_es = y_all[30][es_idx]
del X; gc.collect()
print(f'  X_tr_subset={X_tr.shape}')

# CB H=30 only
print('[2] CB H=30...', flush=True)
cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                         subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                         verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
cb.fit(X_tr, y30_tr, eval_set=(X_es, y30_es), use_best_model=True)
pv_cb30_te = cb.predict_proba(X_te)[:,1]
del cb, X_tr, X_es; gc.collect()
print(f'  CB H=30 te_auc={roc_auc_score(y_all[30][te_idx], pv_cb30_te):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# Load full X for LGB 3-seed H=30
print('[3] LGB 3-seed H=30...', flush=True)
D2 = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X_full = D2['X'].astype(np.float32)
X_tr = X_full[tr_idx]; X_es = X_full[es_idx]
del X_full, D2; gc.collect()

y30_tr = y_all[30][tr_idx]; y30_es = y_all[30][es_idx]
y30_te = y_all[30][te_idx]

els=[]; tls=[]
for seed in [42, 49, 56]:
    p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
         'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
         'verbose':-1,'n_jobs':3,'seed':seed}
    m = lgb.train(p, lgb.Dataset(X_tr, y30_tr), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X_es, y30_es)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    els.append(m.predict(X_es)); tls.append(m.predict(X_te)); del m; gc.collect()
pv_lgb30_es = np.mean(els, axis=0)
pv_lgb30_te = np.mean(tls, axis=0)
del X_tr, X_es; gc.collect()
print(f'  LGB H=30 te_auc={roc_auc_score(y30_te, pv_lgb30_te):.4f}')

# CB H=15 with BTC feats
print('[4] CB H=15...', flush=True)
D3 = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X_tr = D3['X'].astype(np.float32)[tr_idx]; X_es = D3['X'].astype(np.float32)[es_idx]
del D3; gc.collect()
y15_tr = y_all[15][tr_idx]; y15_es = y_all[15][es_idx]; y15_te = y_all[15][te_idx]
cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                         subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                         verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
cb.fit(X_tr, y15_tr, eval_set=(X_es, y15_es), use_best_model=True)
pv_cb15_te = cb.predict_proba(X_te)[:,1]
del cb, X_tr, X_es; gc.collect()
print(f'  CB H=15 te_auc={roc_auc_score(y15_te, pv_cb15_te):.4f}')

# Quick LGB H=15 (BTC)
print('[5] LGB H=15...', flush=True)
D4 = np.load('/workspace/models/eth_full.npz', allow_pickle=True)
X_tr = D4['X'].astype(np.float32)[tr_idx]; X_es = D4['X'].astype(np.float32)[es_idx]
del D4; gc.collect()
els=[]; tls=[]
for seed in [42, 49, 56]:
    p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
         'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
         'verbose':-1,'n_jobs':3,'seed':seed}
    m = lgb.train(p, lgb.Dataset(X_tr, y15_tr), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X_es, y15_es)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    els.append(m.predict(X_es)); tls.append(m.predict(X_te)); del m; gc.collect()
pv_lgb15_te = np.mean(tls, axis=0)
del X_tr, X_es; gc.collect()
print(f'  LGB H=15 te_auc={roc_auc_score(y15_te, pv_lgb15_te):.4f}')

# ===== SAVE ALL =====
print('\n[6] SAVE...', flush=True)
np.savez('/workspace/models/predictions_final2.npz',
         te_idx=te_idx, ts_te=ts[te_idx],
         y15=y15_te, y30=y30_te,
         p_lgb30=pv_lgb30_te, p_cb30=pv_cb30_te,
         p_lgb15=pv_lgb15_te, p_cb15=pv_cb15_te)
print('saved!')

# ===== EVAL =====
print('\n[7] FINAL EVAL...', flush=True)
daily = 1440

for target_H, y_te, keys in [(15, y15_te, ['lgb15', 'cb15']), (30, y30_te, ['lgb30', 'cb30'])]:
    print(f'\n  === H={target_H} ===')
    for k in keys:
        pv = locals()[f'pv_{k}_te']
        auc = roc_auc_score(y_te, pv)
        line = f'  [{k:>12s}] auc={auc:.4f}'
        for pct in [0.5, 1, 2, 3, 5, 8, 10]:
            kk = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:kk]
            acc = y_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
            line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
        print(line)

# Blend H=30
print('\n  H=30 BLEND (lgb30+cb30 equal):')
pv_blend = (pv_lgb30_te + pv_cb30_te) / 2.0
auc = roc_auc_score(y30_te, pv_blend)
line = f'  [{"blend":>12s}] auc={auc:.4f}'
for pct in [0.5, 1, 2, 3, 5, 8, 10]:
    kk = max(1, int(len(pv_blend)*pct/100))
    idx = np.argsort(-pv_blend)[:kk]
    acc = y30_te[idx].mean()*100; tpd = kk*daily/len(te_idx)
    line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
print(line)

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
