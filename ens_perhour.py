"""PER-HOUR-OF-DAY LGB: Each hour of day gets its own model."""
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

close_raw = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
y_15 = np.full(len(close_raw), np.nan, np.float32)
y_15[:-15] = (close_raw[15:]/close_raw[:-15]-1).astype(np.float32)
y_15 = (y_15[vi] > 0).astype(np.int8)
y_30 = np.full(len(close_raw), np.nan, np.float32)
y_30[:-30] = (close_raw[30:]/close_raw[:-30]-1).astype(np.float32)
y_30 = (y_30[vi] > 0).astype(np.int8)

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_idx = np.where(ts_mask('2020-01-01','2024-06-30'))[0]
es_idx = np.where(ts_mask('2024-06-30','2024-09-30'))[0]
te_idx = np.where(ts_mask('2025-09-30','2026-08-29'))[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# Split
X_tr = X[tr_idx]; X_es = X[es_idx]; X_te = X[te_idx]
del X; gc.collect()

# Hour of day (UTC)
hr_tr = (ts[tr_idx] % 86400) // 3600
hr_es = (ts[es_idx] % 86400) // 3600
hr_te = (ts[te_idx] % 86400) // 3600

# ===== Phase 1: Global model baseline =====
print('\n[2] GLOBAL LGB (H=30)...', flush=True)
y30_tr = y_30[tr_idx]; y30_es = y_30[es_idx]
m_global = lgb.train({'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
                       'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                       'verbose':-1,'n_jobs':3,'seed':42},
                      lgb.Dataset(X_tr, y30_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_es, y30_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
pv_global_te = m_global.predict(X_te)
pv_global_es = m_global.predict(X_es)
del m_global; gc.collect()
print(f'  global H=30 te_auc={roc_auc_score(y_30[te_idx], pv_global_te):.4f}')

# ===== Phase 2: Per-Hour Models =====
print('\n[3] PER-HOUR LGB (H=30)...', flush=True)
pv_perhr_te = np.zeros(len(te_idx), np.float32)
pv_perhr_es = np.zeros(len(es_idx), np.float32)
perhr_aucs = {}

for h in range(24):
    tr_h = hr_tr == h
    es_h = hr_es == h
    te_h = hr_te == h
    n_tr = tr_h.sum(); n_es = es_h.sum(); n_te = te_h.sum()
    if n_tr < 1000 or n_es < 50 or n_te < 50:
        # fallback to global
        pv_perhr_te[te_h] = pv_global_te[te_h]
        pv_perhr_es[es_h] = pv_global_es[es_h]
        continue

    m_h = lgb.train({'objective':'binary','learning_rate':0.05,'num_leaves':31,'min_child_samples':100,
                      'feature_fraction':0.9,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                      'verbose':-1,'n_jobs':3,'seed':42},
                     lgb.Dataset(X_tr[tr_h], y_30[tr_idx][tr_h]), num_boost_round=5000,
                     valid_sets=[lgb.Dataset(X_es[es_h], y_30[es_idx][es_h])],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pv_perhr_te[te_h] = m_h.predict(X_te[te_h])
    pv_perhr_es[es_h] = m_h.predict(X_es[es_h])

    auc_h = roc_auc_score(y_30[te_idx][te_h], pv_perhr_te[te_h])
    perhr_aucs[h] = auc_h

    # Clean
    del m_h; gc.collect()

    if h % 4 == 0:
        print(f'  h{h:02d}: tr={n_tr:,} es={n_es:,} te={n_te:,} auc={auc_h:.4f}  ({time.time()-t0:.0f}s)', flush=True)

# Fill remaining (shouldn't happen since all hours have data)
pv_perhr_te[pv_perhr_te == 0] = pv_global_te[pv_perhr_te == 0]
pv_perhr_es[pv_perhr_es == 0] = pv_global_es[pv_perhr_es == 0]

print(f'\n  Per-hour AUC summary:')
for h in sorted(perhr_aucs):
    print(f'    h{h:02d}: auc={perhr_aucs[h]:.4f}')

# ===== Phase 3: Weighted combination =====
print('\n[4] COMBINE...', flush=True)
# Simple options
pv_avg_te = (pv_global_te + pv_perhr_te) / 2.0
pv_avg_es = (pv_global_es + pv_perhr_es) / 2.0

# Meta-LR
meta = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs')
meta.fit(np.column_stack([pv_global_es, pv_perhr_es]), y_30[es_idx])
pv_meta_te = meta.predict_proba(np.column_stack([pv_global_te, pv_perhr_te]))[:,1]

# Also: confidence weighting based on per-hour AUC
# If per-hour AUC > global for that hour, trust per-hour more
pv_conf_te = np.where(
    np.array([perhr_aucs.get(h, 0.5) for h in hr_te]) > 0.5435,  # global baseline
    pv_perhr_te, pv_global_te
)

# ===== Phase 4: EVAL =====
print('\n[5] EVAL (H=30, TE)...', flush=True)
y_te = y_30[te_idx]
daily = 1440

configs = [
    ('global', pv_global_te),
    ('perhr', pv_perhr_te),
    ('avg', pv_avg_te),
    ('meta', pv_meta_te),
    ('conf', pv_conf_te),
]

for label, pv in configs:
    auc = roc_auc_score(y_te, pv)
    line = f'  [{label:>8s}] auc={auc:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean()*100; tpd = k*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# Also per-hour breakdown of global vs perhr
print('\n  Per-hour top-1% accuracy (H=30):')
for h in range(24):
    hmask = hr_te == h
    if hmask.sum() < 100: continue
    k = max(1, int(hmask.sum()*0.01))
    y_h = y_te[hmask]
    acc_g = y_h[np.argsort(-pv_global_te[hmask])[:k]].mean()*100
    acc_p = y_h[np.argsort(-pv_perhr_te[hmask])[:k]].mean()*100
    better = 'PERHR' if acc_p > acc_g else 'GLOBAL'
    print(f'    h{h:02d}: global={acc_g:>5.1f}%  perhr={acc_p:>5.1f}%  ({better})')

# Save
np.savez('/workspace/models/predictions_perhour.npz',
         te_idx=te_idx, ts_te=ts[te_idx], hr_te=hr_te, y30_te=y_te,
         pv_global=pv_global_te, pv_perhr=pv_perhr_te, pv_meta=pv_meta_te, pv_conf=pv_conf_te)
print(f'\n⏱ {time.time()-t0:.0f}s')
