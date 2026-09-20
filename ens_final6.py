"""ENS_FINAL6: Smart blend — per-hour weighted combination."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('[1] load...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts = D['ts'].astype(np.int64)[vi]

close_raw = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet', columns=['close'])['close'].values.astype(np.float64)
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

X_tr = X[tr_idx]; X_es = X[es_idx]; X_te = X[te_idx]
del X; gc.collect()

hr_es = (ts[es_idx] % 86400) // 3600
hr_te = (ts[te_idx] % 86400) // 3600

# Global model on ES to learn per-hour weights
print('\n[2] Global LGB H=30...', flush=True)
y30_tr = y_30[tr_idx]; y30_es = y_30[es_idx]
m_g = lgb.train({'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
                 'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                 'verbose':-1,'n_jobs':3,'seed':42},
                lgb.Dataset(X_tr, y30_tr), num_boost_round=5000,
                valid_sets=[lgb.Dataset(X_es, y30_es)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
pv_g_es = m_g.predict(X_es)
pv_g_te = m_g.predict(X_te)
print(f'  global te_auc={roc_auc_score(y_30[te_idx], pv_g_te):.4f}  ({time.time()-t0:.0f}s)')

# Per-hour models on ES
print('\n[3] Per-hour LGB H=30...', flush=True)
pv_ph_es = np.zeros(len(es_idx), np.float32)
pv_ph_te = np.zeros(len(te_idx), np.float32)

for h in range(24):
    es_h = hr_es == h; te_h = hr_te == h; tr_h = (ts[tr_idx] % 86400) // 3600 == h
    pv_ph_es[es_h] = pv_g_es[es_h]
    pv_ph_te[te_h] = pv_g_te[te_h]
    if tr_h.sum() < 5000: continue
    
    m_h = lgb.train({'objective':'binary','learning_rate':0.05,'num_leaves':31,'min_child_samples':50,
                     'feature_fraction':0.9,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
                     'verbose':-1,'n_jobs':3,'seed':42},
                    lgb.Dataset(X_tr[tr_h], y30_tr[tr_h]), num_boost_round=5000,
                    valid_sets=[lgb.Dataset(X_es[es_h], y30_es[es_h])],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pv_ph_es[es_h] = m_h.predict(X_es[es_h])
    pv_ph_te[te_h] = m_h.predict(X_te[te_h])
    del m_h; gc.collect()

print(f'  per-hour done ({time.time()-t0:.0f}s)')

# Per-hour AUC on ES (for learning weights)
print('\n[4] LEARN WEIGHTS on ES...', flush=True)
y30_es_arr = y_30[es_idx]
y30_te_arr = y_30[te_idx]
# For each hour, decide blend weight by comparing AUC
w_g = np.zeros(24, np.float32)
for h in range(24):
    hmask = hr_es == h
    if hmask.sum() < 50:
        w_g[h] = 1.0  # default to global
        continue
    auc_g = roc_auc_score(y30_es_arr[hmask], pv_g_es[hmask])
    auc_ph = roc_auc_score(y30_es_arr[hmask], pv_ph_es[hmask])
    w_g[h] = max(0.0, min(1.0, (auc_g - 0.5) / max(auc_g - 0.5 + auc_ph - 0.5, 1e-6)))
    if auc_g >= auc_ph:
        w_g[h] = 1.0  # trust global
    else:
        w_g[h] = 0.0  # trust perhr

print(f'  Weights (0=perhr, 1=global):')
for h in range(24):
    hmask = hr_es == h
    if hmask.sum() < 50: continue
    auc_g = roc_auc_score(y30_es_arr[hmask], pv_g_es[hmask])
    auc_ph = roc_auc_score(y30_es_arr[hmask], pv_ph_es[hmask])
    use = 'G' if w_g[h] >= 0.5 else 'P'
    print(f'    h{h:02d}: auc_g={auc_g:.4f} auc_ph={auc_ph:.4f} → {use}')

# Blend
hr_te_arr = hr_te.astype(np.float32)
# Create weight array
w_arr = np.zeros(len(te_idx), np.float32)
for h in range(24):
    w_arr[hr_te_arr == h] = w_g[h]
pv_blend_te = w_arr * pv_g_te + (1 - w_arr) * pv_ph_te

# ===== Also try TOP-K PER-HOUR =====
print('\n[5] TOP-K PER-HOUR SELECTION...', flush=True)
# Instead of global top-1%, take top-pct *within each hour*
pv_phr_topk = np.zeros(len(te_idx), np.float32)
daily = 1440
for h in range(24):
    hmask = hr_te == h
    n = hmask.sum()
    if n < 50: continue
    scores = pv_g_te[hmask]  # use global model scores
    ranked = np.argsort(-scores)
    for pct in [0.5, 1, 2, 3, 5]:
        k = max(1, int(n*pct/100))
        idx_h = ranked[:k]
        global_idx = np.where(hmask)[0][idx_h]
        pv_phr_topk[global_idx] = scores[idx_h]  # keep top-scores
    # Rest get -inf
    low_idx = np.argsort(scores)[int(n*0.05):]
    global_low = np.where(hmask)[0][low_idx]
    pv_phr_topk[global_low] = -1e18

# ===== Also try: only trade in high-AUC hours =====
print('\n[6] HIGH-AUC-HOUR FILTER...', flush=True)
# Use ES auc to select which hours to trade
trade_hours = []
for h in range(24):
    hmask = hr_es == h
    if hmask.sum() < 50: continue
    auc_g = roc_auc_score(y30_es_arr[hmask], pv_g_es[hmask])
    if auc_g > 0.54:
        trade_hours.append(h)
print(f'  Trading hours (ES AUC > 0.54): {trade_hours}')
pv_hfilt_te = pv_g_te.copy()
# Zero out non-trade hours
not_trade = ~np.isin(hr_te, trade_hours)
pv_hfilt_te[not_trade] = -1e18

# ===== EVAL ALL =====
print('\n[7] FINAL EVAL (H=30)...', flush=True)
y_te = y30_te_arr

configs = [
    ('global', pv_g_te),
    ('perhr', pv_ph_te),
    ('blend', pv_blend_te),
    ('phr_topk', pv_phr_topk),  # only top within each hour
    ('hfilt', pv_hfilt_te),    # only trade high-AUC hours
]

for label, pv in configs:
    # For configs that have -1e18, filter them out
    if pv.min() < -1e9:
        valid = pv > -1e9
        auc = roc_auc_score(y_te[valid], pv[valid]) if valid.sum() > 100 else 0.5
    else:
        auc = roc_auc_score(y_te, pv)
    
    line = f'  [{label:>10s}] auc={auc:.4f}'
    for pct in [0.5, 1, 2, 3, 5, 8, 10]:
        if pv.min() < -1e9:
            # Only consider valid entries
            valid = pv > -1e9
            n_valid = valid.sum()
            k = max(1, int(n_valid*pct/100))
            idx_valid = np.argsort(-pv[valid])[:k]
            acc = y_te[valid][idx_valid].mean()*100
            tpd = k*daily/len(te_idx)  # tpd based on total days
        else:
            k = max(1, int(len(pv)*pct/100))
            idx = np.argsort(-pv)[:k]
            acc = y_te[idx].mean()*100; tpd = k*daily/len(te_idx)
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line)

# Save
np.savez('/workspace/models/predictions_v6.npz',
         te_idx=te_idx, ts_te=ts[te_idx], hr_te=hr_te, y30_te=y_te,
         pv_global=pv_g_te, pv_perhr=pv_ph_te, pv_blend=pv_blend_te, pv_hfilt=pv_hfilt_te)
print(f'\n⏱ {time.time()-t0:.0f}s')
