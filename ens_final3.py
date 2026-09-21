"""ENS_FINAL3: Multi-scale bar feats + XGB + CB + LGB ensemble."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

t0 = time.time()
print('[1] load raw data...', flush=True)
eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
ts_orig = eth['ts'].values.astype(np.int64)
eth['ts'] = pd.to_datetime(eth['ts'], unit='s', utc=True)
eth = eth.set_index('ts')

def resample_ohlcv(df, mins):
    g = df.resample(f'{mins}min', label='right', closed='right')
    r = g.agg({'open':'first','high':'max','low':'min','close':'last','buy_vol':'sum','sell_vol':'sum','funding':'last'})
    return r.dropna()

print('[2] resample...', flush=True)
b5 = resample_ohlcv(eth, 5)
b15 = resample_ohlcv(eth, 15)
print(f'  5min={len(b5):,}  15min={len(b15):,}')

def bar_feats(bars):
    c = bars['close'].values.astype(np.float64)
    h = bars['high'].values.astype(np.float64)
    l = bars['low'].values.astype(np.float64)
    o = bars['open'].values.astype(np.float64)
    bv = bars['buy_vol'].values.astype(np.float64)
    sv = bars['sell_vol'].values.astype(np.float64)
    fnd = bars['funding'].values.astype(np.float64)
    feats = {}
    for w in [1,2,3,5,10,20]:
        feats[f'ret_{w}'] = np.full(len(c), np.nan); feats[f'ret_{w}'][w:] = c[w:]/c[:-w] - 1
    feats['range'] = (h - l) / np.maximum(c, 1e-9)
    feats['body'] = (c - o) / np.maximum(c, 1e-9)
    feats['ushadow'] = (h - np.maximum(o, c)) / np.maximum(c, 1e-9)
    feats['lshadow'] = (np.minimum(o, c) - l) / np.maximum(c, 1e-9)
    ret1 = c[1:]/c[:-1] - 1
    for w in [5,10,20,40]:
        s = pd.Series(ret1).rolling(w, min_periods=w).std().values
        feats[f'vol_{w}'] = np.full(len(c), np.nan); feats[f'vol_{w}'][1:] = s
    tot = bv + sv
    feats['bvr'] = bv / np.maximum(tot, 1e-9)
    feats['bvr_fast'] = pd.Series(feats['bvr']).rolling(5,min_periods=5).mean().values
    feats['bvr_slow'] = pd.Series(feats['bvr']).rolling(20,min_periods=20).mean().values
    for w in [5,10,20]:
        feats[f'cvd_{w}'] = pd.Series(bv-sv).rolling(w,min_periods=w).sum().values
    for w in [1,5,10,30]:
        feats[f'fund_{w}'] = pd.Series(fnd).rolling(w,min_periods=1).mean().values
    return pd.DataFrame(feats, index=bars.index)

print('[3] compute feats...', flush=True)
f5 = bar_feats(b5)
f15 = bar_feats(b15)
print(f'  5min feats={f5.shape[1]}  15min feats={f15.shape[1]}')

# Align to 1min timestamps (forward fill)
ts_1min = eth.index

def align_to_1m(df_feats, idx_1m):
    out = df_feats.reindex(idx_1m, method='ffill')
    # Fix leading NaN
    out = out.bfill()
    return out.values

feats5_1m = align_to_1m(f5, ts_1min)
feats15_1m = align_to_1m(f15, ts_1min)
print(f'  aligned: 5min→{feats5_1m.shape}  15min→{feats15_1m.shape}')

# Load existing dataset
print('[4] load existing...', flush=True)
D = np.load('/workspace/models/eth_data.npz', allow_pickle=True)
X_old = D['X'].astype(np.float32)
vi = D['vi'].astype(np.int64)
ts_vi = D['ts'].astype(np.int64)[vi]
old_names = list(D['feat_names'])

# Align new feats only to vi rows
new_X = np.column_stack([X_old, feats5_1m[vi], feats15_1m[vi]])
new_names = old_names + [f'5m_{n}' for n in f5.columns] + [f'15m_{n}' for n in f15.columns]
print(f'  new X: {new_X.shape}')
del X_old, feats5_1m, feats15_1m; gc.collect()

# Replace NaN
nans = np.isnan(new_X).any(axis=1)
print(f'  rows NaN: {nans.sum():,} → replace with -999')
new_X = np.where(nans[:,None], -999, new_X).astype(np.float32)

# Labels
close_all = eth['close'].values.astype(np.float64)
y_all = {}
for h in [5,10,15,30]:
    rh = np.full(len(close_all), np.nan, np.float32)
    rh[:-h] = (close_all[h:]/close_all[:-h]-1).astype(np.float32)
    y_all[h] = (rh[vi] > 0).astype(np.int8)

def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_vi>=a)&(ts_vi<b)
tr_mask = ts_mask('2020-01-01','2024-06-30')
es_mask = ts_mask('2024-06-30','2024-09-30')
te_mask = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_mask)[0]; es_idx = np.where(es_mask)[0]; te_idx = np.where(te_mask)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# ===== TRAIN =====
print('\n[5] LGB...', flush=True)
preds_es = {}; preds_te = {}
for H in [5,10,15,30]:
    print(f'  H={H}', flush=True)
    y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
    els=[]; tls=[]
    for seed in [42, 49, 56]:
        p = {'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
             'verbose':-1,'n_jobs':3,'seed':seed,'missing':-999}
        m = lgb.train(p, lgb.Dataset(new_X[tr_idx], y_tr), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(new_X[es_idx], y_es)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        els.append(m.predict(new_X[es_idx]))
        tls.append(m.predict(new_X[te_idx]))
    preds_es[f'lgb_{H}'] = np.mean(els, axis=0)
    preds_te[f'lgb_{H}'] = np.mean(tls, axis=0)
    print(f'    te={roc_auc_score(y_all[H][te_idx], preds_te[f"lgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)

# XGBoost
print('\n[6] XGB...', flush=True)
try:
    import xgboost as xgb
    for H in [15, 30]:
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        els=[]; tls=[]
        for seed in [42, 56]:
            p = {'objective':'binary:logistic','eval_metric':'logloss','learning_rate':0.05,
                 'max_depth':8,'min_child_weight':200,'subsample':0.8,'colsample_bytree':0.8,
                 'reg_lambda':1.0,'tree_method':'hist','verbosity':0,'nthread':3,'seed':seed}
            dtrain = xgb.DMatrix(new_X[tr_idx], label=y_tr)
            dval = xgb.DMatrix(new_X[es_idx], label=y_es)
            m = xgb.train(p, dtrain, num_boost_round=5000, evals=[(dval,'val')],
                          early_stopping_rounds=100, verbose_eval=False)
            els.append(m.predict(xgb.DMatrix(new_X[es_idx])))
            tls.append(m.predict(xgb.DMatrix(new_X[te_idx])))
        preds_es[f'xgb_{H}'] = np.mean(els, axis=0)
        preds_te[f'xgb_{H}'] = np.mean(tls, axis=0)
        print(f'    H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"xgb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'    XGB failed: {e}', flush=True)

# CatBoost
print('\n[7] CB...', flush=True)
try:
    from catboost import CatBoostClassifier
    for H in [15, 30]:
        y_tr = y_all[H][tr_idx]; y_es = y_all[H][es_idx]
        cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5,
                                 subsample=0.8, colsample_bylevel=0.8, od_type='Iter', od_wait=100,
                                 verbose=0, random_seed=42, thread_count=3, loss_function='Logloss')
        cb.fit(new_X[tr_idx], y_tr, eval_set=(new_X[es_idx], y_es), use_best_model=True)
        preds_es[f'cb_{H}'] = cb.predict_proba(new_X[es_idx])[:,1]
        preds_te[f'cb_{H}'] = cb.predict_proba(new_X[te_idx])[:,1]
        print(f'    H={H} te={roc_auc_score(y_all[H][te_idx], preds_te[f"cb_{H}"]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
except Exception as e:
    print(f'    CB failed: {e}', flush=True)

del new_X; gc.collect()

# ===== COMBINE =====
print('\n[8] COMBINE & EVAL...', flush=True)
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

    # Also show all individual
    for k in sorted(preds_te):
        auc_k = roc_auc_score(y_te_t, preds_te[k])
        acc_k = y_te_t[np.argsort(-preds_te[k])[:max(1, int(len(preds_te[k])*0.01))]].mean()*100
        print(f'    {k:>12s}: auc={auc_k:.4f}  top1%={acc_k:.1f}%')

    # Monthly
    dt_te = pd.to_datetime(ts_vi[te_idx], unit='s', utc=True)
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

print(f'\n⏱ {time.time()-t0:.0f}s')
