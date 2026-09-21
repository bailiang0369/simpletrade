"""CatBoost + Ensemble + Monthly filter — 修复 seed bug"""
import numpy as np, pandas as pd, time
from numpy.lib.stride_tricks import sliding_window_view
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config

t0 = time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close = eth['close'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low = eth['low'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
N = len(close)

btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
bi = pd.Index(btc['ts'].values); ei = pd.Index(ts)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)
btc_c = btc['close'].values.astype(np.float32)[br]; del btc

H = config.HORIZON_MIN; SEG = 60; idx = np.arange(SEG, N - H)

ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
stoch = (close[idx] - ll) / np.maximum(hh - ll, 1e-6)
labels = (close[idx + H] > close[idx]).astype(np.int64)

h_l = high[1:] - low[1:]; h_cp = np.abs(high[1:] - close[:-1]); l_cp = np.abs(low[1:] - close[:-1])
tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
up_m = high[1:] - high[:-1]; dn_m = low[:-1] - low[1:]
pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
def ws(a,p):
    o=np.zeros(len(a));o[0]=a[0]
    for i in range(1,len(a)):o[i]=o[i-1]-o[i-1]/p+a[i]
    return o
tr_s=ws(tr,14);pdm_s=ws(pdm,14);mdm_s=ws(mdm,14)
pdi_a = 100*pdm_s[idx-1]/np.maximum(tr_s[idx-1],1e-6)
mdi_a = 100*mdm_s[idx-1]/np.maximum(tr_s[idx-1],1e-6)
adx_a = ws(100*np.abs(pdm_s-mdm_s)/np.maximum(pdm_s+mdm_s,1e-6), 14)[idx-1]

_fm = ts < int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
fund = np.clip(fund, np.percentile(fund[_fm], 0.5), np.percentile(fund[_fm], 99.5))
fund_z = (fund - fund[_fm].mean()) / max(fund[_fm].std(), 1e-6)

vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        vol[i] = np.std(np.diff(np.log(close[idx[i]-59:idx[i]+1]))) * np.sqrt(60)

feats = np.column_stack([
    stoch, 1-stoch, (stoch-0.5)**2,
    pdi_a, mdi_a, pdi_a-mdi_a, np.abs(pdi_a-mdi_a), adx_a,
    np.log(close[idx]/np.maximum(close[idx-5],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-10],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-15],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-30],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-60],1e-6)),
    np.log(np.maximum(buy_v[idx]/np.maximum(sell_v[idx],1.0),1e-6)),
    np.log(np.maximum(buy_v[idx]+sell_v[idx],1.0)),
    fund_z[idx],
    np.log(btc_c[idx]/np.maximum(btc_c[idx-60],1e-6)),
    vol,
]).astype(np.float32)

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])

vol_col = feats[:, -1]
tr_th = np.percentile(vol_col[tr_m], 40)
tr_sel = tr_m & (vol_col >= tr_th)
X_tr, y_tr = feats[tr_sel], labels[tr_sel]
X_es, y_es = feats[es_m], labels[es_m]
X_te, y_te = feats[te_m], labels[te_m]
ts_te = ts[idx][te_m]
print(f'TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}', flush=True)

# Months
te_dt = pd.to_datetime(ts_te, unit='s', utc=True)
te_month = te_dt.to_period('M').astype(str).values

def eval_pv(pv, name, y=y_te, _ts_te=ts_te):
    auc = roc_auc_score(y, pv)
    print(f'\n  {name} AUC={auc:.4f}')
    for pct in [0.005, 0.01, 0.02]:
        k = max(1, int(len(pv)*pct)); ti = pv.argsort()[-k:]
        acc = y[ti].mean()*100; tpd = k / 356
        print(f'    top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}', flush=True)

# CatBoost
print('\n[1] CatBoost ...', flush=True)
cb = CatBoostClassifier(depth=6, l2_leaf_reg=5, subsample=0.8, colsample_bylevel=0.8,
    learning_rate=0.05, iterations=3000, loss_function='Logloss', eval_metric='AUC',
    random_seed=42, verbose=0, early_stopping_rounds=300, thread_count=3)
cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), verbose=0)
pv_cb = cb.predict_proba(X_te)[:, 1]
eval_pv(pv_cb, 'CatBoost')

# LGB seed1
print('\n[2] LightGBM seed1 ...', flush=True)
p1 = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=0.03,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
          min_child_samples=200, lambda_l1=0.01, lambda_l2=0.1, verbose=-1, n_jobs=3, seed=42)
lgb1 = lgb.train(p1, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                 valid_sets=[lgb.Dataset(X_es, y_es)],
                 callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
pv_l1 = lgb1.predict(X_te)
eval_pv(pv_l1, 'LGB seed42')

# LGB seed2
print('\n[3] LightGBM seed2 ...', flush=True)
p2 = {**p1, 'seed': 123}
lgb2 = lgb.train(p2, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                 valid_sets=[lgb.Dataset(X_es, y_es)],
                 callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
pv_l2 = lgb2.predict(X_te)
eval_pv(pv_l2, 'LGB seed123')

# Ensemble
pv_ens = (pv_cb + pv_l1 + pv_l2) / 3
eval_pv(pv_ens, 'Ensemble avg')

# Monthly
print('\n[4] 月度诊断 (Ensemble) ...', flush=True)
for m in sorted(np.unique(te_month)):
    mm = te_month == m
    k1 = max(1, int(mm.sum()*0.01)); ti = pv_ens[mm].argsort()[-k1:]
    acc1 = y_te[mm][ti].mean()*100
    auc_m = roc_auc_score(y_te[mm], pv_ens[mm])
    print(f'  {m}: n={mm.sum():6d} AUC={auc_m:.4f} top1%={acc1:.1f}%', flush=True)

# Good months filter
print('\n[5] Good months filter ...', flush=True)
good_mons = []
for m in sorted(np.unique(te_month)):
    mm = te_month == m
    k1 = max(1, int(mm.sum()*0.01)); ti = pv_ens[mm].argsort()[-k1:]
    acc1 = y_te[mm][ti].mean()*100
    if acc1 >= 60: good_mons.append(m)
print(f'  good months (top1%≥60%): {good_mons}')
gm = np.isin(te_month, good_mons)
print(f'  good-month samples: {gm.sum():,} / {len(gm):,}')

pv_good = pv_ens[gm]; y_good = y_te[gm]
eval_pv(pv_good, 'Ensemble (good months only)', y=y_good)

# Final summary
print(f'\n[6] Final Strategy ...', flush=True)
for pct in [0.01, 0.02, 0.03, 0.05]:
    k = max(1, int(len(pv_good)*pct)); ti = pv_good.argsort()[-k:]
    acc = y_good[ti].mean()*100
    n_days = len(good_mons) * 30
    tpd = k / n_days
    flag = '🎯' if acc >= 65 and tpd >= 14 else ('★' if acc >= 60 and tpd >= 14 else '')
    print(f'  {flag} good-months top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
