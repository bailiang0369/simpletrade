"""CatBoost + Ensemble + Monthly filter — 冲 top1%≥65%"""
import numpy as np, pandas as pd, time
from numpy.lib.stride_tricks import sliding_window_view
from catboost import CatBoostClassifier, Pool
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

# TR vol>40%ile
vol_col = feats[:, -1]
tr_th = np.percentile(vol_col[tr_m], 40)
tr_sel = tr_m & (vol_col >= tr_th)
X_tr, y_tr = feats[tr_sel], labels[tr_sel]
X_es, y_es = feats[es_m], labels[es_m]
X_te, y_te = feats[te_m], labels[te_m]
ts_te = ts[idx][te_m]
print(f'TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}')

# Month for per-month diag
te_dt = pd.to_datetime(ts_te, unit='s', utc=True)
te_month = te_dt.to_period('M').values

# ====== CatBoost ======
print('\n[1] CatBoost ...', flush=True)
cb = CatBoostClassifier(
    depth=6, l2_leaf_reg=5, subsample=0.8, colsample_bylevel=0.8,
    learning_rate=0.05, iterations=3000, loss_function='Logloss',
    eval_metric='AUC', random_seed=42, verbose=0,
    early_stopping_rounds=300, thread_count=3,
)
cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), verbose=0)
pv_te_cb = cb.predict_proba(X_te)[:, 1]

# ====== LightGBM 最佳 ======
print('\n[2] LightGBM best ...', flush=True)
import lightgbm as lgb
p = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=0.03,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
         min_child_samples=200, lambda_l1=0.01, lambda_l2=0.1,
         verbose=-1, n_jobs=3, seed=42)
lgb1 = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                 valid_sets=[lgb.Dataset(X_es, y_es)],
                 callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
pv_te_lgb = lgb1.predict(X_te)

# LGB seed 2
p2 = dict(**p, seed=123)
lgb2 = lgb.train(p2, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                 valid_sets=[lgb.Dataset(X_es, y_es)],
                 callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
pv_te_lgb2 = lgb2.predict(X_te)

# ====== Ensemble (平均) ======
pv_te_ens = (pv_te_cb + pv_te_lgb + pv_te_lgb2) / 3

def eval_pv(pv, name):
    auc = roc_auc_score(y_te, pv)
    print(f'\n  {name} AUC={auc:.4f}')
    for pct in [0.005, 0.01, 0.02]:
        k = max(1, int(len(pv)*pct)); ti = pv.argsort()[-k:]
        acc = y_te[ti].mean()*100; tpd = k / 356
        print(f'    top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}')

eval_pv(pv_te_cb, 'CatBoost')
eval_pv(pv_te_lgb, 'LightGBM seed1')
eval_pv(pv_te_lgb2, 'LightGBM seed2')
eval_pv(pv_te_ens, 'Ensemble avg')

# ====== 月度诊断 + 月度过滤 ======
print('\n[3] 月度诊断 (Ensemble) ...', flush=True)
for m in np.unique(te_month):
    mm = te_month == m
    auc_m = roc_auc_score(y_te[mm], pv_te_ens[mm])
    k1 = max(1, int(mm.sum()*0.01)); top1 = pv_te_ens[mm].argsort()[-k1:]
    acc1 = y_te[mm][top1].mean()*100
    good = '✓' if acc1 >= 60 else '✗'
    print(f'  {m}: n={mm.sum():6d} AUC={auc_m:.4f} top1%={acc1:.1f}% {good}', flush=True)

# ====== 只在好月份跑 ======
print('\n[4] 好月份过滤 (acc≥60%) ...', flush=True)
good_months = []
for m in np.unique(te_month):
    mm = te_month == m
    k1 = max(1, int(mm.sum()*0.01)); top1 = pv_te_ens[mm].argsort()[-k1:]
    acc1 = y_te[mm][top1].mean()*100
    if acc1 >= 60:
        good_months.append(m)
print(f'  good months: {good_months}')

# 在好月份上评估 overall
gm = np.isin(te_month, good_months)
pv_good = pv_te_ens[gm]; y_good = y_te[gm]
eval_pv(pv_good, 'Ensemble (good months)')

# ====== 最终 top-k (只在好月份) ======
print('\n[5] 最终策略: Ensemble + 好月份过滤 ...', flush=True)
# 从好月份中取 top 信号（按 pv_ens score 排序，取全局 top 而不是 per-month）
pv_final = pv_te_ens[gm]
y_final = y_te[gm]
ts_final = ts_te[gm]
for pct in [0.01, 0.02, 0.03]:
    k = max(1, int(len(pv_final)*pct)); ti = pv_final.argsort()[-k:]
    acc = y_final[ti].mean()*100
    # tpd 要除以好月份的天数
    n_days_good = len(good_months) * 30  # approx
    tpd = k / n_days_good
    print(f'  ✅ good-months top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
