"""严格流程: train → ES early_stop → meta_val 选阈值 → 最终 test 评估"""
import numpy as np, pandas as pd, time
from numpy.lib.stride_tricks import sliding_window_view
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
mv_m = mk(*config.SPLITS['meta_val'])
te_m = mk(*config.SPLITS['test'])

# Train on vol>40%ile
vol_col = feats[:, -1]
tr_th = np.percentile(vol_col[tr_m], 40)
tr_sel = tr_m & (vol_col >= tr_th)
X_tr, y_tr = feats[tr_sel], labels[tr_sel]
X_es, y_es = feats[es_m], labels[es_m]
X_mv, y_mv = feats[mv_m], labels[mv_m]
X_te, y_te = feats[te_m], labels[te_m]
ts_mv = ts[idx][mv_m]
ts_te = ts[idx][te_m]
print(f'TR={len(X_tr):,} ES={len(X_es):,} MV={len(X_mv):,} TE={len(X_te):,}')

# 训练 3 个 seed 做 ensemble
print('\n[1] 训练 ensemble ...', flush=True)
p1 = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=0.03,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
          min_child_samples=200, lambda_l1=0.01, lambda_l2=0.1, verbose=-1, n_jobs=3, seed=42)

pv_mv_all, pv_te_all = [], []
for seed in [42, 123, 777]:
    p = {**p1, 'seed': seed}
    m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(X_es, y_es)],
                  callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
    pv_mv_all.append(m.predict(X_mv))
    pv_te_all.append(m.predict(X_te))

pv_mv = np.mean(pv_mv_all, axis=0)
pv_te = np.mean(pv_te_all, axis=0)

auc_mv = roc_auc_score(y_mv, pv_mv)
auc_te = roc_auc_score(y_te, pv_te)
print(f'  MV AUC={auc_mv:.4f}  TE AUC={auc_te:.4f}')

# ============ Meta-Val 阈值选择 ============
# 目标: 在 meta_val 上找 top-k 使得 acc>=65% + tpd>=14
print('\n[2] Meta-Val 阈值/TopK 选择 ...', flush=True)
n_days_mv = 365  # 近似
best_plan = None
for pct_name, pct in [('top0.5%', 0.005), ('top1%', 0.01), ('top1.5%', 0.015),
                      ('top2%', 0.02), ('top3%', 0.03), ('top5%', 0.05), ('top10%', 0.10)]:
    k = max(1, int(len(pv_mv) * pct))
    ti = pv_mv.argsort()[-k:]
    acc = y_mv[ti].mean() * 100
    tpd = k / n_days_mv
    good = '🎯' if acc >= 65 and tpd >= 14 else ('★' if acc >= 60 and tpd >= 14 else '')
    print(f'  MV {pct_name}: acc={acc:.1f}% tpd={tpd:.1f} {good}', flush=True)

# 选 pv 阈值方式: pv > th 而不是 top-k
print('\n[3] Meta-Val pv-score 阈值扫描 ...', flush=True)
for th in [0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.54]:
    sig = pv_mv > th
    n = sig.sum()
    if n < 10: continue
    # 预测 1 (涨) if pv>th, 预测 0 (跌) if pv<1-th
    # 简化: 只看 top pv (long only)
    acc = y_mv[sig].mean() * 100
    tpd = n / n_days_mv
    print(f'  MV pv>{th}: n={n} acc={acc:.1f}% tpd={tpd:.1f}', flush=True)

# ============ Test 最终评估 ============
print('\n[4] Test 最终评估 (严格) ...', flush=True)
for pct_name, pct in [('top0.5%', 0.005), ('top1%', 0.01), ('top1.5%', 0.015),
                      ('top2%', 0.02), ('top3%', 0.03), ('top5%', 0.05)]:
    k = max(1, int(len(pv_te) * pct))
    ti = pv_te.argsort()[-k:]
    acc = y_te[ti].mean() * 100
    tpd = k / 356
    ret_top = (close[idx[te_m][ti] + H] / close[idx[te_m][ti]] - 1).mean() * 100
    flag = '🎯' if acc >= 65 and tpd >= 14 else ('★' if acc >= 60 and tpd >= 14 else '')
    print(f'  {flag} TE {pct_name}: acc={acc:.1f}% tpd={tpd:.1f} avg_ret={ret_top:.3f}%', flush=True)

# Test 月度分布 (仅事后分析, 不用于选阈值)
print('\n[5] Test 月度分布 (事后) ...', flush=True)
te_dt = pd.to_datetime(ts_te, unit='s', utc=True)
te_month = te_dt.to_period('M').astype(str).values
for m in sorted(np.unique(te_month)):
    mm = te_month == m
    k1 = max(1, int(mm.sum()*0.01)); ti = pv_te[mm].argsort()[-k1:]
    acc1 = y_te[mm][ti].mean()*100
    print(f'  {m}: top1%={acc1:.1f}% n={mm.sum():,}', flush=True)

print(f'\n⏱ {time.time()-t0:.0f}s')
