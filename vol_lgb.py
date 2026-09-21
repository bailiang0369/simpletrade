"""Vol-filtered LightGBM：高波动子集上训练 + top-k 排序"""
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
ts = eth['ts'].values.astype(np.int64)
N = len(close)

H = config.HORIZON_MIN; SEG = 60; idx = np.arange(SEG, N - H)

# Stoch + labels
ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
c_now = close[idx]
stoch = (c_now - ll) / np.maximum(hh - ll, 1e-6)
labels = (close[idx + H] > c_now).astype(np.int64)

# DMI
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
pdi=100*pdm_s/np.maximum(tr_s,1e-6);mdi=100*mdm_s/np.maximum(tr_s,1e-6)
pdi_a=pdi[idx-1];mdi_a=mdi[idx-1];di_diff=pdi_a-mdi_a; abs_di=np.abs(di_diff)
dx = 100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-6)
adx = ws(dx, 14); adx_a = adx[idx-1]

# Volatility
vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        seg = close[idx[i]-59:idx[i]+1]
        vol[i] = np.std(np.diff(np.log(seg))) * np.sqrt(60)

# 更多特征
ret15 = np.log(close[idx] / np.maximum(close[idx-15], 1e-6))
ret30 = np.log(close[idx] / np.maximum(close[idx-30], 1e-6))
ret60 = np.log(close[idx] / np.maximum(close[idx-60], 1e-6))
buy_r = np.log(np.maximum(buy_v[idx] / np.maximum(sell_v[idx], 1.0), 1e-6))
log_vol = np.log(np.maximum(buy_v[idx] + sell_v[idx], 1.0))
stoch_d = np.zeros(len(idx))
stoch_d[1:] = stoch[1:] - stoch[:-1]
# Stoch 分档
stoch_overbought = (stoch > 0.85).astype(np.float32)
stoch_oversold = (stoch < 0.15).astype(np.float32)

feats = np.column_stack([
    stoch, 1-stoch, (stoch-0.5)**2, stoch_d, stoch_overbought, stoch_oversold,
    pdi_a, mdi_a, di_diff, abs_di, adx_a,
    ret15, ret30, ret60,
    buy_r, log_vol, vol,
]).astype(np.float32)
print(f'features: {feats.shape}')

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])

tr_vol = feats[tr_m, -1]  # vol 在最后一列

print('\n[1] 全量 vs vol-filtered LGB ...', flush=True)

configs = [
    ('全量', np.ones(len(tr_m), bool), np.ones(len(te_m), bool)),
    ('vol>40%ile', tr_vol > np.percentile(tr_vol, 40),
                 feats[te_m, -1] > np.percentile(feats[te_m, -1], 40)),
    ('vol>50%ile', tr_vol > np.percentile(tr_vol, 50),
                 feats[te_m, -1] > np.percentile(feats[te_m, -1], 50)),
    ('vol>60%ile', tr_vol > np.percentile(tr_vol, 60),
                 feats[te_m, -1] > np.percentile(feats[te_m, -1], 60)),
    ('vol>70%ile', tr_vol > np.percentile(tr_vol, 70),
                 feats[te_m, -1] > np.percentile(feats[te_m, -1], 70)),
]

for name, tr_vf, te_vf in configs:
    X_tr = feats[tr_m & tr_vf]; y_tr = labels[tr_m & tr_vf]
    X_es = feats[es_m]; y_es = labels[es_m]
    X_te = feats[te_m & te_vf]; y_te = labels[te_m & te_vf]
    n_days_te = 356  # test 覆盖 ~356 天
    print(f'\n  === {name}: TR={len(X_tr):,} TE={len(X_te):,} ===', flush=True)

    best_auc = 0
    for leaves in [15, 31, 63]:
        p = dict(objective='binary', metric='auc', num_leaves=leaves, learning_rate=0.05,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                 min_child_samples=200, verbose=-1, n_jobs=3, seed=42)
        m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X_es, y_es)],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        pv_te = m.predict(X_te)
        auc = roc_auc_score(y_te, pv_te)
        line = f'    L={leaves:3d}: auc={auc:.4f}'
        for pct in [0.01, 0.02, 0.03, 0.05]:
            k = max(1, int(len(pv_te)*pct))
            ti = pv_te.argsort()[-k:]
            acc = y_te[ti].mean()*100
            tpd = k / n_days_te
            line += f'  top{pct*100:.0f}%={acc:.1f}%(tpd={tpd:.1f})'
        print(line, flush=True)
        if auc > best_auc: best_auc = auc

print(f'\n⏱ {time.time()-t0:.0f}s')
