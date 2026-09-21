"""快速测：只用 CNN 的 6 个原始特征（逐 bar 值），LightGBM 能到多少 AUC？"""
import numpy as np, pandas as pd, gc, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config

t0 = time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')

eth_idx = pd.Index(eth['ts'].values)
btc_reorder = np.clip(btc['ts'].values.searchsorted(eth_idx.values, side='right') - 1, 0, len(btc)-1)
close = eth['close'].values.astype(np.float64)
buy_v = eth['buy_vol'].values.astype(np.float64)
sell_v = eth['sell_vol'].values.astype(np.float64)
fund = eth['funding'].values.astype(np.float64)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float64)[btc_reorder]

# 清洗 funding
_tr_end = int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
_tr_m = ts < _tr_end
fp01 = np.percentile(fund[_tr_m & (fund > -100)], 0.1)
fp999 = np.percentile(fund[_tr_m & (fund < 100)], 99.9)
fund = np.clip(fund, fp01, fp999)
fund_z = (fund - fund[_tr_m].mean()) / max(fund[_tr_m].std(), 1e-6)

N = len(close); H = config.HORIZON_MIN; SEG = 64; STRIDE = 32
rows, tgt, tss = [], [], []

for start in range(0, N - SEG - H, STRIDE):
    end = start + SEG
    if end + H > N: break
    if np.isnan(close[end]): continue
    c0 = close[start]; bc0 = btc_c[start]
    w = np.zeros(6 * SEG, np.float64)
    w[0::6] = np.log(close[start:end] / max(c0, 1e-6))      # close_ret
    bs_win = buy_v[start:end] - sell_v[start:end]
    cum_bs = np.cumsum(bs_win)
    w[1::6] = cum_bs / max(np.abs(cum_bs[-1]), 1.0)          # cvd_diff
    br = np.log(np.maximum(buy_v[start:end] / np.maximum(sell_v[start:end], 1.0), 1e-6))
    w[2::6] = br
    tv = np.log(np.maximum(buy_v[start:end] + sell_v[start:end], 1.0))
    w[3::6] = tv - tv.mean()
    w[4::6] = fund_z[start:end]
    w[5::6] = np.log(btc_c[start:end] / max(bc0, 1e-6))
    if np.isnan(w).any(): continue
    rows.append(w); tgt.append(1 if close[end+H] > close[end] else 0)
    tss.append(ts[end])

X = np.array(rows, np.float32); y = np.array(tgt, np.int64); ts_all = np.array(tss)
del rows, tgt, buy_v, sell_v, fund, close, btc_c; gc.collect()
print(f'构建 {X.shape} pos={y.mean():.3f} ({time.time()-t0:.0f}s)', flush=True)

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_all >= a) & (ts_all < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])
X_tr, y_tr = X[tr_m], y[tr_m]
X_es, y_es = X[es_m], y[es_m]
X_te, y_te = X[te_m], y[te_m]
del X; gc.collect()
print(f'TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}')

# 试不同配置
for leaves in [31, 63, 127]:
    for lr in [0.05, 0.1]:
        p = dict(
            objective='binary', metric='auc', num_leaves=leaves,
            learning_rate=lr, feature_fraction=0.5, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=100, lambda_l1=0.01, lambda_l2=0.1,
            verbose=-1, n_jobs=3, seed=42,
        )
        m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X_es, y_es)],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        pv_te = m.predict(X_te)
        auc = roc_auc_score(y_te, pv_te)
        top1 = pv_te.argsort()[-max(1, int(len(pv_te)*0.01)):]
        acc1 = y_te[top1].mean()*100
        print(f'  leaves={leaves:3d} lr={lr} → te_auc={auc:.4f} top1%={acc1:.1f}%  ({time.time()-t0:.0f}s)', flush=True)

print(f'\n⏱ {time.time()-t0:.0f}s')
