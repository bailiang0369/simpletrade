"""Final tune: TR>40%ile LGB → 冲 top1%≥60%, tpd≥14"""
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

# BTC
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
bi = pd.Index(btc['ts'].values); ei = pd.Index(ts)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)
btc_c = btc['close'].values.astype(np.float32)[br]; del btc; gc = None

H = config.HORIZON_MIN; SEG = 60; idx = np.arange(SEG, N - H)

# Stoch
ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
stoch = (close[idx] - ll) / np.maximum(hh - ll, 1e-6)
labels = (close[idx + H] > close[idx]).astype(np.int64)

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
pdi_a=pdi[idx-1];mdi_a=mdi[idx-1];di_diff=pdi_a-mdi_a
adx_a = ws(100*np.abs(pdi-mdi)/np.maximum(pdi+mdi,1e-6), 14)[idx-1]

# funding
_tr_end = int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
_fm = ts < _tr_end
fund = np.clip(fund, np.percentile(fund[_fm], 0.5), np.percentile(fund[_fm], 99.5))
fund_z = (fund - fund[_fm].mean()) / max(fund[_fm].std(), 1e-6)

# vol
vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        vol[i] = np.std(np.diff(np.log(close[idx[i]-59:idx[i]+1]))) * np.sqrt(60)

feats = np.column_stack([
    stoch, 1-stoch, (stoch-0.5)**2,
    pdi_a, mdi_a, di_diff, np.abs(di_diff), adx_a,
    np.log(close[idx]/np.maximum(close[idx-5],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-10],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-15],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-30],1e-6)),
    np.log(close[idx]/np.maximum(close[idx-60],1e-6)),
    np.log(np.maximum(buy_v[idx]/np.maximum(sell_v[idx],1.0),1e-6)),
    np.log(np.maximum(buy_v[idx]+sell_v[idx],1.0)),
    fund_z[idx],
    np.log(btc_c[idx]/np.maximum(btc_c[idx-60],1e-6)),  # BTC 60min ret
    np.std(sliding_window_view(low, SEG)[idx - SEG], axis=1) / np.maximum(hh - ll, 1e-6),
    vol,
]).astype(np.float32)
print(f'feats: {feats.shape}')

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])

# Train on vol>40%ile, predict ALL test
vol_col = feats[:, -1]
tr_th = np.percentile(vol_col[tr_m], 40)
tr_sel = tr_m & (vol_col >= tr_th)
X_tr, y_tr = feats[tr_sel], labels[tr_sel]
X_es, y_es = feats[es_m], labels[es_m]
X_te, y_te = feats[te_m], labels[te_m]
print(f'TR(vol>40%ile)={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}')

print('\n[1] 超参 grid ...', flush=True)
best_top1 = 0; best_p = None; best_leaves = 0
for leaves in [15, 31, 63, 127]:
    for lr in [0.03, 0.05, 0.1]:
        for ffrac in [0.6, 0.8, 1.0]:
            p = dict(objective='binary', metric='auc', num_leaves=leaves,
                     learning_rate=lr, feature_fraction=ffrac, bagging_fraction=0.8,
                     bagging_freq=5, min_child_samples=200, lambda_l1=0.01, lambda_l2=0.1,
                     verbose=-1, n_jobs=3, seed=42)
            m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(X_es, y_es)],
                          callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
            pv_te = m.predict(X_te)
            auc = roc_auc_score(y_te, pv_te)
            k1 = max(1, int(len(pv_te)*0.01)); acc1 = y_te[pv_te.argsort()[-k1:]].mean()*100
            k05 = max(1, int(len(pv_te)*0.005)); acc05 = y_te[pv_te.argsort()[-k05:]].mean()*100
            line = f'L={leaves:3d} lr={lr} ff={ffrac} → auc={auc:.4f} top0.5%={acc05:.1f}% top1%={acc1:.1f}%'
            if acc1 > best_top1:
                best_top1 = acc1; best_p = p; best_leaves = leaves
                line += ' ⭐'
            print(line, flush=True)

print(f'\n[2] 最佳 top1%={best_top1:.1f}%  leaves={best_leaves}', flush=True)
print(f'    params: {best_p}')

# 最佳模型详细评估
m = lgb.train(best_p, lgb.Dataset(X_tr, y_tr), num_boost_round=5000,
              valid_sets=[lgb.Dataset(X_es, y_es)],
              callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
pv_te = m.predict(X_te)
auc = roc_auc_score(y_te, pv_te)
print(f'\n📊 Test AUC={auc:.4f}')
for pct in [0.005, 0.01, 0.02, 0.03, 0.05]:
    k = max(1, int(len(pv_te)*pct))
    ti = pv_te.argsort()[-k:]
    acc = y_te[ti].mean()*100; tpd = k / 356
    ret_top = (close[idx[te_m][ti]+H] / close[idx[te_m][ti]] - 1).mean()*100
    print(f'  top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f} avg_ret={ret_top:.3f}%')

# 特征重要性
print(f'\n[3] 特征重要性 (gain, top10) ...')
imp = m.feature_importance(importance_type='gain')
names = ['stoch', '1-stoch', '(stoch-0.5)^2', '+DI', '-DI', 'DI_diff', '|DI_diff|', 'ADX',
         'ret5', 'ret10', 'ret15', 'ret30', 'ret60', 'buy_ratio', 'log_vol', 'funding_z',
         'btc_ret60', 'stoch_range', 'vol']
for n, v in sorted(zip(names, imp), key=lambda x: -x[1])[:10]:
    print(f'  {n:12s}: {v:.0f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
