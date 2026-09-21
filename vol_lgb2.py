"""Vol-filtered LightGBM v2 — 修 mask bug + 扫描"""
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
pdi_a=pdi[idx-1];mdi_a=mdi[idx-1];di_diff=pdi_a-mdi_a; abs_di=np.abs(di_diff)
dx = 100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-6)
adx_a = ws(dx, 14)[idx-1]

# Vol
vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        seg = close[idx[i]-59:idx[i]+1]
        vol[i] = np.std(np.diff(np.log(seg))) * np.sqrt(60)

# Features
ret15 = np.log(close[idx] / np.maximum(close[idx-15], 1e-6))
ret30 = np.log(close[idx] / np.maximum(close[idx-30], 1e-6))
ret60 = np.log(close[idx] / np.maximum(close[idx-60], 1e-6))
buy_r = np.log(np.maximum(buy_v[idx] / np.maximum(sell_v[idx], 1.0), 1e-6))
log_vol = np.log(np.maximum(buy_v[idx] + sell_v[idx], 1.0))
stoch_d = np.zeros(len(idx))
stoch_d[1:] = stoch[1:] - stoch[:-1]

feats = np.column_stack([
    stoch, 1-stoch, (stoch-0.5)**2, stoch_d,
    pdi_a, mdi_a, di_diff, abs_di, adx_a,
    ret15, ret30, ret60, buy_r, log_vol, vol,
]).astype(np.float32)

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])

print(f'TR={tr_m.sum():,} ES={es_m.sum():,} TE={te_m.sum():,}', flush=True)

vol_col = feats[:, -1]  # vol

# ======= Configs: vol filter on train AND test =======
print('\n[1] Vol-filtered 训练+预测 ...', flush=True)
for tr_v, te_v, name in [
    (0.0, 0.0, '全量'),
    (0.4, 0.0, 'TR>40%ile (全量test)'),
    (0.4, 0.4, 'TR+TE 都>40%ile'),
    (0.5, 0.5, 'TR+TE 都>50%ile'),
    (0.6, 0.6, 'TR+TE 都>60%ile'),
    (0.7, 0.7, 'TR+TE 都>70%ile'),
]:
    tr_th = np.percentile(vol_col[tr_m], tr_v * 100) if tr_v > 0 else -999
    te_th = np.percentile(vol_col[te_m], te_v * 100) if te_v > 0 else -999
    
    tr_sel = tr_m & (vol_col >= tr_th)
    te_sel = te_m & (vol_col >= te_th)
    
    X_tr, y_tr = feats[tr_sel], labels[tr_sel]
    X_es, y_es = feats[es_m], labels[es_m]
    X_te, y_te = feats[te_sel], labels[te_sel]
    
    print(f'\n  === {name} TR={len(X_tr):,} TE={len(X_te):,} ===', flush=True)
    for leaves in [15, 31]:
        p = dict(objective='binary', metric='auc', num_leaves=leaves, learning_rate=0.05,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                 min_child_samples=200, verbose=-1, n_jobs=3, seed=42)
        m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X_es, y_es)],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        pv_te = m.predict(X_te)
        auc = roc_auc_score(y_te, pv_te)
        line = f'    L={leaves:3d}: auc={auc:.4f}'
        for pct in [0.01, 0.02, 0.05, 0.10]:
            k = max(1, int(len(pv_te)*pct))
            ti = pv_te.argsort()[-k:]
            acc = y_te[ti].mean()*100
            tpd = k / 356
            line += f'  top{pct*100:.0f}%={acc:.1f}%/tpd={tpd:.1f}'
        print(line, flush=True)

# ======= Configs: 预测时只用 Stoch+DI 高置信样本 (threshold + LGB 排序) =======
print('\n[2] 高置信子集 → LGB 排序 ...', flush=True)
for conf_th in [0, 0.01, 0.05, 0.10, 0.15]:
    # 先用全量 LGB 预测 test，然后只在 top/bottom conf_th 里取
    # 或者先选高置信 bar，再排序
    # 高置信 = |stoch - 0.5| > threshold (极端位置) | abs_di > th
    tr_conf = (np.abs(stoch[tr_m] - 0.5) > 0.35) | (abs_di[tr_m] > 20)
    te_conf = (np.abs(stoch[te_m] - 0.5) > 0.35) | (abs_di[te_m] > 20)
    if conf_th > 0:
        # 只保留更极端的
        tr_conf = np.abs(stoch[tr_m] - 0.5) > 0.5 - conf_th
        te_conf = np.abs(stoch[te_m] - 0.5) > 0.5 - conf_th
    
    tr_sel = tr_m & tr_conf
    te_sel = te_m & te_conf
    if te_sel.sum() < 500: continue
    
    X_tr, y_tr = feats[tr_sel], labels[tr_sel]
    X_es, y_es = feats[es_m], labels[es_m]
    X_te, y_te = feats[te_sel], labels[te_sel]
    
    p = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=0.05,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             min_child_samples=200, verbose=-1, n_jobs=3, seed=42)
    m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(X_es, y_es)],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pv_te = m.predict(X_te)
    auc = roc_auc_score(y_te, pv_te)
    
    print(f'\n  conf_th={conf_th}: TR={len(X_tr):,} TE={len(X_te):,} auc={auc:.4f}', flush=True)
    line = f'    L=31:'
    for pct in [0.05, 0.10, 0.20, 0.30]:
        k = max(1, int(len(pv_te)*pct))
        ti = pv_te.argsort()[-k:]
        acc = y_te[ti].mean()*100
        tpd = k / 356
        line += f'  top{pct*100:.0f}%={acc:.1f}%/tpd={tpd:.1f}'
    print(line, flush=True)

print(f'\n⏱ {time.time()-t0:.0f}s')
