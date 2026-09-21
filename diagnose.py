"""诊断: horizon 扫描 + 复现手动交易逻辑"""
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

# === 预计算 Stoch(60,1,1) + DMI ===
def compute_indicators(H_target):
    """在 horizon=H_target 下计算指标 + label"""
    SEG = 60
    idx = np.arange(SEG, N - H_target)
    # Stoch
    ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
    hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
    c_now = close[idx]
    stoch = (c_now - ll) / np.maximum(hh - ll, 1e-6)
    
    # DMI
    h_l = high[1:] - low[1:]
    h_cp = np.abs(high[1:] - close[:-1])
    l_cp = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    def ws(a, p):
        o = np.zeros(len(a)); o[0] = a[0]
        for i in range(1, len(a)):
            o[i] = o[i-1] - o[i-1]/p + a[i]
        return o
    tr_s = ws(tr, 14); pdm_s = ws(pdm, 14); mdm_s = ws(mdm, 14)
    pdi = 100 * pdm_s / np.maximum(tr_s, 1e-6)
    mdi = 100 * mdm_s / np.maximum(tr_s, 1e-6)
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-6)
    adx = ws(dx, 14)
    
    dmi_idx = idx - 1
    valid = dmi_idx >= 0
    return stoch[valid], pdi[dmi_idx][valid], mdi[dmi_idx][valid], adx[dmi_idx][valid], idx[valid]

def mk(ts_aligned, s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_aligned >= a) & (ts_aligned < b)

# === 扫 horizon ===
print('[1] Horizon 扫描 (Stoch+DMI) ...', flush=True)
for H in [1, 3, 5, 10, 15, 30, 60]:
    stoch, pdi, mdi, adx, idx = compute_indicators(H)
    labels = (close[idx + H] > close[idx]).astype(np.int64)
    ret = close[idx + H] / close[idx] - 1
    ts_a = ts[idx]
    
    # 只用 10 个最核心特征
    feats = np.column_stack([
        stoch, 1-stoch, (stoch-0.5)**2, pdi, mdi, pdi-mdi, np.abs(pdi-mdi), adx,
        np.log(buy_v[idx]/np.maximum(sell_v[idx], 1.0)),
        np.log(buy_v[idx]+sell_v[idx]),
    ]).astype(np.float32)
    
    tr_m = mk(ts_a, *config.SPLITS['train'])
    es_m = mk(ts_a, *config.SPLITS['early_stop'])
    te_m = mk(ts_a, *config.SPLITS['test'])
    
    p = dict(objective='binary', metric='auc', num_leaves=31, learning_rate=0.05,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             min_child_samples=200, verbose=-1, n_jobs=3, seed=42)
    m = lgb.train(p, lgb.Dataset(feats[tr_m], labels[tr_m]), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(feats[es_m], labels[es_m])],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pv = m.predict(feats[te_m])
    auc = roc_auc_score(labels[te_m], pv)
    k1 = max(1, int(len(pv)*0.01)); top1 = pv.argsort()[-k1:]
    acc1 = labels[te_m][top1].mean()*100
    ret_top1 = ret[te_m][top1].mean()*100
    print(f'  H={H:2d}: AUC={auc:.4f}  top1%={acc1:.1f}%  avg_ret_top1={ret_top1:.3f}%  '
          f'te_pos={labels[te_m].mean():.3f} ({time.time()-t0:.0f}s)', flush=True)

# === 手动交易逻辑复现 ===
print(f'\n[2] 复现手动交易逻辑 (H=5) ...', flush=True)
H = 5
stoch, pdi, mdi, adx, idx = compute_indicators(H)
labels = (close[idx + H] > close[idx]).astype(np.int64)
ret = close[idx + H] / close[idx] - 1
ts_a = ts[idx]

# 逻辑 1: Stoch oversold(<%K<10) + +DI 上穿 -DI → 做多
#        Stoch overbought(%K>90) + -DI 上穿 +DI → 做空
# 简化: Stoch<15 且 +DI > -DI → 做多信号
#       Stoch>85 且 -DI > +DI → 做空信号
long_sig = (stoch < 0.15) & (pdi > mdi)
short_sig = (stoch > 0.85) & (mdi > pdi)
all_sig = long_sig | short_sig

te_m = mk(ts_a, *config.SPLITS['test'])
n_long_te = long_sig[te_m].sum()
n_short_te = short_sig[te_m].sum()
acc_long = labels[te_m][long_sig[te_m]].mean()*100 if n_long_te>0 else 0
acc_short = (1 - labels[te_m][short_sig[te_m]]).mean()*100 if n_short_te>0 else 0
acc_all = (labels[te_m][long_sig[te_m]].tolist() + 
           (1 - labels[te_m][short_sig[te_m]]).tolist())
acc_all = np.mean(acc_all)*100 if acc_all else 0

# 每天交易数
n_days = 356
tpd = (n_long_te + n_short_te) / n_days
avg_ret = ret[te_m][all_sig[te_m]].mean()*100 if all_sig[te_m].sum()>0 else 0

print(f'  信号数: 做多={n_long_te} 做空={n_short_te} 合计={n_long_te+n_short_te}')
print(f'  胜率: 做多={acc_long:.1f}% 做空={acc_short:.1f}% 合计={acc_all:.1f}%')
print(f'  每天信号: {tpd:.1f}  平均ret={avg_ret:.3f}%')

# 逻辑 2: 单指标阈值 - 只用 Stoch
print(f'\n[3] 纯 Stoch(60) 阈值扫描 H={H} ...', flush=True)
te_stoch = stoch[te_m]
te_labels = labels[te_m]
for lo_th, hi_th in [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.05, 0.95)]:
    sig = (te_stoch < lo_th) | (te_stoch > hi_th)
    n = sig.sum()
    tp = (te_stoch < lo_th)
    sp = (te_stoch > hi_th)
    acc_tp = te_labels[tp].mean()*100 if tp.sum()>0 else 0
    acc_sp = (1 - te_labels[sp]).mean()*100 if sp.sum()>0 else 0
    tpd_ = n / n_days
    print(f'  Stoch<{lo_th} / Stoch>{hi_th}: n={n} tpd≈{tpd_:.1f} '
          f'long_acc={acc_tp:.1f}% short_acc={acc_sp:.1f}%')

# 逻辑 3: Stoch + ADX 过滤 (只在有趋势时交易)
print(f'\n[4] Stoch + ADX 过滤 H={H} ...', flush=True)
te_adx = adx[te_m]
for adx_min in [10, 15, 20, 25]:
    sig = ((te_stoch < 0.15) | (te_stoch > 0.85)) & (te_adx > adx_min)
    n = sig.sum(); tpd_ = n / n_days
    # long
    tp = (te_stoch < 0.15) & (te_adx > adx_min)
    sp = (te_stoch > 0.85) & (te_adx > adx_min)
    acc_tp = te_labels[tp].mean()*100 if tp.sum()>0 else 0
    acc_sp = (1 - te_labels[sp]).mean()*100 if sp.sum()>0 else 0
    acc_all2 = (te_labels[tp].tolist() + (1-te_labels[sp]).tolist())
    acc_all2 = np.mean(acc_all2)*100 if acc_all2 else 0
    print(f'  ADX>{adx_min}: n={n} tpd≈{tpd_:.1f} acc_all={acc_all2:.1f}% '
          f'long={acc_tp:.1f}% short={acc_sp:.1f}%')

print(f'\n⏱ {time.time()-t0:.0f}s')
