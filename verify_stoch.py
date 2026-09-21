"""快速验证: Stoch(60,1,1) + DMI + close → top1% accuracy"""
import numpy as np, pandas as pd, gc, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config

t0 = time.time()
print('[1] 加载 + 计算 Stoch/DMI ...', flush=True)

eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
H = config.HORIZON_MIN  # 15
SEG = 60  # Stoch period
close = eth['close'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low = eth['low'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
N = len(close)

# === Stochastic(60,1,1) ===
# %K = (C - LL) / (HH - LL)  in past 60 bars
# %D = SMA(%K, 1)  (= %K itself with smooth=1)
window = SEG  # 60
# Rolling LL and HH — vectorized
# Use stride tricks for fast rolling
from numpy.lib.stride_tricks import sliding_window_view

print('  computing rolling stats...', flush=True)
# 只需要 end=SEG:end=N-H 范围的
idx = np.arange(window, N - H)
ll = np.min(sliding_window_view(low, window)[idx - window], axis=1)
hh = np.max(sliding_window_view(high, window)[idx - window], axis=1)
c_now = close[idx]
hk = hh - ll
hk = np.maximum(hk, 1e-6)
stoch_k = (c_now - ll) / hk  # 0-1 range

# === DMI: +DI / -DI (ATR-based) ===
# TR = max(H-L, |H-Cp|, |L-Cp|)
# +DM, -DM based on up move vs down move
# ADX period=14, DI period=14
period_dmi = 14
print('  computing DMI...', flush=True)
# TR
h_l = high[1:] - low[1:]
h_cp = np.abs(high[1:] - close[:-1])
l_cp = np.abs(low[1:] - close[:-1])
tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
# DM
up_move = high[1:] - high[:-1]
down_move = low[:-1] - low[1:]
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

# Wilder smoothing (EMA with alpha=1/period)
def wilder_smooth(arr, p):
    out = np.zeros(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = out[i-1] - out[i-1]/p + arr[i]
    return out

tr_smooth = wilder_smooth(tr, period_dmi)
pdm_smooth = wilder_smooth(plus_dm, period_dmi)
mdm_smooth = wilder_smooth(minus_dm, period_dmi)

pdi = 100 * pdm_smooth / np.maximum(tr_smooth, 1e-6)
mdi = 100 * mdm_smooth / np.maximum(tr_smooth, 1e-6)
dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-6)
adx = wilder_smooth(dx, period_dmi)

# Align with Stoch indices (which start at idx=60)
# DMI arrays start at index 1 (shifted by 1 because of diff)
# We need DMI values at time t (which is close[idx] time)
# tr/pdm/mdm are length N-1, aligned with eth[1:N]
# close[idx] where idx >= 60 corresponds to original index idx
# DMI at original index idx: dmi[idx-1] (because DMI was computed on [1:])
# But actually wilder_smooth is applied in order, so dmi[i] corresponds to tr[i]
# which is eth[i+1]. So dmi at eth index t is dmi[t-1] where t>=1

# Align: close[idx] at original index idx (>=SEG)
# DMI at original index idx = pdi[idx-1], mdi[idx-1], adx[idx-1] where idx>=1
dmi_idx = idx - 1  # shift
valid = dmi_idx >= 0
pdi_v = pdi[dmi_idx][valid]
mdi_v = mdi[dmi_idx][valid]
adx_v = adx[dmi_idx][valid]
stoch_v = stoch_k[valid]
c_aligned = close[idx][valid]
ts_aligned = ts[idx][valid]
labels = (close[idx + H][valid] > c_aligned).astype(np.int64)
ret_vals = close[idx + H][valid] / c_aligned - 1

N2 = len(stoch_v)
print(f'  aligned N={N2:,}')

# === 构建特征 ===
# 只用 Stoch + DMI + 少量衍生
buy_ratio = np.log(np.maximum(buy_v[idx][valid] / np.maximum(sell_v[idx][valid], 1.0), 1e-6))
tot_vol = np.log(np.maximum(buy_v[idx][valid] + sell_v[idx][valid], 1.0))

features = np.column_stack([
    stoch_v,            # [0] Stoch %K 60
    1 - stoch_v,        # [1] inverse (for oversold)
    (stoch_v - 0.5)**2, # [2] distance from mid
    pdi_v,              # [3] +DI
    mdi_v,              # [4] -DI
    pdi_v - mdi_v,      # [5] DI diff (趋势方向)
    adx_v,              # [6] ADX (趋势强度)
    np.abs(pdi_v - mdi_v), # [7] abs DI diff
    buy_ratio,          # [8] log(buy/sell)
    tot_vol,            # [9] log(vol)
    np.log(c_aligned / close[idx - 15][valid]),  # [10] 15min ret
    np.log(c_aligned / close[idx - 60][valid]),  # [11] 60min ret
    (stoch_v > 0.85).astype(float),  # [12] Stoch overbought flag
    (stoch_v < 0.15).astype(float),  # [13] Stoch oversold flag
    (pdi_v > mdi_v).astype(float),    # [14] +DI > -DI
]).astype(np.float32)

print(f'  features: {features.shape} ({time.time()-t0:.0f}s)', flush=True)

# === 切分 ===
def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_aligned >= a) & (ts_aligned < b)

tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])
X_tr, y_tr = features[tr_m], labels[tr_m]
X_es, y_es = features[es_m], labels[es_m]
X_te, y_te = features[te_m], labels[te_m]
print(f'  TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}  pos_rate_tr={y_tr.mean():.3f}')

# === 训练 LGB ===
print('\n[2] LightGBM ...', flush=True)
best_res = None
for leaves in [15, 31, 63, 127]:
    for lr in [0.03, 0.05, 0.1]:
        p = dict(
            objective='binary', metric='auc', num_leaves=leaves,
            learning_rate=lr, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=200, lambda_l1=0.01, lambda_l2=0.1,
            verbose=-1, n_jobs=3, seed=42,
        )
        m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X_es, y_es)],
                      callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
        pv_te = m.predict(X_te)
        auc = roc_auc_score(y_te, pv_te)
        # top-k
        line = f'  L={leaves:3d} lr={lr} → auc={auc:.4f}'
        for pct in [0.01, 0.02, 0.05]:
            k = max(1, int(len(pv_te)*pct))
            ti = pv_te.argsort()[-k:]
            acc = y_te[ti].mean()*100
            line += f'  top{pct*100:.0f}%={acc:.1f}%'
        print(line, flush=True)
        
        if best_res is None or auc > best_res[0]:
            best_res = (auc, leaves, lr, m)

print(f'\n[3] 最佳模型特征重要性 ...', flush=True)
_, best_l, best_lr, best_m = best_res
imp = best_m.feature_importance(importance_type='gain')
names = ['stoch_k', 'stoch_inv', 'stoch_mid2', '+DI', '-DI', 'DI_diff', 'ADX', '|DI_diff|',
         'buy_ratio', 'log_vol', 'ret15', 'ret60', 'OB_flag', 'OS_flag', 'DI_bull']
for name, v in sorted(zip(names, imp), key=lambda x: -x[1]):
    print(f'  {name:12s}: {v:.0f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
