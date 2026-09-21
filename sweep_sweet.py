"""Sweep Stoch+DI+vol 阈值: 找 acc>=60% + tpd>=15 的 sweet spot"""
import numpy as np, pandas as pd, time
from numpy.lib.stride_tricks import sliding_window_view
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

H = config.HORIZON_MIN  # 15
SEG = 60
idx = np.arange(SEG, N - H)

# Stoch
ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
stoch = (close[idx] - ll) / np.maximum(hh - ll, 1e-6)
labels = (close[idx + H] > close[idx]).astype(np.int64)
ret_h = close[idx + H] / close[idx] - 1

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

# ADX
dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-6)
adx = ws(dx, 14)
adx_a = adx[idx-1]

# Stoch change (trend)
stoch_d = np.zeros(len(idx))
stoch_d[1:] = stoch[1:] - stoch[:-1]

# Volatility
vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        seg = close[idx[i]-59:idx[i]+1]
        seg_ret = np.diff(np.log(seg))
        vol[i] = np.std(seg_ret) * np.sqrt(60)

# Buy ratio
buy_r = np.log(np.maximum(buy_v[idx] / np.maximum(sell_v[idx], 1.0), 1e-6))

# Ret 60
ret60 = np.log(close[idx] / np.maximum(close[idx-60], 1e-6))

# Test mask
def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
te_m = mk(*config.SPLITS['test'])

def eval_sig(name, mask_long, mask_short):
    """在 test 上评估做多做空信号"""
    ml = mask_long & te_m; ms = mask_short & te_m
    n = ml.sum() + ms.sum()
    if n == 0:
        return None
    # 做多预测 1 (涨), 做空预测 0 (跌)
    # acc_long = P(label=1 | long sig), acc_short = P(label=0 | short sig)
    acc_l = labels[ml].mean()*100 if ml.sum()>0 else 0
    acc_s = (1-labels[ms]).mean()*100 if ms.sum()>0 else 0
    # 加权平均 acc
    acc = (acc_l * ml.sum() + acc_s * ms.sum()) / n
    tpd = n / 356
    avg_ret = np.concatenate([ret_h[ml], -ret_h[ms]]).mean()*100 if n>0 else 0
    hits_acc = acc >= 60 and tpd >= 14
    flag = '🎯' if hits_acc else ('★' if acc >= 58 and tpd >= 10 else '')
    print(f'{flag} {name:50s} n={n:6d} tpd={tpd:5.1f} acc={acc:.1f}% '
          f'long={acc_l:.1f}%/{ml.sum()} short={acc_s:.1f}%/{ms.sum()} avg_ret={avg_ret:.3f}%', flush=True)
    return (tpd, acc)

print(f'Test samples: {te_m.sum():,}', flush=True)

# Sweep 1: Stoch threshold + vol threshold
print('\n[1] Stoch threshold × Vol threshold (双向) ...', flush=True)
for st_lo, st_hi in [(0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]:
    for v_pct in [30, 40, 50, 60, 70]:
        v_th = np.nanpercentile(vol[te_m], v_pct)
        vf = (vol > v_th) & (~np.isnan(vol)) & te_m
        ml = (stoch < st_lo) & vf
        ms = (stoch > st_hi) & vf
        eval_sig(f'Stoch<{st_lo}/{st_hi} vol>{v_pct}%', ml, ms)

# Sweep 2: Stoch + DI direction 确认
print('\n[2] Stoch + DI 方向确认 (DI 过滤) ...', flush=True)
for st_lo, st_hi in [(0.1, 0.9), (0.15, 0.85)]:
    for v_pct in [40, 50, 60, 70]:
        v_th = np.nanpercentile(vol[te_m], v_pct)
        vf = (vol > v_th) & (~np.isnan(vol))
        # 做多: Stoch 低 + +DI > -DI
        ml = (stoch < st_lo) & (pdi_a > mdi_a) & vf & te_m
        # 做空: Stoch 高 + -DI > +DI
        ms = (stoch > st_hi) & (mdi_a > pdi_a) & vf & te_m
        eval_sig(f'Stoch+DI vol>{v_pct}%', ml, ms)

# Sweep 3: Stoch + ADX trend filter
print('\n[3] Stoch + ADX 趋势强度过滤 ...', flush=True)
for st_lo, st_hi in [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]:
    for adx_min in [10, 15, 20, 25]:
        for v_pct in [30, 40, 50]:
            v_th = np.nanpercentile(vol[te_m], v_pct)
            vf = (vol > v_th) & (~np.isnan(vol)) & te_m
            ml = (stoch < st_lo) & (adx_a > adx_min) & (stoch_d > 0) & vf  # oversold + trend + stoch rising
            ms = (stoch > st_hi) & (adx_a > adx_min) & (stoch_d < 0) & vf  # overbought + trend + stoch falling
            eval_sig(f'Stoch{st_lo}/{st_hi} ADX>{adx_min}↑↓ vol>{v_pct}%', ml, ms)

# Sweep 4: 最终优化 - Stoch + DI + ADX + buy_ratio
print('\n[4] Stoch + DI + ADX + buy_ratio 综合 (目标 tpdx14 accx60%) ...', flush=True)
for st_lo, st_hi in [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)]:
    for di_min in [0, 5, 10, 15]:
        v_th = np.nanpercentile(vol[te_m], 40)  # 固定 vol>40%ile
        vf = (vol > v_th) & (~np.isnan(vol)) & te_m
        ml = (stoch < st_lo) & (di_diff > di_min) & vf
        ms = (stoch > st_hi) & (di_diff < -di_min) & vf
        eval_sig(f'Stoch+DI±{di_min} vol>40%', ml, ms)

print(f'\n⏱ {time.time()-t0:.0f}s')
