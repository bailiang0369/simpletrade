"""诊断2: 小时段 + 波动过滤 + Stoch/DMI 信号 → 找出用户手动交易场景"""
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

H = 15; SEG = 60

# Stoch
idx = np.arange(SEG, N - H)
ll = np.min(sliding_window_view(low, SEG)[idx - SEG], axis=1)
hh = np.max(sliding_window_view(high, SEG)[idx - SEG], axis=1)
c_now = close[idx]
stoch = (c_now - ll) / np.maximum(hh - ll, 1e-6)
labels = (close[idx + H] > c_now).astype(np.int64)

# ret over H
ret_h = close[idx + H] / c_now - 1

# 小时 + 星期
dt = pd.to_datetime(ts[idx], unit='s', utc=True)
hour = dt.hour.values
dow = dt.dayofweek.values

# 过去 60 bar 波动率
ret_1m = np.log(close[idx] / close[idx-1])  # ~idx-1
# 用 rolling std 做 volatility proxy
vol = np.zeros(len(idx))
for i in range(len(idx)):
    if idx[i] > 60:
        seg = close[idx[i]-59:idx[i]+1]
        seg_ret = np.diff(np.log(seg))
        vol[i] = np.std(seg_ret) * np.sqrt(60)  # annualized-ish
    else:
        vol[i] = np.nan

# Test mask
def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts[idx] >= a) & (ts[idx] < b)
te_m = mk(*config.SPLITS['test'])
print(f'Test: {te_m.sum():,} samples', flush=True)

# === 1. 小时段 × Stoch 阈值 ===
print('\n[1] 小时段 × Stoch oversold/overbought (H=15) ...', flush=True)
stoch_th_lo, stoch_th_hi = 0.15, 0.85
for h_start, h_end in [(0,6), (6,12), (12,18), (18,24), (0,24)]:
    hr = (hour >= h_start) & (hour < h_end)
    sig_long = (stoch < stoch_th_lo) & hr & te_m
    sig_short = (stoch > stoch_th_hi) & hr & te_m
    n = sig_long.sum() + sig_short.sum()
    if n == 0: continue
    acc = (labels[sig_long].mean() * sig_long.sum() + (1-labels[sig_short]).mean() * sig_short.sum()) / n * 100
    tpd = n / 356
    print(f'  hours [{h_start:2d}-{h_end:2d}): n={n:6d} tpd≈{tpd:5.1f} acc={acc:.1f}% '
          f'(long={labels[sig_long].mean()*100:.1f}% short={(1-labels[sig_short]).mean()*100:.1f}%)', flush=True)

# === 2. 波动过滤 + Stoch 阈值 ===
print('\n[2] 波动过滤 + Stoch(H=15) ...', flush=True)
for v_th in [np.nanpercentile(vol[te_m], 50),
             np.nanpercentile(vol[te_m], 60),
             np.nanpercentile(vol[te_m], 70),
             np.nanpercentile(vol[te_m], 80)]:
    vf = (vol > v_th) & te_m & (~np.isnan(vol))
    sl = (stoch < 0.15) & vf; ss = (stoch > 0.85) & vf
    n = sl.sum() + ss.sum()
    if n == 0: continue
    acc = (labels[sl].mean()*sl.sum() + (1-labels[ss]).mean()*ss.sum()) / n * 100
    tpd = n / 356
    avg_ret = np.concatenate([ret_h[sl], -ret_h[ss]]).mean()*100
    print(f'  vol>{v_th:.4f}: n={n:6d} tpd≈{tpd:5.1f} acc={acc:.1f}% avg_ret={avg_ret:.3f}%', flush=True)

# === 3. 同时看 Stoch+DMI → 更精确信号 ===
print('\n[3] Stoch 上穿/下穿 + DMI 确认 (金叉/死叉 逻辑) ...', flush=True)
# DMI (已经在 eth array 算, 对齐 idx-1)
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
dmi_i=idx-1
pdi_a=pdi[dmi_i];mdi_a=mdi[dmi_i]

# Stoch K (60) 和 K(60) - K(60)[-1] 的 change 判定趋势
# 金叉: Stoch 从下往上穿过 0.15 且 +DI 刚上穿 -DI
# 死叉: Stoch 从上往下穿过 0.85 且 -DI 刚上穿 +DI
# 简化: 用 (stoch < 0.15) & (pdi_a > mdi_a) 并确保 Stoch 在上升趋势
stoch_diff = np.zeros(len(idx))
for i in range(1, len(idx)):
    if idx[i] < N-5:
        stoch_diff[i] = stoch[i] - stoch[i-1]

for v_th in [np.nanpercentile(vol[te_m], 50), np.nanpercentile(vol[te_m], 70), np.nanpercentile(vol[te_m], 80)]:
    vf = (vol > v_th) & (~np.isnan(vol)) & te_m
    sl = (stoch < 0.15) & (pdi_a > mdi_a) & (stoch_diff > -0.01) & vf  # oversold + DI bull + stoch not crashing
    ss = (stoch > 0.85) & (mdi_a > pdi_a) & (stoch_diff < 0.01) & vf  # overbought + DI bear
    n = sl.sum() + ss.sum()
    if n < 10: continue
    acc = (labels[sl].mean()*sl.sum() + (1-labels[ss]).mean()*ss.sum()) / n * 100
    tpd = n / 356
    avg_ret = np.concatenate([ret_h[sl], -ret_h[ss]]).mean()*100
    print(f'  vol>{v_th:.4f} + Stoch+DI+dyn: n={n:6d} tpd≈{tpd:5.1f} acc={acc:.1f}% avg_ret={avg_ret:.3f}%', flush=True)

# === 4. Stoch 金叉死叉 (SMA 金叉, smooth=3 默认) ===
print('\n[4] 更精确: Stoch 从下往上穿过 阈值 (金叉事件) ...', flush=True)
# 检查 idx 位置的 Stoch 和 idx-1 位置的
for v_th in [np.nanpercentile(vol[te_m], 50), np.nanpercentile(vol[te_m], 70)]:
    vf = (vol > v_th) & (~np.isnan(vol)) & te_m
    # 金叉: stoch[i-1] < 0.15 且 stoch[i] >= 0.15 (刚刚穿上来)
    # 需要 idx[i-1] = idx[i]-1
    prev_stoch = stoch[np.where(idx-1 >= SEG)[0]]  # 不对, idx 本身就是时间点
    # idx[i] 对应时间点 t, stoch[i] 用的是 low/high in [t-60, t]
    # idx[i-1] 对应时间点 t-1, stoch[i-1] 用的是 [t-61, t-1]
    golden = np.zeros(len(idx), dtype=bool)
    death = np.zeros(len(idx), dtype=bool)
    for i in range(1, min(len(idx), len(stoch))):
        golden[i] = (stoch[i-1] < 0.15) & (stoch[i] >= 0.15)
        death[i] = (stoch[i-1] > 0.85) & (stoch[i] <= 0.85)
    sl = golden & vf & (pdi_a > mdi_a)
    ss = death & vf & (mdi_a > pdi_a)
    n = sl.sum() + ss.sum()
    if n < 10: continue
    acc = (labels[sl].mean()*sl.sum() + (1-labels[ss]).mean()*ss.sum()) / n * 100
    tpd = n / 356
    avg_ret = np.concatenate([ret_h[sl], -ret_h[ss]]).mean()*100
    print(f'  vol>{v_th:.4f}: golden={sl.sum()} death={ss.sum()} n={n} tpd≈{tpd:.1f} acc={acc:.1f}% avg_ret={avg_ret:.3f}%', flush=True)

print(f'\n⏱ {time.time()-t0:.0f}s')
