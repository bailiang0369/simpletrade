"""快速验证 Stoch 趋势线 + 突破信号在 test 集上的有效性."""
import numpy as np
import pandas as pd
import time, config
from sklearn.metrics import roc_auc_score
from trendline_breakout import compute_stoch, build_trendline_features

t0 = time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close = eth['close'].values.astype(np.float64)
high = eth['high'].values.astype(np.float64)
low = eth['low'].values.astype(np.float64)
ts = eth['ts'].values.astype(np.int64)
N = len(close); H = config.HORIZON_MIN
del eth

# Stoch(60,1,1) 在线
stoch = compute_stoch(close, high, low, k=60, d=1, smooth=1)
print(f'[1] Stoch computed ({time.time()-t0:.0f}s)', flush=True)

# 只跑 test 集区间做趋势线特征 (避免全量太慢)
te_a = int(pd.Timestamp(config.SPLITS['test'][0], tz='UTC').timestamp())
te_b = int(pd.Timestamp(config.SPLITS['test'][1], tz='UTC').timestamp())
# 取 test 起始往前回溯足够长(如 2000 bar)算趋势线, 然后只评估 test 部分
lookback = 3000
te_start_idx = np.searchsorted(ts, te_a)
window_start = max(0, te_start_idx - lookback)
sub_stoch = stoch[window_start: te_start_idx + 479520]

print(f'[2] computing trendlines on sub ({len(sub_stoch):,})', flush=True)
feats = build_trendline_features(sub_stoch, order=8, min_span=20)
print(f'[3] done ({time.time()-t0:.0f}s)', flush=True)

# 映射回来: 评估只在 test 集
eval_slice = slice(lookback, None)
te_idx = np.arange(window_start, te_start_idx + 479520)[eval_slice]
y_te = (close[te_idx + H] > close[te_idx]).astype(int)
te_mask_ts = (ts[te_idx] >= te_a)

y_te = y_te[te_mask_ts]
for k in feats:
    v = feats[k][eval_slice]
    feats[k] = v[te_mask_ts]

print(f'[4] test eval pts: {len(y_te):,}  pos={y_te.mean():.3f}', flush=True)

# --- 规则信号 ---
# 多头信号: 上升趋势线(up_slope>0, r2 高) + Stoch 回踩在趋势线上方后回升
# 空头信号: 下降趋势线(dn_slope<0) + 反弹后再下
trend = feats['trend_dir']
up_slope = feats['up_slope']; dn_slope = feats['dn_slope']
up_r2 = np.nan_to_num(feats['up_r2'], nan=-1)
dn_r2 = np.nan_to_num(feats['dn_r2'], nan=-1)
dist_above = feats['dist_above']; dist_below = feats['dist_below']
brk_up = feats['on_break_up']; brk_dn = feats['on_break_dn']

# 规则打分
score = np.zeros(len(y_te), np.float32)
# 多头: 上升趋势线向上 + 回踩(在下/贴线) -> 做多
long_ok = (up_slope > 0) & (up_r2 > 0.6)
score[long_ok] += up_slope[long_ok] * 2
# 刚上穿下降趋势线 (突破) -> 做多
score[brk_up > 0] += 1.0
# 空头
short_ok = (dn_slope < 0) & (dn_r2 > 0.6)
score[short_ok] -= dn_slope[short_ok] * 2  # dn_slope<0, -dn>0
score[brk_dn > 0] -= 1.0

print(f'\n====== Stoch 趋势线规则信号 (test) ======')
auc = roc_auc_score(y_te, score)
print(f'AUC = {auc:.4f}  (历史最佳 LGBM 87feats = 0.544)')
for pct in [0.005, 0.01, 0.02, 0.05, 0.1]:
    k = max(1, int(len(y_te) * pct))
    si = np.argsort(score)
    aL = y_te[si[-k:]].mean() * 100
    aS = (1 - y_te[si[:k]]).mean() * 100
    print(f'  top{pct*100:>5.1f}%: long={aL:>5.1f}% short={aS:>5.1f}%')

# 只看多头信号样本的质量
sig = np.abs(score) > 0
print(f'\n有信号样本: {sig.sum():,} ({(sig.mean()*100):.2f}%)')
if sig.sum() > 100:
    sub = score[sig]; suby = y_te[sig]
    aL = suby[sub > 0].mean()*100 if (sub > 0).sum() else 0
    aS = (1 - suby[sub < 0]).mean()*100 if (sub < 0).sum() else 0
    print(f'  多头 acc={aL:.1f}% ({ (sub>0).sum():,})  空头 acc={aS:.1f}% ({(sub<0).sum():,})')

print(f'\n⏱ {time.time()-t0:.0f}s')