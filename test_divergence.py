"""测试背离形态 — 价格 vs Stoch / DMI."""
import numpy as np, pandas as pd, time
from scipy.signal import find_peaks
import config

t0 = time.time()
print('[1] 加载', flush=True)
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close = eth['close'].values.astype(np.float64)
high = eth['high'].values.astype(np.float64)
low  = eth['low'].values.astype(np.float64)
ts = eth['ts'].values.astype(np.int64)
N = len(close); H = config.HORIZON_MIN

ll = pd.Series(low).rolling(60, min_periods=60).min().values
hh = pd.Series(high).rolling(60, min_periods=60).max().values
stoch = (close - ll) / np.maximum(hh - ll, 1e-6)
stoch = np.nan_to_num(stoch, nan=0.5)
del eth

def find_local_peaks(seq, window=30):
    """滑动窗口找局部峰/谷位置。返回峰值数组 (0/1) 和值数组."""
    peaks = np.zeros(len(seq), np.float64)
    troughs = np.zeros(len(seq), np.float64)
    # 峰: 比前后各 window/2 个 bar 都大
    half = window // 2
    for i in range(half, len(seq) - half):
        seg = seq[i-half:i+half+1]
        if seq[i] == seg.max(): peaks[i] = seq[i]
        if seq[i] == seg.min(): troughs[i] = seq[i]
    return peaks, troughs

def divergence_score(close, stoch, window=60):
    """对每个 bar, 在过去 window 内找最近两个价格峰 vs Stoch 峰.
    顶背离 = 价格峰新高但 Stoch 峰没新高 → -1 (看跌)
    底背离 = 价格峰新低但 Stoch 峰没新低 → +1 (看涨)
    返回 score ∈ [-1, 1]"""
    N = len(close)
    scores = np.zeros(N, np.float32)

    for t in range(window, N):
        c_seg = close[t-window:t]
        s_seg = stoch[t-window:t]

        c_peaks, c_troughs = find_local_peaks(c_seg, 15)
        s_peaks, s_troughs = find_local_peaks(s_seg, 15)

        # 找最近两个价格峰
        c_pk_pos = np.where(c_peaks > 0)[0]
        s_pk_pos = np.where(s_peaks > 0)[0]
        c_tr_pos = np.where(c_troughs > 0)[0]
        s_tr_pos = np.where(s_troughs > 0)[0]

        score = 0.0
        if len(c_pk_pos) >= 2 and len(s_pk_pos) >= 2:
            c_now = c_seg[c_pk_pos[-1]]; c_prev = c_seg[c_pk_pos[-2]]
            s_now = s_seg[s_pk_pos[-1]]; s_prev = s_seg[s_pk_pos[-2]]
            if c_now > c_prev and s_now < s_prev:   # 顶背离
                score -= 0.7
            elif c_now < c_prev and s_now > s_prev: # 双升 → 趋势强
                score += 0.3

        if len(c_tr_pos) >= 2 and len(s_tr_pos) >= 2:
            c_now = c_seg[c_tr_pos[-1]]; c_prev = c_seg[c_tr_pos[-2]]
            s_now = s_seg[s_tr_pos[-1]]; s_prev = s_seg[s_tr_pos[-2]]
            if c_now < c_prev and s_now > s_prev:   # 底背离
                score += 0.7
            elif c_now > c_prev and s_now < s_prev: # 双跌 → 趋势弱
                score -= 0.3

        # Stoch 位置 + slope
        score += (s_seg[-1] - 0.5) * 0.3  # 超卖区正, 超买区负
        score += np.sign(s_seg[-1] - s_seg[-5]) * 0.2  # 最近方向

        scores[t] = score

        if t % 500_000 == 0:
            print(f'  ... t={t:,} ({time.time()-t0:.0f}s)', flush=True)

    return scores

# 先只算 train+es+test 区域 (省时间)
a_tr = int(pd.Timestamp('2020-01-01', tz='UTC').timestamp())
b_te = int(pd.Timestamp(config.SPLITS['test'][1], tz='UTC').timestamp())
mask = (ts >= a_tr) & (ts < b_te)
idx = np.where(mask)[0]
print(f'[2] 只算 train+es+test {len(idx):,} bar ({time.time()-t0:.1f}s)', flush=True)

# 只在需要的位置算
def divergence_score_sparse(close, stoch, target_idx, window=60):
    scores = np.zeros(len(close), np.float32)
    half = 7
    for ti, t in enumerate(target_idx):
        c_seg = close[t-window:t].astype(np.float64)
        s_seg = stoch[t-window:t].astype(np.float64)

        # 快速找峰谷: 比前后 half 都大/小
        c_pk = []; c_tr = []; s_pk = []; s_tr = []
        for i in range(half, len(c_seg)-half):
            seg_c = c_seg[i-half:i+half+1]
            seg_s = s_seg[i-half:i+half+1]
            if c_seg[i] == seg_c.max(): c_pk.append((i, c_seg[i]))
            if c_seg[i] == seg_c.min(): c_tr.append((i, c_seg[i]))
            if s_seg[i] == seg_s.max(): s_pk.append((i, s_seg[i]))
            if s_seg[i] == seg_s.min(): s_tr.append((i, s_seg[i]))

        score = 0.0
        if len(c_pk) >= 2 and len(s_pk) >= 2:
            if c_pk[-1][1] > c_pk[-2][1] and s_pk[-1][1] < s_pk[-2][1]: score -= 0.8
            elif c_pk[-1][1] < c_pk[-2][1] and s_pk[-1][1] > s_pk[-2][1]: score += 0.3
        if len(c_tr) >= 2 and len(s_tr) >= 2:
            if c_tr[-1][1] < c_tr[-2][1] and s_tr[-1][1] > s_tr[-2][1]: score += 0.8
            elif c_tr[-1][1] > c_tr[-2][1] and s_tr[-1][1] < s_tr[-2][1]: score -= 0.3

        score += (s_seg[-1] - 0.5) * 0.3
        score += np.sign(s_seg[-1] - s_seg[-5]) * 0.2
        scores[t] = score

        if ti % 200_000 == 0:
            print(f'  ... {ti/len(target_idx)*100:.0f}% ({time.time()-t0:.0f}s)', flush=True)

    return scores

# 先抽 10 万样本验证这个方法是不是对的
sample_idx = idx[::max(1, len(idx)//100_000)]
print(f'\n[3] 快速验证 ({len(sample_idx):,} sample)', flush=True)
scores_s = divergence_score_sparse(close, stoch, sample_idx, window=60)

# 硬规则 threshold 扫描
print(f'\n[4] 阈值扫描 (sample only):')
for thr in [0.3, 0.5, 0.7, 0.9]:
    for direction in [1, -1]:  # 1=看涨阈值, -1=看跌阈值
        if direction == 1:
            sig = sample_idx[scores_s[sample_idx] > thr]
        else:
            sig = sample_idx[scores_s[sample_idx] < -thr]
        if len(sig) == 0: continue
        y = (close[sig + H] > close[sig]).astype(int)
        acc = y.mean() * 100
        tpd = len(sig) / 365
        tag = 'long' if direction == 1 else 'short'
        print(f'  thr={thr} {tag}: {len(sig)} sig, acc={acc:.1f}%, tpd={tpd:.1f}')

# 如果有 promising 阈值, 跑全量
print(f'\n[5] 全量计算', flush=True)
scores = divergence_score_sparse(close, stoch, idx, window=60)

# Test set 评估
def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    m = (idx >= a) & (idx < b)
    return idx[m], scores[idx[m]]

te_idx, te_scores = mk(*config.SPLITS['test'])
y_te = (close[te_idx + H] > close[te_idx]).astype(int)

# AUC-style
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_te, te_scores)
print(f'\n  Test AUC = {auc:.4f}')

# top-k
for pct in [0.005, 0.01, 0.02, 0.05]:
    k = max(1, int(len(te_idx) * pct))
    # 多头: score 最高的
    top_long = np.argsort(te_scores)[-k:]
    acc_l = y_te[top_long].mean() * 100
    # 空头: score 最低的
    top_short = np.argsort(te_scores)[:k]
    acc_s = (1 - y_te[top_short]).mean() * 100
    print(f'  top {pct*100:.1f}%: long={acc_l:.1f}% short={acc_s:.1f}%')

print(f'\n⏱ {time.time()-t0:.0f}s')
