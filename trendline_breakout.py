"""基于 Stoch 趋势线识别 + 突破检测 (顺大逆小 + 突破回调).

核心思想 (来自用户手动画线):
1. Stoch 天然有界 [0,1], 钝化后仍回归 -> 适合画趋势线.
2. 趋势线形成需要 >=20 根 K; 连接显著摆动低点(上升线) 或摆动高点(下降线).
3. 突破 = Stoch 上穿下降趋势线 / 下穿上升趋势线.
4. 两个信号阶段:
   a. 顺势阶段: 趋势线已形成, 未突破, 顺趋势.
   b. 突破回调阶段: 突破后回踩趋势线, 确认后继续.
"""
import numpy as np
import pandas as pd


def compute_stoch(close, high, low, k=60, d=1, smooth=1):
    """K-周期 Stochastic. 返回 [0,1] 的有界序列 (NaN->0.5)."""
    ll = pd.Series(low).rolling(k, min_periods=k).min().values
    hh = pd.Series(high).rolling(k, min_periods=k).max().values
    sk = np.nan_to_num((close - ll) / np.maximum(hh - ll, 1e-6), nan=0.5)
    if smooth > 1:
        sk = pd.Series(sk).rolling(smooth, min_periods=1).mean().values
    if d > 1:
        sk = pd.Series(sk).rolling(d, min_periods=1).mean().values
    return sk


def find_pivots(x, order=5):
    """检测局部摆动点 (peak/trough). order=两侧需要多少根bar.
    返回 (pivot_idx, pivot_val, type) 其中 type +1=peak, -1=trough.
    """
    n = len(x)
    peak_idx, trough_idx = [], []
    for i in range(order, n - order):
        win = x[i - order: i + order + 1]
        if x[i] == win.max() and win.max() > win[order - 1] and win.max() > win[order + 1]:
            peak_idx.append(i)
        if x[i] == win.min() and win.min() < win[order - 1] and win.min() < win[order + 1]:
            trough_idx.append(i)
    return peak_idx, trough_idx


def fit_swing_line(swing_idx, x):
    """用摆动点集合拟合趋势线, 返回 (slope, intercept, r2, start_t, end_t).
    slope>0 上升, slope<0 下降.
    """
    if len(swing_idx) < 2:
        return None, None, -np.inf
    ys = x[swing_idx]
    t = np.arange(len(swing_idx), dtype=np.float64)
    slope, intercept = np.polyfit(t, ys, 1)
    yhat = slope * t + intercept
    ss_res = np.sum((ys - yhat) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


def build_trendline_features(x, order=5, min_span=20):
    """为每个时间点 t 计算 Stoch 趋势线特征.
    返回 dict of arrays, 均为 (n,) length.
    """
    n = len(x)
    peak_idx, trough_idx = find_pivots(x, order)

    # 预计算所有摆动点
    # 维护: 到当前 t 为止, 最近的摆动高点序列 / 摆动低点序列
    # 特征:
    #   up_slope: 最近上升趋势线斜率 (用最近>=2个抬升低点)
    #   dn_slope: 最近下降趋势线斜率 (用最近>=2个下降高点)
    #   up_r2, dn_r2
    #   dist_above: Stoch 相对上升趋势线的偏离 (正=在线之上)
    #   dist_below: Stoch 相对下降趋势线的偏离
    #   on_break_up: 是否刚刚上穿下降趋势线
    #   on_break_dn: 是否刚刚下穿上升趋势线
    #   trend_dir: +1 上升趋势, -1 下降趋势, 0 无清晰趋势

    up_slope = np.zeros(n, np.float32)
    dn_slope = np.zeros(n, np.float32)
    up_r2 = np.full(n, -np.inf, np.float32)
    dn_r2 = np.full(n, -np.inf, np.float32)
    trend_dir = np.zeros(n, np.float32)
    dist_above = np.full(n, np.nan, np.float32)  # + 在线之上
    dist_below = np.full(n, np.nan, np.float32)
    on_break_up = np.zeros(n, np.float32)
    on_break_dn = np.zeros(n, np.float32)

    # 扫描推进, 维护每个时间点能看到的摆动点 (只看过去, 无未来函数)
    for t in range(min_span, n):
        # 过去 [0, t] 内所有摆动点
        past_peak = [i for i in peak_idx if i <= t]
        past_trough = [i for i in trough_idx if i <= t]

        # --- 上升趋势线: 用抬升的摆动低点 ---
        if len(past_trough) >= 2:
            last = past_trough[-2:]
            # 至少要跨度 min_span 且低点抬升
            if last[1] - last[0] >= 5:
                s, ic, r2 = fit_swing_line(last, x)
                if s is not None and s > 0 and last[1] - last[0] >= 5:
                    up_slope[t] = s
                    up_r2[t] = r2
                    # 趋势线在 t 的值
                    n_sw = len(last) - 1
                    line_val = x[last[0]] + s * (t - last[0])
                    dist_above[t] = x[t] - line_val  # + 在线之上
                    if dist_above[t] <= 0:
                        trend_dir[t] = 1  # 下降回踩到线, 视为上升趋势的回调
                    on_break_dn[t] = 1.0 if dist_above[t] < 0 else 0.0

        # --- 下降趋势线: 用下降的摆动高点 ---
        if len(past_peak) >= 2:
            last = past_peak[-2:]
            if last[1] - last[0] >= 5:
                s, ic, r2 = fit_swing_line(last, x)
                if s is not None and s < 0 and last[1] - last[0] >= 5:
                    dn_slope[t] = s
                    dn_r2[t] = r2
                    n_sw = len(last) - 1
                    line_val = x[last[0]] + s * (t - last[0])
                    dist_below[t] = x[t] - line_val  # + 在线之上
                    if dist_below[t] >= 0:
                        trend_dir[t] = -1  # 突破到线之上, 视为下降趋势突破
                    on_break_up[t] = 1.0 if dist_below[t] > 0 else 0.0

    # 优先趋势方向: 用断续但明确的 (up_r2 / dn_r2 更好的那类)
    # up_slope 有效则视为多头, dn_slope 有效则视为空头; 取决定性更强的
    return {
        "up_slope": up_slope,
        "dn_slope": dn_slope,
        "up_r2": up_r2,
        "dn_r2": dn_r2,
        "trend_dir": trend_dir,
        "dist_above": dist_above,
        "dist_below": dist_below,
        "on_break_up": on_break_up,
        "on_break_dn": on_break_dn,
    }