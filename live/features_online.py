#!/usr/bin/env python3
"""在线特征引擎: 从"截至 t 及以前的 1min bars"复原训练期 get_X 的 58 列。

关键设计:
- **不出新特征公式**。base/cross 直接复用 features.build_features 与 build_dataset 的
  跨资产特征规则(同一代码路径), 仅限制在所给 bars 窗口内计算; 由 replay_online 抽样
  证伪"窗口截断/只看到 t 及以前"不会改变尾部特征值 = 在线无未来泄漏。
- 输入: 每币一个按 ts 升序的 bars df(open/high/low/close/buy_vol/sell_vol/funding/ts)。
- 输出: 与 live/spec/feature_spec.json get_column_order(symbol)["order"] 对齐的
  numpy [n, 58]; 对单时刻取最后一行即 [1, 58]。

base(38) <- features.build_features; extra(3) <- compute_extra_raw(UTC时段/rvol60);
cross(17) <- build_cross_features 规则(前缀 BTC_/ETH_)。
"""
import numpy as np
import pandas as pd
import polars as pl

import config
from features import build_features
from validate_eth_quick import FEATURES, EXTRA_FEATURE_NAMES, CROSS_FEATURES

CROSS_PREFIX = {"ETH": "BTC_", "BTC": "ETH_"}
CROSS_LR_WINDOWS = (5, 15, 30, 60, 120, 240, 480, 960)
CROSS_Z_WINDOWS = (30, 60, 120, 240, 480)
CROSS_RVOL_WINDOWS = (60, 240)
CROSS_CVD_WINDOWS = (30, 60)


# ---------- 数值工具(O(n) 滚动, 纯 numpy, 无未来泄漏) ----------
def _log_return_from_close(close, k):
    close = np.asarray(close, dtype=np.float64)
    out = np.full(len(close), np.nan)
    if len(close) > k:
        out[k:] = np.log(close[k:] / np.maximum(close[:-k], 1e-12))
    return out


def _rolling_sum(x, w):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n)
    if n >= w:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out[w - 1:] = cs[w:] - cs[: n - w + 1]
    return out


def _zscore(close, w):
    s = np.asarray(close, dtype=np.float64)
    n = len(s)
    mu = np.full(n, np.nan); sd = np.full(n, np.nan)
    if n >= w:
        cs = np.cumsum(s); cs2 = np.cumsum(s * s)
        m = (cs[w - 1:] - np.concatenate([[0.0], cs[: n - w]])) / w
        v = np.maximum((cs2[w - 1:] - np.concatenate([[0.0], cs2[: n - w]])) / w - m * m, 0.0)
        mu[w - 1:] = m; sd[w - 1:] = np.sqrt(v)
    return np.where(sd > 1e-9, (s - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)


def _l1(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close); out = np.zeros(n)
    out[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
    return out


def _roll_std_sample(lr1, w):
    n = len(lr1); out = np.zeros(n)
    if n >= w:
        cs = np.cumsum(lr1); cs2 = np.cumsum(lr1 * lr1)
        m = (cs[w - 1:] - np.concatenate([[0.0], cs[: n - w]])) / w
        v = np.maximum((cs2[w - 1:] - np.concatenate([[0.0], cs2[: n - w]])) / w - m * m, 0.0)
        out[w - 1:] = np.sqrt(v * w / (w - 1)) * 100.0
    return out


# ---------- 跨资产特征(对手盘, 17 列) ----------
def _cross_z_log(close, w):
    """复刻 build_cross_features 的 z_w: 对 log(close) 因果滚窗均值/样本std 标准化。

    离线 ds 的 z 由 pandas rolling(w).std() 生成(其底层为在线两趟平均/样本标准差, 数值精确,
    无 E[x^2]-E[x]^2 灾难消去; 实测对全量窗口逐位一致)。在线用滑动窗口 np.std(ddof=1)
    (np 内部两趟居中), 与 pandas 逐窗口结果在 float64 下逐位一致, 仅少数近常数窗口
    (sd→1e-9 阈值附近)有 ~1e-4 以下舍入差异, 且与离线差同量级, 不构成未来泄漏/公式偏差。
    """
    from numpy.lib.stride_tricks import sliding_window_view
    lc = np.log(np.maximum(np.asarray(close, dtype=np.float64), 1e-12))
    n = len(lc)
    if n < w:
        return np.full(n, 0.0)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    s = pd.Series(lc)
    mu = s.rolling(w).mean().to_numpy()
    W = sliding_window_view(lc, w)
    ss = np.std(W, axis=1, ddof=1)
    sd[w - 1:] = ss
    return np.where(sd > 1e-9, (lc - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)


def cross_features(other_df):
    ts = other_df["ts"].to_numpy().astype(np.int64)
    close = other_df["close"].to_numpy().astype(np.float64)
    buy = other_df["buy_vol"].to_numpy().astype(np.float64)
    sell = other_df["sell_vol"].to_numpy().astype(np.float64)
    n = len(close)
    cols = {}
    for k in CROSS_LR_WINDOWS:
        cols[f"lr_{k}"] = _log_return_from_close(close, k)
    for w in CROSS_Z_WINDOWS:
        cols[f"z_{w}"] = _cross_z_log(close, w)
    lr1 = _l1(close)
    for w in CROSS_RVOL_WINDOWS:
        cols[f"rvol_{w}"] = _roll_std_sample(lr1, w)
    tt = buy + sell
    d = buy - sell
    for w in CROSS_CVD_WINDOWS:
        cols[f"cvd_{w}"] = _rolling_sum(d, w) / np.maximum(_rolling_sum(tt, w), 1e-12)
    return cols


# ---------- extra(3 列, 复刻 compute_extra_raw) ----------
def _extra_features(df):
    ts = df["ts"].to_numpy().astype(np.int64)
    close = df["close"].to_numpy().astype(np.float64)
    rvol_60 = _roll_std_sample(_l1(close), 60).astype(np.float32)
    hour = ((ts % 86400) // 3600).astype(np.float64)
    hour_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    hour_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    minute_of_day = (ts % 86400) // 60
    shift = np.where(hour < 8, 0.0, np.where(hour < 13, -8 * 60, np.where(hour < 21, -13 * 60, -21 * 60)))
    session_minutes = (minute_of_day + shift).astype(np.float32)
    return {
        "hour_sin_rvol_60": (hour_sin * rvol_60).astype(np.float32),
        "session_minutes": (session_minutes / 480.0).astype(np.float32),
        "hour_sin_hour_cos": (hour_sin * hour_cos).astype(np.float32),
    }


def bars_frame(ts, o, h, l, c, buy, sell, funding=None):
    """组装 polars bars df(ts 秒升序)。"""
    return pl.DataFrame({
        "ts": ts.astype(np.int64),
        "open": o.astype(np.float32), "high": h.astype(np.float32),
        "low": l.astype(np.float32), "close": c.astype(np.float32),
        "buy_vol": buy.astype(np.float32), "sell_vol": sell.astype(np.float32),
        "funding": (funding if funding is not None else np.zeros(len(ts))).astype(np.float32),
    })


def compute_X(self_df, cross_df, symbol):
    """两币 bars df -> 目标 symbol 的 [n, 58], 列序同 feature_spec order。

    对齐: cross 特征按"与 self 每行相同的时刻"取对手盘值(searchsorted), 保证在线/离线
    语义一致且无未来泄漏(对手盘 bar 取 <= self.ts 的那根)。self/cross 长度可不同。
    """
    feat_df = build_features(self_df)
    names = list(feat_df.columns)
    bmap = {c: i for i, c in enumerate(names)}
    base_df = feat_df.to_numpy()
    base = np.full((len(self_df), len(FEATURES)), np.nan, dtype=np.float32)
    for j, c in enumerate(FEATURES):
        if c in bmap:
            base[:, j] = base_df[:, bmap[c]]
    extra = _extra_features(self_df)
    ex = np.column_stack([extra[c] for c in EXTRA_FEATURE_NAMES])
    # 对手盘特征按时刻对齐
    other_ts = cross_df["ts"].to_numpy().astype(np.int64)
    own_ts = self_df["ts"].to_numpy().astype(np.int64)
    idx = np.searchsorted(other_ts, own_ts, side="right") - 1
    idx = np.clip(idx, 0, len(cross_df) - 1)
    cr = cross_features(cross_df)
    cr_stack = np.column_stack([np.asarray(cr[c], dtype=np.float32) for c in CROSS_FEATURES])
    cx = cr_stack[idx]
    return np.concatenate([base, ex, cx], axis=1).astype(np.float32)