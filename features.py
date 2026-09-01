"""特征工程 (polars 版, float32 输出)。

严格遵守用户约束:
- 仅使用价格衍生特征 (O/H/L/C 及其滚动统计) 与主动买卖成交量 (buy_vol/sell_vol 及其滚动统计)
- 禁用: 总成交量 volume、quote_asset_volume、ATR、number_of_trades
- 特征均只使用 t 时刻及之前的收盘信息 (滚动/滞后), 无未来函数泄漏
- 滚动窗口不足处产生 null, 由 build_dataset 统一剔除; 不会向数据集泄漏 NaN
"""
import numpy as np
import polars as pl

EPS = 1e-12


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """df: polars DataFrame, 列含 ts(int64秒)/open/high/low/close/buy_vol(主动买)/sell_vol(主动卖)。
    返回与 df 等长的特征 DataFrame(float32), 与旧版 pandas 特征列完全一致。"""
    C = pl.col("close")
    O = pl.col("open")
    H = pl.col("high")
    L = pl.col("low")
    TB = pl.col("buy_vol")          # 主动买量
    TS = pl.col("sell_vol")         # 主动卖量
    DT = pl.from_epoch(pl.col("ts"), time_unit="s")

    lr = (C / C.shift(1)).log()

    e = {}
    # ---- 收益率 / 动量 ----
    for w in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
        e[f"lr_{w}"] = (C / C.shift(w)).log()
    for w in [5, 10, 15, 30, 60]:
        e[f"mom_{w}"] = (C / C.shift(w) - 1) * 100

    # ---- 已实现波动率 (价格衍生, 非ATR) ----
    for w in [5, 10, 15, 30, 60, 120]:
        e[f"rvol_{w}"] = lr.rolling_std(w, ddof=1) * 100
    e["rvol_ratio_60_5"] = e["rvol_60"] / (e["rvol_5"] + EPS)
    e["rvol_ratio_120_15"] = e["rvol_120"] / (e["rvol_15"] + EPS)

    # ---- 趋势 / Z分数 ----
    for w in [10, 30, 60, 120]:
        mu = C.rolling_mean(w)
        sd = C.rolling_std(w, ddof=1)
        e[f"z_{w}"] = (C - mu) / (sd + EPS)
    for w in [10, 30, 60]:
        e[f"ema_dev_{w}"] = (C - C.ewm_mean(span=w, min_samples=w // 2)) / C * 100

    # ---- 区间位置 / 支撑压力 ----
    for w in [10, 30, 60, 120, 240]:
        lo = L.rolling_min(w)
        hi = H.rolling_max(w)
        e[f"pos_{w}"] = (C - lo) / (hi - lo + EPS)
        e[f"dd_{w}"] = (C / hi - 1) * 100
        e[f"ru_{w}"] = (C / lo - 1) * 100
    # 突破前高/前低
    e["brk_hi_60"] = (C > H.rolling_max(60).shift(1)).cast(pl.Float32)
    e["brk_lo_60"] = (C < L.rolling_min(60).shift(1)).cast(pl.Float32)

    # ---- K线形态 ----
    rng = (H - L) + EPS
    body = C - O
    e["body_ratio"] = body / rng
    e["up_wick"] = (H - pl.max_horizontal(C, O)) / rng
    e["lo_wick"] = (pl.min_horizontal(C, O) - L) / rng
    e["is_green"] = (C > O).cast(pl.Float32)
    for w in [3, 5, 10]:
        e[f"ngreen_{w}"] = (C > O).cast(pl.Float32).rolling_mean(w)

    # ---- 缺口 ----
    e["gap"] = (O - C.shift(1)) / (C.shift(1) + EPS) * 100

    # ---- 随机指标 (Stochastic K) 长周期 ----
    for k in [29, 50, 69]:
        lo_k = L.rolling_min(k)
        hi_k = H.rolling_max(k)
        e[f"stoch_k_{k}"] = (C - lo_k) / (hi_k - lo_k + EPS)

    # ---- DMI 方向动量 (DI+ / DI-) ----
    # 真波幅 TR
    tr = pl.max_horizontal(
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs(),
    ) + EPS
    # 方向运动
    h_up = H - H.shift(1)
    l_dn = L.shift(1) - L
    pos_dm = pl.when((h_up > 0) & (h_up > l_dn)).then(h_up).otherwise(0.0)
    neg_dm = pl.when((l_dn > 0) & (l_dn > h_up)).then(l_dn).otherwise(0.0)
    # DMI 标准周期 14
    dm_period = 14
    tr_smooth = tr.rolling_sum(dm_period)
    pos_dm_smooth = pos_dm.rolling_sum(dm_period)
    neg_dm_smooth = neg_dm.rolling_sum(dm_period)
    e["di_plus"] = pos_dm_smooth / tr_smooth
    e["di_minus"] = neg_dm_smooth / tr_smooth

    # ---- 主动买卖量特征 (仅主动买/主动卖) ----
    tot = TB + TS + EPS
    tbr = TB / tot                                  # 主动买占比 [0,1]
    e["tbr"] = tbr
    e["cvd_delta"] = (TB - TS) / tot          # 当根主动净买
    for w in [5, 10, 15, 30, 60, 120]:
        bt = TB.rolling_sum(w)
        st = TS.rolling_sum(w)
        e[f"tbr_{w}"] = bt / (bt + st + EPS)
        e[f"cvd_{w}"] = (bt - st) / (bt + st + EPS)   # 主动净买占比(滚动)
    for w in [10, 30]:
        e[f"buyvol_strength_{w}"] = TB.rolling_mean(w) / (TS.rolling_mean(w) + EPS) - 1
    for w in [15, 60]:
        e[f"tb_act_{w}"] = TB.rolling_std(w, ddof=1) / (TB.rolling_mean(w) + EPS)
        e[f"ts_act_{w}"] = TS.rolling_std(w, ddof=1) / (TS.rolling_mean(w) + EPS)
    # cvd 趋势(斜率)
    e["cvd_slope_30"] = (TB - TS).rolling_sum(30).diff(5) / (
        (TB + TS).rolling_sum(30).diff(5) + EPS)

    # ---- 日内/时间特征 (UTC) ----
    hour = DT.dt.hour()
    minute = DT.dt.minute()
    dow = DT.dt.weekday() - 1                    # 与 pandas dayofweek 对齐: 周一=0
    e["hour_sin"] = (hour * 2 * np.pi / 24).sin()
    e["hour_cos"] = (hour * 2 * np.pi / 24).cos()
    e["min_sin"] = (minute * 2 * np.pi / 60).sin()
    e["min_cos"] = (minute * 2 * np.pi / 60).cos()
    e["dow_sin"] = (dow * 2 * np.pi / 7).sin()
    e["dow_cos"] = (dow * 2 * np.pi / 7).cos()
    e["is_us"] = ((hour >= 13) & (hour < 21)).cast(pl.Float32)   # 美盘活跃时段
    e["is_asia"] = ((hour >= 0) & (hour < 8)).cast(pl.Float32)
    e["is_eu"] = ((hour >= 8) & (hour < 13)).cast(pl.Float32)
    # 距当日开盘的累计收益 (组内首值广播)
    e["ret_day"] = (C / C.first().over(DT.dt.truncate("1d")) - 1) * 100

    # ---- 增强: 长周期收益/均值回归 (提升 top1% 区分度) ----
    for w in [360, 480, 720, 1440]:          # 6h/8h/12h/24h
        e[f"lr_{w}"] = (C / C.shift(w)).log()
        lo_l = L.rolling_min(w); hi_l = H.rolling_max(w)
        e[f"pos_{w}"] = (C - lo_l) / (hi_l - lo_l + EPS)
        e[f"dd_{w}"] = (C / hi_l - 1) * 100
        e[f"ru_{w}"] = (C / lo_l - 1) * 100
        mu_l = C.rolling_mean(w); sd_l = C.rolling_std(w, ddof=1)
        e[f"z_{w}"] = (C - mu_l) / (sd_l + EPS)
    # 多周期动量一致性 (短中长期是否同向)
    e["mom_align_30_240"] = ((C / C.shift(30) - 1) * (C / C.shift(240) - 1) * 1e8)
    e["mom_align_60_1440"] = ((C / C.shift(60) - 1) * (C / C.shift(1440) - 1) * 1e8)

    # ---- 增强: 主动买卖量微观结构 ----
    # tbr(主动买占比)的 z-score 与极值
    for w in [10, 30, 60, 120]:
        tbmu = tbr.rolling_mean(w); tbsd = tbr.rolling_std(w, ddof=1)
        e[f"tbr_z_{w}"] = (tbr - tbmu) / (tbsd + EPS)
    # 主动买量加速度 (主导流突变)
    for w in [10, 30]:
        e[f"tb_acc_{w}"] = TB.rolling_sum(w).diff(5) / (TB.rolling_sum(w) + EPS) * 100
        e[f"ts_acc_{w}"] = TS.rolling_sum(w).diff(5) / (TS.rolling_sum(w) + EPS) * 100
    # 主动买卖量不平衡的动量
    cvd = (TB - TS) / (TB + TS + EPS)
    for w in [5, 15, 30]:
        e[f"cvd_slope_{w}"] = (cvd.diff(w)) 
    # tbr 极值/方向: 最近N根主动买占比维持高位
    e["tbr_hi_60"] = (tbr.rolling_mean(60) > 0.55).cast(pl.Float32)
    e["cvd_dir_30"] = (cvd.rolling_mean(30) > cvd.rolling_mean(60)) * 1.0

    # ---- 增强: 波动状态与K线结构 ----
    # 波动率归一化(相对近期)
    e["rvol_z_60"] = (e["rvol_60"] - e["rvol_60"].rolling_mean(240)) / (e["rvol_60"].rolling_std(240, ddof=1) + EPS)
    # 连续波动放大/收缩
    e["rvol_dir"] = (e["rvol_30"] - e["rvol_60"]) / (e["rvol_60"] + EPS)
    # 实体/范围在近期的位置 (动量确认)
    for w in [30, 60]:
        e[f"body_pos_{w}"] = (body - body.rolling_min(w)) / (body.rolling_max(w) - body.rolling_min(w) + EPS)
    # HH/HL 结构 (N根内最高点距现在的回撤)
    for w in [20, 60]:
        e[f"hh_dd_{w}"] = (C / H.rolling_max(w) - 1) * 100
        e[f"ll_ru_{w}"] = (C / L.rolling_min(w) - 1) * 100

    # ---- 增强: 回报分布偏度 (价格结构尾部) ----
    for w in [30, 60]:
        e[f"lr_skew_{w}"] = lr.rolling_skew(w)
        e[f"lr_kurt_{w}"] = lr.rolling_kurtosis(w)
    # 上升/下降K线实体比
    up_body = pl.max_horizontal(body, 0.0)
    dn_body = pl.min_horizontal(body, 0.0)
    for w in [10, 30]:
        e[f"up_body_ratio_{w}"] = up_body.rolling_mean(w) / (up_body.rolling_mean(w) + (-dn_body).rolling_mean(w) + EPS)
    # 近期最大单根波动 (尾部风险)
    rng_r = rng.rolling_max(30)
    e["max_range_30"] = rng_r / (rng.rolling_mean(240) + EPS)

    out = df.select([expr.alias(name) for name, expr in e.items()])

    # ---- 连续同向K线 (需临时分组列) ----
    green = (df["close"] > df["open"]).cast(pl.Int8)
    grp = (green.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    tmp = df.with_columns([green.alias("_g"), grp.alias("_grp")])
    # 组内累计序号(自1起); 乘 green/(1-green) 清零反向K线; 全程 int32/float32, 不用 float64
    run_idx = tmp.select(pl.col("_g").cum_count().over("_grp")).to_series().cast(pl.Int32)
    dn_run_idx = tmp.select((1 - pl.col("_g")).cum_count().over("_grp")).to_series().cast(pl.Int32)
    out = out.with_columns([
        (green.cast(pl.Float32) * run_idx.cast(pl.Float32) / 20).clip(0, 1).alias("streak_up"),
        ((1 - green.cast(pl.Float32)) * dn_run_idx.cast(pl.Float32) / 20).clip(0, 1).alias("streak_dn"),
    ])
    return out.cast(pl.Float32)


def build_label(c, horizon):
    """label: 1 若 close[t+horizon] > close[t] (预测未来horizon根后涨), 否则 0。"""
    return (c.shift(-horizon) > c).cast(pl.Int8)


def build_ret_future(c, horizon):
    """未来horizon根的真实对数收益(仅用于分析/确认阈值, 不是特征)。"""
    return (c.shift(-horizon) / c).log().cast(pl.Float32)
