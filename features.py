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
    """df: polars DataFrame, 列含 ts/ open/high/low/close/buy_vol/sell_vol/funding。
    返回与 df 等长的特征 DataFrame(float32)。"""
    C = pl.col("close")
    O = pl.col("open")
    H = pl.col("high")
    L = pl.col("low")
    TB = pl.col("buy_vol")          # 主动买量
    TS = pl.col("sell_vol")         # 主动卖量
    F = pl.col("funding")           # 资金费率
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

    # ---- 增强: 交叉特征 (提升树模型区分度) ----
    pos_30 = e["pos_30"]; pos_120 = e["pos_120"]
    tbr_z_30 = e["tbr_z_30"]
    rvol_60 = e["rvol_60"]; rvol_30 = e["rvol_30"]
    cvd_30 = e["cvd_30"]; cvd_60 = e["cvd_60"]
    di_plus = e["di_plus"]; di_minus = e["di_minus"]
    mom_60 = e["mom_60"]
    z_30 = e["z_30"]; z_120 = e["z_120"]
    # 价格位置 × 买量强度: 高位+买量强 = 突破信号
    e["pos_tbr_interact"] = pos_30 * tbr_z_30
    # 波动率 × 动量: 高波动+强动量 = 趋势延续
    e["vol_mom_interact"] = rvol_60 * mom_60 / 100
    # 价格位置 × 主动净买: 低位+买量涌入 = 反弹信号
    e["pos_cvd_interact"] = (1 - pos_120) * cvd_60
    # DMI 方向差: DI+ - DI- 的强度, 正值=多头趋势
    e["di_spread"] = di_plus - di_minus
    # DMI 确认: DI+ > DI- 且位置高
    e["di_uptrend"] = (di_plus > di_minus).cast(pl.Float32) * pos_30
    # 动量 × 波动率方向: 动量向上且波动扩张
    e["mom_vol_confirm"] = (mom_60 > 0).cast(pl.Float32) * (rvol_60 / (rvol_30 + EPS))
    # Z分数差值: 短周期偏离vs长周期均值回归张力
    e["z_divergence"] = z_30 - z_120
    # CVD 加速度: 短期净买与中期净买之差
    cvd_5 = e["cvd_5"] if "cvd_5" in e else (TB - TS).rolling_sum(5) / ((TB + TS).rolling_sum(5) + EPS)
    e["cvd_accel"] = cvd_5 - cvd_30
    # 波动率 × CVD: 高波动+强买量 = 方向性信号
    e["vol_cvd_interact"] = rvol_60 * cvd_30

    out = df.select([expr.alias(name) for name, expr in e.items()])

    # ---- 连续同向K线 (需临时分组列) ----
    green = (df["close"] > df["open"]).cast(pl.Int8)
    grp = (green.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    tmp = df.with_columns([green.alias("_g"), grp.alias("_grp")])
    run_idx = tmp.select(pl.col("_g").cum_count().over("_grp")).to_series().cast(pl.Int32)
    dn_run_idx = tmp.select((1 - pl.col("_g")).cum_count().over("_grp")).to_series().cast(pl.Int32)
    out = out.with_columns([
        (green.cast(pl.Float32) * run_idx.cast(pl.Float32) / 20).clip(0, 1).alias("streak_up"),
        ((1 - green.cast(pl.Float32)) * dn_run_idx.cast(pl.Float32) / 20).clip(0, 1).alias("streak_dn"),
    ])

    # ================================================================
    # 独特新特征: CVD深度衍生 (使用原始df列) — 精选有信号+去噪声
    # ================================================================
    Cc = df["close"]; Oo = df["open"]; Hh = df["high"]; Ll = df["low"]
    TBb = df["buy_vol"]; TSs = df["sell_vol"]
    tb_ts = TBb + TSs + EPS
    cvd_delta = (TBb - TSs) / tb_ts
    lr_1 = (Cc / Cc.shift(1)).log()

    # 1. CVD 持久性: CVD连续同向的根数
    cvd_sign = (TBb - TSs).sign().cast(pl.Int8)
    cvd_grp = (cvd_sign.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    cvd_tmp = df.with_columns([cvd_sign.alias("_cs"), cvd_grp.alias("_cg")])
    cvd_run = cvd_tmp.select(pl.col("_cs").cum_count().over("_cg")).to_series().cast(pl.Int32)
    cvd_run_pos = (cvd_sign.cast(pl.Float32) > 0).cast(pl.Float32) * cvd_run.cast(pl.Float32)
    cvd_run_neg = (cvd_sign.cast(pl.Float32) < 0).cast(pl.Float32) * cvd_run.cast(pl.Float32)
    out = out.with_columns([
        (cvd_run_pos / 50).clip(0, 1).alias("cvd_run_pos"),
        (cvd_run_neg / 50).clip(0, 1).alias("cvd_run_neg"),
        ((cvd_run_pos - cvd_run_neg) / 50).clip(-1, 1).alias("cvd_run_net"),
    ])

    # 2. CVD 吸收率 (保留)
    cvd_30 = (TBb - TSs).rolling_sum(30) / ((TBb + TSs).rolling_sum(30) + EPS)
    cvd_60 = (TBb - TSs).rolling_sum(60) / ((TBb + TSs).rolling_sum(60) + EPS)
    lr_60 = (Cc / Cc.shift(60)).log()
    for w in [30, 60]:
        cvd_w = (TBb - TSs).rolling_sum(w) / ((TBb + TSs).rolling_sum(w) + EPS)
        px_vol_w = lr_1.rolling_std(w, ddof=1) * 100
        out = out.with_columns([
            (cvd_w.abs() / (px_vol_w + 1e-6)).alias(f"cvd_absorption_{w}"),
        ])
    out = out.with_columns([
        (cvd_30.sign().cast(pl.Float32) * lr_60.sign().cast(pl.Float32)).alias("cvd_px_confluence_30"),
        (cvd_60.sign().cast(pl.Float32) * lr_60.sign().cast(pl.Float32)).alias("cvd_px_confluence_60"),
    ])

    # 3. CVD 状态分类 (保留, 有信号)
    cvd_trend_60 = cvd_60 - cvd_60.shift(60)
    out = out.with_columns([
        ((cvd_60 > 0.02).cast(pl.Float32) * (cvd_trend_60 > 0).cast(pl.Float32)).alias("cvd_accumulate_60"),
        ((cvd_60 < -0.02).cast(pl.Float32) * (cvd_trend_60 < 0).cast(pl.Float32)).alias("cvd_distribute_60"),
        (cvd_trend_60 * 10).clip(-1, 1).alias("cvd_trend_60"),
    ])

    # 4. (已移除) 不再使用 cvd_vol_z / cvd_px_vol_ratio (噪声)
    # 5. (已移除) 不再使用 Roll's Spread (不适合加密货币)
    # 6. (已移除) 不再使用 price_impact (噪声)
    # 7. (已移除) 不再使用 tb_skew / ts_skew (噪声), 保留 vol_skew_diff

    # 8. 成交量偏度差 (保留, 有弱信号)
    for w in [30, 60]:
        tb_skew = TBb.cast(pl.Float64).rolling_skew(w).cast(pl.Float32)
        ts_skew = TSs.cast(pl.Float64).rolling_skew(w).cast(pl.Float32)
        out = out.with_columns([
            (tb_skew - ts_skew).clip(-3, 3).alias(f"vol_skew_diff_{w}"),
        ])

    # ================================================================
    # 新增: 真正独特的CVD衍生特征 (市场微观结构)
    # ================================================================

    # 9. CVD 分歧: 价格与CVD方向不一致 → 阻力/支撑信号
    cvd_delta_5 = cvd_delta.rolling_mean(5)
    lr_5 = lr_1.rolling_mean(5)
    price_up_5 = (lr_5 > 0).cast(pl.Float32)
    price_dn_5 = (lr_5 < 0).cast(pl.Float32)
    cvd_bullish = (cvd_delta_5 > 0.02).cast(pl.Float32)
    cvd_bearish = (cvd_delta_5 < -0.02).cast(pl.Float32)
    out = out.with_columns([
        # 看涨分歧: 价格跌但CVD为正 → 有资金在吸筹 (独特)
        (price_dn_5 * cvd_bullish * (cvd_delta_5.abs() * 10).clip(0, 1)).alias("cvd_bull_divergence"),
        # 看跌分歧: 价格涨但CVD为负 → 有资金在派发 (独特)
        (price_up_5 * cvd_bearish * (cvd_delta_5.abs() * 10).clip(0, 1)).alias("cvd_bear_divergence"),
    ])

    # 10. CVD 流动强度: 结合量能的CVD信号 (高CVD+高成交量=强信号)
    tb_ts_ma30 = tb_ts.rolling_mean(30)
    tb_ts_std30 = tb_ts.rolling_std(30, ddof=1)
    vol_z = (tb_ts - tb_ts_ma30) / (tb_ts_std30 + EPS)
    out = out.with_columns([
        # 流动强度: 净方向 × 成交量异常度 (独特)
        (cvd_delta.sign().cast(pl.Float32) * vol_z.clip(-3, 3) / 3).alias("cvd_flow_intensity"),
        # CVD 极端脉冲: 大单边放量 (独特)
        ((cvd_delta.abs() > 0.3).cast(pl.Float32) * (tb_ts > tb_ts_ma30 * 1.5).cast(pl.Float32)).alias("cvd_surge"),
    ])

    # 11. CVD 多时间尺度动量 (独特)
    for w in [10, 60]:
        cvd_mom = cvd_delta - cvd_delta.shift(w)
        out = out.with_columns([
            cvd_mom.clip(-0.5, 0.5).alias(f"cvd_mom_{w}"),
        ])

    # 12. 价格-CVD 效率比: 价格变动中有多少被CVD解释 (独特)
    # 价格变动大但CVD小 → 非流动性驱动 (噪声)
    # 价格变动小但CVD大 → 吸筹/派发 (信号)
    px_vol_30 = lr_1.rolling_std(30, ddof=1)
    cvd_vol_30 = cvd_delta.rolling_std(30, ddof=1)
    out = out.with_columns([
        (px_vol_30 / (cvd_vol_30 + EPS)).log1p().clip(0, 5).alias("px_cvd_inefficiency"),
    ])

    # ================================================================
    # 新增: 资金费率特征 (独特, 大多数模型不使用1分钟级别资金费率)
    # 使用 df["funding"] 直接引用原始列, 避免 out 中无 "funding" 列
    # ================================================================
    Ff = df["funding"]

    # 13. 资金费率原始值 + 滚动统计
    out = out.with_columns([
        Ff.cast(pl.Float32).alias("funding_raw"),
    ])
    for w in [5, 15, 30, 60]:
        out = out.with_columns([
            Ff.rolling_mean(w).cast(pl.Float32).alias(f"funding_ma_{w}"),
            (Ff - Ff.rolling_mean(w)).cast(pl.Float32).alias(f"funding_dev_{w}"),
        ])

    # 14. 资金费率Z分数 (极端值检测)
    for w in [30, 60, 120]:
        f_mu = Ff.rolling_mean(w)
        f_sd = Ff.rolling_std(w, ddof=1)
        out = out.with_columns([
            ((Ff - f_mu) / (f_sd + EPS)).clip(-5, 5).cast(pl.Float32).alias(f"funding_z_{w}"),
        ])

    # 15. 资金费率变化率 (动量)
    for w in [5, 15, 30]:
        out = out.with_columns([
            (Ff - Ff.shift(w)).cast(pl.Float32).alias(f"funding_delta_{w}"),
        ])

    # 16. 资金费率与CVD交互 (独特: 资金费率+主动买卖量)
    f_ma30 = Ff.rolling_mean(30)
    TBb2 = df["buy_vol"]; TSs2 = df["sell_vol"]
    cvd_30_fund = (TBb2 - TSs2).rolling_sum(30) / ((TBb2 + TSs2).rolling_sum(30) + EPS)
    out = out.with_columns([
        # 资金费率 * CVD: 高资金费率+主动买 = 强多头信心
        (f_ma30 * cvd_30_fund * 100).cast(pl.Float32).alias("funding_cvd_interact"),
        # 资金费率方向 * CVD方向: 一致时信号强
        (Ff.sign().cast(pl.Float32) * cvd_30_fund.sign().cast(pl.Float32)).cast(pl.Float32).alias("funding_cvd_confluence"),
    ])

    # 17. 资金费率与价格分歧 (独特)
    Cc2 = df["close"]
    lr_60_fund = (Cc2 / Cc2.shift(60)).log()
    f_ma60 = Ff.rolling_mean(60)
    out = out.with_columns([
        ((lr_60_fund < -0.005).cast(pl.Float32) * (f_ma60 < -0.0001).cast(pl.Float32)).cast(pl.Float32).alias("funding_bull_div"),
        ((lr_60_fund > 0.005).cast(pl.Float32) * (f_ma60 > 0.0001).cast(pl.Float32)).cast(pl.Float32).alias("funding_bear_div"),
    ])

    # ================================================================
    # 第二轮独特特征挖掘: 真正与众不同的视角
    # ================================================================
    Cc3 = df["close"]; Oo3 = df["open"]; Hh3 = df["high"]; Ll3 = df["low"]
    TBb3 = df["buy_vol"]; TSs3 = df["sell_vol"]
    tb_ts3 = TBb3 + TSs3 + EPS
    cvd_delta3 = (TBb3 - TSs3) / tb_ts3
    lr_1_3 = (Cc3 / Cc3.shift(1)).log()

    # 18. CVD Aging: CVD信号的"老化"效应
    # 持续同向CVD越久, 衰竭概率越高, 预测力衰减
    cvd_sign3 = (TBb3 - TSs3).sign().cast(pl.Int8)
    cvd_grp3 = (cvd_sign3.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    cvd_tmp3 = df.with_columns([cvd_sign3.alias("_cs2"), cvd_grp3.alias("_cg2")])
    cvd_run3 = cvd_tmp3.select(pl.col("_cs2").cum_count().over("_cg2")).to_series().cast(pl.Int32)
    # CVD老化衰减因子: e^(-run_length/30), 持续越久信号越弱
    cvd_age = pl.from_numpy(np.exp(-cvd_run3.to_numpy().astype(np.float32) / 30.0)).to_series()
    cvd_run_net3 = (cvd_sign3.cast(pl.Float32) * cvd_run3.cast(pl.Float32) / 50).clip(-1, 1)
    out = out.with_columns([
        cvd_age.alias("cvd_age_decay"),                               # 老化衰减 [0,1]
        (cvd_run_net3 * cvd_age).alias("cvd_age_signal"),             # 老化后的信号
        (cvd_run3.cast(pl.Float32) / 100).clip(0, 1).alias("cvd_run_len"),  # 原始运行长度
    ])

    # 19. CVD多时间尺度一致性 (独特: 不同周期的CVD是否同向)
    cvd_5_3 = (TBb3 - TSs3).rolling_sum(5) / ((TBb3 + TSs3).rolling_sum(5) + EPS)
    cvd_15_3 = (TBb3 - TSs3).rolling_sum(15) / ((TBb3 + TSs3).rolling_sum(15) + EPS)
    cvd_30_3 = (TBb3 - TSs3).rolling_sum(30) / ((TBb3 + TSs3).rolling_sum(30) + EPS)
    cvd_60_3 = (TBb3 - TSs3).rolling_sum(60) / ((TBb3 + TSs3).rolling_sum(60) + EPS)
    cvd_120_3 = (TBb3 - TSs3).rolling_sum(120) / ((TBb3 + TSs3).rolling_sum(120) + EPS)
    out = out.with_columns([
        # 多尺度CVD一致性: 所有尺度同向=强信号
        ((cvd_5_3.sign() + cvd_15_3.sign() + cvd_30_3.sign() + cvd_60_3.sign() + cvd_120_3.sign()).abs().cast(pl.Float32) / 5)
        .alias("cvd_align_5_120"),
        # 短-中-长 三层一致性
        ((cvd_15_3.sign() + cvd_60_3.sign() + cvd_120_3.sign()).abs().cast(pl.Float32) / 3)
        .alias("cvd_align_15_120"),
        # 短周期CVD斜率: 短期CVD变化方向
        (cvd_5_3 - cvd_30_3).clip(-0.5, 0.5).alias("cvd_slope_5_30"),
        # 长周期CVD趋势: 中期CVD变化方向
        (cvd_30_3 - cvd_120_3).clip(-0.5, 0.5).alias("cvd_slope_30_120"),
    ])

    # 20. 收益率自相关性 (Hurst指数代理, 独特: 捕捉趋势/均值回归状态)
    for w in [30, 60, 120]:
        lr_roll = lr_1_3
        # lag-1自相关: 短期趋势持续性
        lr_lag = lr_roll.shift(1)
        corr_1 = ((lr_roll - lr_roll.rolling_mean(w)) * (lr_lag - lr_lag.rolling_mean(w))).rolling_sum(w) / \
                 (lr_roll.rolling_std(w, ddof=1) * lr_lag.rolling_std(w, ddof=1) * w + EPS)
        # lag-2自相关: 中期记忆
        lr_lag2 = lr_roll.shift(2)
        corr_2 = ((lr_roll - lr_roll.rolling_mean(w)) * (lr_lag2 - lr_lag2.rolling_mean(w))).rolling_sum(w) / \
                 (lr_roll.rolling_std(w, ddof=1) * lr_lag2.rolling_std(w, ddof=1) * w + EPS)
        out = out.with_columns([
            corr_1.clip(-1, 1).cast(pl.Float32).alias(f"lr_ac1_{w}"),
            corr_2.clip(-1, 1).cast(pl.Float32).alias(f"lr_ac2_{w}"),
        ])

    # 21. Volume Profile: 成交量在价格区间内的分布偏度 (独特)
    # 量集中在顶部 → 阻力; 量集中在底部 → 支撑
    for w in [30, 60, 120]:
        rng_w = (Hh3.rolling_max(w) - Ll3.rolling_min(w) + EPS)
        # 价格在区间内的位置 [0,1]
        px_pos = (Cc3 - Ll3.rolling_min(w)) / rng_w
        # 加权成交量位置: 成交量×价格位置 / 总成交量
        tot_vol = TBb3 + TSs3
        vol_pos = (tot_vol * px_pos).rolling_sum(w) / (tot_vol.rolling_sum(w) + EPS)
        # Volume Profile Skew: 加权位置偏离0.5 → 正=量偏上(阻力), 负=量偏下(支撑)
        vp_skew = (vol_pos - 0.5) * 2
        out = out.with_columns([
            vp_skew.clip(-1, 1).cast(pl.Float32).alias(f"vp_skew_{w}"),
        ])

    # 22. 资金费率制度持续时间 (独特: 拥挤交易持续性)
    fund_sign = Ff.sign().cast(pl.Int8)
    fund_grp = (fund_sign.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    fund_tmp = df.with_columns([fund_sign.alias("_fs"), fund_grp.alias("_fg")])
    fund_run = fund_tmp.select(pl.col("_fs").cum_count().over("_fg")).to_series().cast(pl.Int32)
    fund_run_len = fund_run.cast(pl.Float32) / 100  # 归一化
    # 资金费率制度强度: 运行长度 × 资金费率绝对值
    fund_regime = fund_run_len * Ff.abs().cast(pl.Float32) * 10000
    out = out.with_columns([
        fund_run_len.clip(0, 1).alias("fund_regime_len"),
        fund_regime.clip(-1, 1).alias("fund_regime_strength"),
        # 资金费率加速度: 近N根变化率
        (Ff - Ff.shift(30)).cast(pl.Float32).alias("fund_delta_30"),
    ])

    # 23. CVD-Volume Ratio: 单位成交量中的CVD (独特: 方向性信念强度)
    for w in [15, 30, 60]:
        cvd_w = (TBb3 - TSs3).rolling_sum(w)
        vol_w = TBb3.rolling_sum(w) + TSs3.rolling_sum(w) + EPS
        # 每单位成交量中的净方向: 高值 = 强信念
        cvd_vol_ratio = cvd_w / vol_w
        out = out.with_columns([
            cvd_vol_ratio.clip(-1, 1).cast(pl.Float32).alias(f"cvd_vol_ratio_{w}"),
        ])

    # 24. 价格水平吸引子 (独特: 价格记忆效应)
    # 价格在当前水平附近停留了多少次? 停留越多 = 支撑/阻力越强
    for w in [60, 120]:
        lo_w = Ll3.rolling_min(w)
        hi_w = Hh3.rolling_max(w)
        rng_w = (hi_w - lo_w + EPS)
        # 将价格离散化为20个区间, 统计当前区间被访问的次数
        px_bin = ((Cc3 - lo_w) / rng_w * 20).cast(pl.Int32).clip(0, 19)
        # 使用shift+eq来近似统计同区间次数 (简化版)
        same_bin = (px_bin == px_bin.shift(1)).cast(pl.Float32).rolling_mean(w)
        out = out.with_columns([
            same_bin.clip(0, 1).cast(pl.Float32).alias(f"px_memory_{w}"),
        ])

    # 25. 波动率锥: 不同时间尺度波动率比率 (独特: 市场结构)
    rvol_5_3 = lr_1_3.rolling_std(5, ddof=1) * 100
    rvol_15_3 = lr_1_3.rolling_std(15, ddof=1) * 100
    rvol_60_3 = lr_1_3.rolling_std(60, ddof=1) * 100
    rvol_240_3 = lr_1_3.rolling_std(240, ddof=1) * 100
    out = out.with_columns([
        # 短期/长期波动率比: 高=短期波动主导(噪声), 低=长期趋势主导
        ((rvol_5_3 / rvol_60_3).log1p()).clip(0, 3).cast(pl.Float32).alias("vol_cone_5_60"),
        ((rvol_15_3 / rvol_60_3).log1p()).clip(0, 3).cast(pl.Float32).alias("vol_cone_15_60"),
        ((rvol_60_3 / rvol_240_3).log1p()).clip(0, 3).cast(pl.Float32).alias("vol_cone_60_240"),
        # 波动率加速: 短期波动率变化率
        ((rvol_5_3 / (rvol_5_3.shift(5) + EPS)).log1p()).clip(-1, 1).cast(pl.Float32).alias("vol_accel_5"),
    ])

    # 26. 价格-CVD相位差 (独特: 价格和CVD的领先滞后关系)
    # 当CVD领先价格变动时 = 信号; 价格领先CVD时 = 噪声
    lr_5_3 = (Cc3 / Cc3.shift(5)).log()
    cvd_5_3b = (TBb3 - TSs3).rolling_sum(5) / ((TBb3 + TSs3).rolling_sum(5) + EPS)
    cvd_5_lag1 = cvd_5_3b.shift(1)
    cvd_5_lag2 = cvd_5_3b.shift(2)
    cvd_5_lag3 = cvd_5_3b.shift(3)
    out = out.with_columns([
        # CVD滞后1期与价格的相关性: CVD领先1根K线预测价格
        (cvd_5_lag1.sign().cast(pl.Float32) * lr_5_3.sign().cast(pl.Float32)).alias("cvd_lead_1"),
        (cvd_5_lag2.sign().cast(pl.Float32) * lr_5_3.sign().cast(pl.Float32)).alias("cvd_lead_2"),
        (cvd_5_lag3.sign().cast(pl.Float32) * lr_5_3.sign().cast(pl.Float32)).alias("cvd_lead_3"),
    ])

    # ================================================================
    # 第三轮: 四个全新认知框架的特征 (信息论/随机游走检验/CVD动力学/行为金融)
    # ================================================================

    # ---- 信息论: 收益率符号熵 (市场确定性) ----
    # 熵低 = 方向高度一致(趋势明确); 熵高 = 完全随机(噪声市)。与"方向+幅度"正交。
    def shannon_entropy(p_expr):
        return pl.when((p_expr > 0.02) & (p_expr < 0.98)).then(
            -(p_expr * p_expr.log(2) + (1 - p_expr) * (1 - p_expr).log(2))
        ).otherwise(0.0)

    for w in [30, 60]:
        up_ratio = (lr_1_3 > 0).cast(pl.Float32).rolling_mean(w)
        out = out.with_columns([
            shannon_entropy(up_ratio).cast(pl.Float32).alias(f"entropy_ret_{w}"),
        ])

    # ---- 随机游走检验: Lo-MacKinlay 方差比 (趋势/均值回归状态) ----
    # VR = Var(聚合k期收益) / (k * Var(单期收益))。VR>1 趋势市, VR<1 均值回归市。
    # 与 lr_ac(滞后相关)互补: VR 同时考虑多期标度, 对1分钟噪音更稳健。
    lr_1_v = lr_1_3.rolling_var(60, ddof=1)
    for k in [5, 10]:
        lr_k = (Cc3 / Cc3.shift(k)).log()
        vr = lr_k.rolling_var(60, ddof=1) / (lr_1_v * k + EPS)
        out = out.with_columns([
            vr.clip(0.2, 3.0).cast(pl.Float32).alias(f"vr_{k}_60"),
        ])

    # ---- CVD 动力学: 耗竭 (运行长但价格移动不足 = 衰竭反转) ----
    # 连续同向主动流本应推动价格; 若运行很长但实际位移 << 理论位移, 说明被对冲/派发吸收 → 反转。
    for w in [30, 60]:
        lr_w = (Cc3 / Cc3.shift(w)).log()
        rvol_w = lr_1_3.rolling_std(w, ddof=1)
        move_ratio = lr_w.abs() / (rvol_w * np.sqrt(w) + EPS)   # 实际位移/理论位移
        exhaustion = cvd_run_net3 * (1.0 - move_ratio.clip(0, 1))  # 运行净方向 × 移动不足度
        out = out.with_columns([
            exhaustion.clip(-1, 1).cast(pl.Float32).alias(f"cvd_exhaustion_{w}"),
        ])

    # ---- CVD 动力学: 峰值回撤 / 谷值反弹 (派发/吸筹完成度) ----
    # CVD 30根累计从近期峰值大幅回撤 = 多头派发进行中; 从谷值大幅反弹 = 空头回补/吸筹。
    cvd_30b = (TBb3 - TSs3).rolling_sum(30) / ((TBb3 + TSs3).rolling_sum(30) + EPS)
    out = out.with_columns([
        (cvd_30b - cvd_30b.rolling_max(30)).clip(-1, 0).cast(pl.Float32).alias("cvd_pullback_30"),
        (cvd_30b - cvd_30b.rolling_min(30)).clip(0, 1).cast(pl.Float32).alias("cvd_rebound_30"),
    ])

    # ---- CVD 动力学: 连续版领先相关性 (CVD滞后1期与当期收益的滚动相关) ----
    # 符号版(cvd_lead_*)丢失强度信息; 连续相关捕捉"CVD领先预测力"的强弱。
    x_lag = cvd_5_3b.shift(1)
    mx = x_lag.rolling_mean(60)
    my = lr_5_3.rolling_mean(60)
    sx = x_lag.rolling_std(60, ddof=1)
    sy = lr_5_3.rolling_std(60, ddof=1)
    ccorr = ((x_lag - mx) * (lr_5_3 - my)).rolling_mean(60) / (sx * sy + EPS)
    out = out.with_columns([
        ccorr.clip(-1, 1).cast(pl.Float32).alias("cvd_lead_corr_60"),
    ])

    # ---- CVD 动力学: 主动流方向熵 (机构单边建仓 vs 散户对倒) ----
    # 主动流方向高度一致 = 单边大资金行为; 高度随机 = 噪声对倒。
    for w in [30, 60]:
        up_cvd = (cvd_delta3 > 0).cast(pl.Float32).rolling_mean(w)
        out = out.with_columns([
            shannon_entropy(up_cvd).cast(pl.Float32).alias(f"cvd_micro_entropy_{w}"),
        ])

    # ---- 行为金融: 心理整数关口距离 (1000 整数位) ----
    # 加密市场对整数价位有强锚定: 价格贴近关口 → 支撑/阻力; 远离 → 自由区间。
    rem1000 = (Cc3 % 1000) / 1000
    out = out.with_columns([
        pl.min_horizontal(rem1000, 1 - rem1000).cast(pl.Float32).alias("round_dist_1000"),
    ])

    # ---- 行为金融: 成交量时间集中度 (事件驱动 vs 均匀流动) ----
    # 量集中在最近5根 = 事件/脉冲驱动(追涨杀跌); 均匀分布 = 持续性建仓。
    tot3 = TBb3 + TSs3 + EPS
    out = out.with_columns([
        (tot3.rolling_sum(5) / (tot3.rolling_sum(30) + EPS)).clip(0, 2).cast(pl.Float32).alias("vol_conc_5_30"),
        (tot3.rolling_sum(10) / (tot3.rolling_sum(60) + EPS)).clip(0, 2).cast(pl.Float32).alias("vol_conc_10_60"),
    ])

    # ---- 行为金融: 波动率不对称 (杠杆效应/恐慌) ----
    # 下行波动 >> 上行波动 = 恐慌/持仓风险厌恶; 反向 = 乐观追涨。
    pos_r = lr_1_3.clip(lower_bound=0.0)
    neg_r = (-lr_1_3).clip(lower_bound=0.0)
    duv = neg_r.rolling_std(60, ddof=1) / (pos_r.rolling_std(60, ddof=1) + EPS)
    out = out.with_columns([
        duv.clip(0.2, 5.0).cast(pl.Float32).alias("down_up_vol_60"),
    ])

    # ================================================================
    # 第四轮: 过程型特征 (状态转变视角: 翻转/衰竭/反转/承接/量价质量)
    # 与"状态"特征(pos/tbr/cvd水平)正交, 捕捉"变化过程"而非"当前状态"
    # ================================================================

    # 27. CVD 方向翻转次数 (犹豫市 vs 单边市)
    # 翻转频繁 = 方向不明/对倒; 长时间不翻转 = 单边大资金行为
    cvd_pos5 = (cvd_5_3b > 0).cast(pl.Int8)
    cvd_flips = cvd_pos5.diff().fill_null(0).abs().rolling_sum(30)
    out = out.with_columns([
        (cvd_flips.cast(pl.Float32) / 30).clip(0, 1).cast(pl.Float32).alias("cvd_flip_30"),
    ])

    # 28. 空头衰竭: 近期阴线实体 vs 中期 (递减 = 卖压减弱 → 反转前兆)
    body3 = Cc3 - Oo3
    rng3 = (Hh3 - Ll3) + EPS
    dn_body = (-body3).clip(lower_bound=0.0) / rng3
    wane = dn_body.rolling_mean(5) / (dn_body.rolling_mean(30) + EPS)
    out = out.with_columns([
        wane.clip(0.2, 3.0).cast(pl.Float32).alias("down_body_wane_30"),
    ])

    # 29. CVD 反转初期: 短期CVD与长期CVD方向相反 (反转刚启动)
    # 短期转多但中期仍空 = 吸筹初期; 短期转空但中期仍多 = 派发初期
    rev = (cvd_5_3 > 0.02).cast(pl.Float32) * (cvd_60_3 < -0.02).cast(pl.Float32) - \
          (cvd_5_3 < -0.02).cast(pl.Float32) * (cvd_60_3 > 0.02).cast(pl.Float32)
    out = out.with_columns([
        rev.cast(pl.Float32).alias("cvd_reversal_30"),
    ])

    # 30. 支撑试探: 价格在区间低位 + 长下影线 (买盘承接成功)
    sup = (out["pos_30"] < 0.15).cast(pl.Float32) * (out["lo_wick"] > 0.5).cast(pl.Float32)
    out = out.with_columns([
        sup.cast(pl.Float32).alias("support_wick_30"),
    ])

    # 31. 量价质量: 上涨量/下跌量 (涨放量跌缩量 = 高质量多头趋势)
    up_vol = (lr_1_3 > 0).cast(pl.Float32) * tot3
    dn_vol = (lr_1_3 < 0).cast(pl.Float32) * tot3
    vq = up_vol.rolling_sum(60) / (dn_vol.rolling_sum(60) + EPS)
    out = out.with_columns([
        vq.clip(0.1, 10).cast(pl.Float32).alias("vol_quality_60"),
    ])

    # ================================================================
    # 第五轮: 最强信号交叉特征 (基于240特征单变量诊断: pos/vp_skew/cvd_vol_ratio/z 最强)
    # 显式交叉在极端稀疏区比树自动学习更高效; 全部输入重算, 避免顺序依赖
    # ================================================================
    Cc5 = df["close"]; Oo5 = df["open"]; Hh5 = df["high"]; Ll5 = df["low"]
    TBb5 = df["buy_vol"]; TSs5 = df["sell_vol"]
    Ff5 = df["funding"]
    lr_1_5 = (Cc5 / Cc5.shift(1)).log()
    lo_120_5 = Ll5.rolling_min(120); hi_120_5 = Hh5.rolling_max(120)
    pos_120_5 = (Cc5 - lo_120_5) / (hi_120_5 - lo_120_5 + EPS)
    z_120_5 = (Cc5 - Cc5.rolling_mean(120)) / (Cc5.rolling_std(120, ddof=1) + EPS)
    rvol_5_5 = lr_1_5.rolling_std(5, ddof=1) * 100
    rvol_60_5 = lr_1_5.rolling_std(60, ddof=1) * 100
    lr_30_5 = (Cc5 / Cc5.shift(30)).log()
    cvd_5_5 = (TBb5 - TSs5).rolling_sum(5) / ((TBb5 + TSs5).rolling_sum(5) + EPS)
    cvd_30_5 = (TBb5 - TSs5).rolling_sum(30) / ((TBb5 + TSs5).rolling_sum(30) + EPS)
    cvd_60_5 = (TBb5 - TSs5).rolling_sum(60) / ((TBb5 + TSs5).rolling_sum(60) + EPS)
    tot5 = TBb5 + TSs5 + EPS
    # vp_skew_60 (与第2轮定义一致)
    rng60_5 = Hh5.rolling_max(60) - Ll5.rolling_min(60) + EPS
    px_pos60_5 = (Cc5 - Ll5.rolling_min(60)) / rng60_5
    vp_skew_60_5 = (((tot5 * px_pos60_5).rolling_sum(60) / tot5.rolling_sum(60)) - 0.5) * 2

    cvd_accel_5 = (cvd_5_5 - cvd_30_5).clip(-0.5, 0.5)
    out = out.with_columns([
        # 32. 深跌 + 主动买加速 (反弹触发)
        ((1 - pos_120_5) * cvd_accel_5).clip(-1, 1).cast(pl.Float32).alias("deep_low_cvd_accel"),
        # 33. 创60根新低 + CVD未同步新低 (吸筹背离)
        ((Ll5 <= Ll5.rolling_min(60).shift(1)).cast(pl.Float32) * (cvd_60_5 > cvd_60_5.shift(1)).cast(pl.Float32))
        .cast(pl.Float32).alias("low_cvd_bull_div"),
        # 34. 波动锥 regime × 动量方向 (趋势市延续 / 均值回归市反转)
        ((rvol_5_5 / (rvol_60_5 + EPS)).log1p().clip(0, 3) * lr_30_5.sign().cast(pl.Float32))
        .cast(pl.Float32).alias("regime_mom_interact"),
        # 35. 资金费拥挤度 × 主动流方向
        ((Ff5.abs() * 10000).clip(0, 5) * cvd_30_5).clip(-2, 2).cast(pl.Float32).alias("funding_cvd_strength"),
        # 36. 低位 × 量分布偏度 (支撑确认)
        ((1 - pos_120_5) * vp_skew_60_5).clip(-1, 1).cast(pl.Float32).alias("pos_vp_support"),
        # 37. Z偏离 × CVD方向 (均值回归张力与资金方向)
        (z_120_5 * cvd_60_5).clip(-2, 2).cast(pl.Float32).alias("z_cvd_interact"),
        # 38. 突破 × 主动流确认 (有效突破)
        ((Cc5 > Hh5.rolling_max(60).shift(1)).cast(pl.Float32) * cvd_30_5).cast(pl.Float32).alias("brk_hi_cvd"),
        ((Cc5 < Ll5.rolling_min(60).shift(1)).cast(pl.Float32) * cvd_30_5).cast(pl.Float32).alias("brk_lo_cvd"),
        # 39. 长下影 + 买盘加速 (承接确认)
        (((pl.min_horizontal(Cc5, Oo5) - Ll5) / (Hh5 - Ll5 + EPS)) * cvd_accel_5).clip(-1, 1).cast(pl.Float32).alias("wick_cvd_accel"),
    ])

    # ================================================================
    # 第六轮: 结构/过程型深度特征 (与统计量正交的独立信息源)
    # 1) 流动效率与残余流: 滚动协方差把CVD分解为"价格驱动流"与"异常流"
    # 2) CVD记忆: 主动流自相关结构与衰减形状 (流能持续多久)
    # 3) 循环相位: 位置-速度相空间重构 (CVD周期的瞬时相位/角速度)
    # 4) 资金费结算时钟: 8h结算制度的行为周期 (00/08/16 UTC)
    # 5) 价格-流加速发散 + 吸收柱频率 (有量无价的柱占比)
    # ================================================================
    Cc6 = df["close"]; Oo6 = df["open"]; Hh6 = df["high"]; Ll6 = df["low"]
    TBb6 = df["buy_vol"]; TSs6 = df["sell_vol"]
    ts6 = df["ts"].cast(pl.Float64)
    tot6 = TBb6 + TSs6 + EPS
    lr_1_6 = (Cc6 / Cc6.shift(1)).log()
    lr_5_6 = (Cc6 / Cc6.shift(5)).log()
    lr_30_6 = (Cc6 / Cc6.shift(30)).log()
    cvd_delta6 = (TBb6 - TSs6) / tot6
    cvd_5_6 = (TBb6 - TSs6).rolling_sum(5) / ((TBb6 + TSs6).rolling_sum(5) + EPS)
    cvd_30_6 = (TBb6 - TSs6).rolling_sum(30) / ((TBb6 + TSs6).rolling_sum(30) + EPS)
    cvd_60_6 = (TBb6 - TSs6).rolling_sum(60) / ((TBb6 + TSs6).rolling_sum(60) + EPS)

    def _roll_cov(x, y, w):
        """滚动协方差: E[xy] - E[x]E[y] (纯polars, 只用t及之前)."""
        return (x * y).rolling_mean(w) - x.rolling_mean(w) * y.rolling_mean(w)

    # --- 1) 流动效率与残余流 ---
    var_lr30 = lr_1_6.rolling_std(30, ddof=1) ** 2
    var_lr60 = lr_1_6.rolling_std(60, ddof=1) ** 2
    var_cvd30 = cvd_delta6.rolling_std(30, ddof=1) ** 2
    var_cvd60 = cvd_delta6.rolling_std(60, ddof=1) ** 2
    cov_30 = _roll_cov(lr_1_6, cvd_delta6, 30)
    cov_60 = _roll_cov(lr_1_6, cvd_delta6, 60)
    r2_30 = (cov_30 ** 2) / (var_lr30 * var_cvd30 + EPS)
    r2_60 = (cov_60 ** 2) / (var_lr60 * var_cvd60 + EPS)
    out = out.with_columns([
        (cov_30 / (var_lr30 + EPS)).clip(-250, 250).cast(pl.Float32).alias("flow_beta_30"),
        (cov_60 / (var_lr60 + EPS)).clip(-250, 250).cast(pl.Float32).alias("flow_beta_60"),
        r2_30.clip(0, 1).cast(pl.Float32).alias("flow_r2_30"),
        r2_60.clip(0, 1).cast(pl.Float32).alias("flow_r2_60"),
        (lr_30_6.abs() / (cvd_30_6.abs() + EPS)).clip(0, 50).cast(pl.Float32).alias("flow_eff_30"),
    ])

    # --- 2) CVD记忆: 主动流自相关与衰减形状 ---
    cvd_ac1_20 = _roll_cov(cvd_delta6, cvd_delta6.shift(1), 20) / (cvd_delta6.rolling_std(20, ddof=1) ** 2 + EPS)
    cvd_ac1_60 = _roll_cov(cvd_delta6, cvd_delta6.shift(1), 60) / (cvd_delta6.rolling_std(60, ddof=1) ** 2 + EPS)
    cvd_ac6_60 = _roll_cov(cvd_delta6, cvd_delta6.shift(6), 60) / (cvd_delta6.rolling_std(60, ddof=1) ** 2 + EPS)
    out = out.with_columns([
        cvd_ac1_20.clip(-1, 1).cast(pl.Float32).alias("cvd_ac1_20"),
        cvd_ac1_60.clip(-1, 1).cast(pl.Float32).alias("cvd_ac1_60"),
        (cvd_ac6_60 / (cvd_ac1_60.abs() + EPS)).clip(0, 20).cast(pl.Float32).alias("cvd_ac_decay_60"),
    ])

    # --- 3) 循环相位: 归一化位置-速度相空间 (瞬时相位与角速度) ---
    # 位置与速度各自按其滚动波动归一, 使相空间两轴同量纲, 相位反映真实循环位置
    cyc_level = cvd_60_6 - cvd_60_6.rolling_mean(240)
    cyc_vel = cyc_level - cyc_level.shift(1)
    lvl_sd = cyc_level.rolling_std(240, ddof=1)
    vel_sd = cyc_vel.rolling_std(240, ddof=1)
    cyc_phase = pl.Series(np.arctan2(
        (cyc_vel / (vel_sd + EPS)).to_numpy(),
        (cyc_level / (lvl_sd + EPS)).to_numpy()))
    dphi = cyc_phase - cyc_phase.shift(1)
    dphi_w = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    out = out.with_columns([
        cyc_phase.cast(pl.Float32).alias("cyc_phase_60"),
        cyc_phase.sin().cast(pl.Float32).alias("cyc_sin_60"),
        cyc_phase.cos().cast(pl.Float32).alias("cyc_cos_60"),
        dphi_w.abs().clip(0, np.pi).cast(pl.Float32).alias("cyc_speed_60"),
    ])

    # --- 4) 资金费结算时钟 (8h周期: 00/08/16 UTC) ---
    since_settle = (ts6 % 28800) / 28800.0
    out = out.with_columns([
        since_settle.cast(pl.Float32).alias("fund_since_settle"),
        (since_settle > 0.875).cast(pl.Float32).alias("fund_pre_settle"),
        (since_settle < 0.125).cast(pl.Float32).alias("fund_post_settle"),
    ])

    # --- 5) 价格-流加速发散 + 吸收柱频率 ---
    px_accel_6 = lr_5_6 - lr_5_6.shift(5)
    cvd_accel_6 = cvd_5_6 - cvd_5_6.shift(5)
    lr1_std30 = lr_1_6.rolling_std(30, ddof=1)
    cvd_std30 = cvd_delta6.rolling_std(30, ddof=1)
    absorb_bar = ((lr_1_6.abs() < lr1_std30 * 0.5) & (cvd_delta6.abs() > cvd_std30)).cast(pl.Float32)
    tbr6 = TBb6 / tot6
    out = out.with_columns([
        px_accel_6.clip(-0.1, 0.1).cast(pl.Float32).alias("px_accel_5"),
        (px_accel_6 * cvd_accel_6).clip(-0.05, 0.05).cast(pl.Float32).alias("px_cvd_accel_div"),
        absorb_bar.rolling_mean(30).cast(pl.Float32).alias("absorb_bar_30"),
        (tbr6 > 0.55).cast(pl.Float32).rolling_mean(30).cast(pl.Float32).alias("buy_dom_freq_30"),
    ])

    # ================================================================
    # 第七轮: 日内季节性/跨日对比特征 (与滚动统计正交的独立信息源)
    # 诊断显示top特征全是位置/均值回归簇, 且缺少24h/48h/72h收益窗口
    # 与"昨日同刻"对比 = 捕捉日内周期性异常 (今日此刻的买压/量/位置是否异于昨日)
    # 15分钟K线: 96根/天, 672根/周
    # ================================================================
    Cc7 = df["close"]; Hh7 = df["high"]; Ll7 = df["low"]
    TBb7 = df["buy_vol"]; TSs7 = df["sell_vol"]
    Ff7 = df["funding"]
    tot7 = TBb7 + TSs7 + EPS
    lr_1_7 = (Cc7 / Cc7.shift(1)).log()
    # 补齐缺失的日/周级收益窗口 (24h/48h/72h)
    out = out.with_columns([
        (Cc7 / Cc7.shift(96)).log().cast(pl.Float32).alias("lr_96"),
        (Cc7 / Cc7.shift(192)).log().cast(pl.Float32).alias("lr_192"),
        (Cc7 / Cc7.shift(288)).log().cast(pl.Float32).alias("lr_288"),
    ])
    # 跨日对比基础量
    tbr30_7 = TBb7.rolling_sum(30) / ((TBb7 + TSs7).rolling_sum(30) + EPS)
    cvd30_7 = (TBb7 - TSs7).rolling_sum(30) / ((TBb7 + TSs7).rolling_sum(30) + EPS)
    rvol60_7 = lr_1_7.rolling_std(60, ddof=1) * 100
    pos120_7 = (Cc7 - Ll7.rolling_min(120)) / (Hh7.rolling_max(120) - Ll7.rolling_min(120) + EPS)
    out = out.with_columns([
        # 40. 主动买占比 与昨日同刻之差 (买压季节性异常)
        (tbr30_7 - tbr30_7.shift(96)).clip(-0.5, 0.5).cast(pl.Float32).alias("tbr_dod_96"),
        # 41. 主动净流 与昨日同刻之差
        (cvd30_7 - cvd30_7.shift(96)).clip(-0.5, 0.5).cast(pl.Float32).alias("cvd_dod_96"),
        # 42. 成交量 与昨日同刻之差 (相对近期量级归一)
        ((tot7 - tot7.shift(96)) / (tot7.rolling_mean(480) + EPS)).clip(-5, 5).cast(pl.Float32).alias("vol_dod_96"),
        # 43. 区间位置 与昨日同刻之差 (日内位置异常)
        (pos120_7 - pos120_7.shift(96)).clip(-1, 1).cast(pl.Float32).alias("pos_dod_96"),
        # 44. 波动率 与昨日同刻之差 (波动季节性异常)
        (rvol60_7 - rvol60_7.shift(96)).clip(-10, 10).cast(pl.Float32).alias("rvol_dod_96"),
        # 45. 资金费率 与昨日同刻之差
        (Ff7 - Ff7.shift(96)).clip(-0.005, 0.005).cast(pl.Float32).alias("funding_dod_96"),
        # 46. 周同比: 主动净流/位置 与上周同刻之差
        (cvd30_7 - cvd30_7.shift(672)).clip(-0.5, 0.5).cast(pl.Float32).alias("cvd_dod_672"),
        (pos120_7 - pos120_7.shift(672)).clip(-1, 1).cast(pl.Float32).alias("pos_dod_672"),
    ])
    # 47. 资金费结算周期内累计收益 (8h窗口内的漂移, 捕获结算制度周期)
    ret_fund_cycle7 = df.select(
        ((pl.col("close") / pl.col("close").first().over(pl.col("ts") // 28800) - 1) * 100)
        .alias("_rfc")
    )["_rfc"]
    out = out.with_columns([
        ret_fund_cycle7.cast(pl.Float32).alias("ret_fund_cycle"),
    ])

    return out.cast(pl.Float32)


def build_label(c, horizon):
    """label: 1 若 close[t+horizon] > close[t] (预测未来horizon根后涨), 否则 0。"""
    return (c.shift(-horizon) > c).cast(pl.Int8)


def build_ret_future(c, horizon):
    """未来horizon根的真实对数收益(仅用于分析/确认阈值, 不是特征)。"""
    return (c.shift(-horizon) / c).log().cast(pl.Float32)
