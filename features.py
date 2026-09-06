"""特征工程 (polars 版, float32 输出)。

严格遵守用户约束:
- 仅使用价格衍生特征 (O/H/L/C 及其滚动统计) 与主动买卖成交量 (buy_vol/sell_vol 及其滚动统计)
- 禁用: 总成交量 volume、quote_asset_volume、ATR、number_of_trades
- 特征均只使用 t 时刻及之前的收盘信息 (滚动/滞后), 无未来函数泄漏
- 滚动窗口不足处产生 null, 由 build_dataset 统一剔除; 不会向数据集泄漏 NaN

精简策略 (2026-09, 依据 JOINT LGB gain 重要性):
- 白名单只保留 top33 高贡献特征 (覆盖 ~95% 贡献), 移除低贡献/零贡献特征
  (consec_up/consec_dn/tbr_z_30/body_pos_60/ngreen_10/up_wick/body_ratio/lo_wick/
   gap/rvol_ratio_60_5/cvd_dir_30/is_us/is_eu/z_10/tbr_hi_60/buyvol_strength_30/
   lr_5/pos_60/up_body_ratio_30/lr_30 等, gain<=0.4%)
- 新增 8 个 regime 特征: 针对 BTC 2026-08 阴跌坏月 (低位做多失效), 刻画
  "阴跌持续度/加速/反弹失败/创新低频率/状态标志", 全部只用 t 及以前滚动信息
- 特征列从 ~250 降到 38, 大幅降低内存占用
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
    DT = pl.from_epoch(pl.col("ts"), time_unit="s")

    lr = (C / C.shift(1)).log()

    e = {}
    # ---- 收益率 / 动量 ----
    e["lr_15"] = (C / C.shift(15)).log()
    e["lr_120"] = (C / C.shift(120)).log()
    e["lr_240"] = (C / C.shift(240)).log()
    e["mom_60"] = (C / C.shift(60) - 1) * 100

    # ---- 已实现波动率 (价格衍生, 非ATR) ----
    e["rvol_30"] = lr.rolling_std(30, ddof=1) * 100
    e["rvol_60"] = lr.rolling_std(60, ddof=1) * 100
    e["rvol_z_60"] = (e["rvol_60"] - e["rvol_60"].rolling_mean(240)) / (e["rvol_60"].rolling_std(240, ddof=1) + EPS)
    e["rvol_dir"] = (e["rvol_30"] - e["rvol_60"]) / (e["rvol_60"] + EPS)

    # ---- 趋势 / Z分数 ----
    for w in [30, 60, 120]:
        mu = C.rolling_mean(w)
        sd = C.rolling_std(w, ddof=1)
        e[f"z_{w}"] = (C - mu) / (sd + EPS)

    # ---- 区间位置 / 支撑压力 ----
    for w in [30, 120, 240]:
        lo = L.rolling_min(w)
        hi = H.rolling_max(w)
        e[f"pos_{w}"] = (C - lo) / (hi - lo + EPS)
    e["dd_240"] = (C / H.rolling_max(240) - 1) * 100
    e["ru_240"] = (C / L.rolling_min(240) - 1) * 100
    # HH/LL 结构
    e["hh_dd_60"] = (C / H.rolling_max(60) - 1) * 100
    e["ll_ru_60"] = (C / L.rolling_min(60) - 1) * 100

    # ---- 尾部/偏度 ----
    e["lr_skew_60"] = lr.rolling_skew(60)
    rng = (H - L) + EPS
    e["max_range_30"] = rng.rolling_max(30) / (rng.rolling_mean(240) + EPS)

    # ---- 主动买卖量 (仅主动买/主动卖) ----
    e["tb_act_60"] = TB.rolling_std(60, ddof=1) / (TB.rolling_mean(60) + EPS)
    e["ts_act_60"] = TS.rolling_std(60, ddof=1) / (TS.rolling_mean(60) + EPS)
    e["cvd_30"] = (TB - TS).rolling_sum(30) / ((TB + TS).rolling_sum(30) + EPS)
    e["cvd_60"] = (TB - TS).rolling_sum(60) / ((TB + TS).rolling_sum(60) + EPS)

    # ---- 动量一致性 ----
    e["mom_align_30_240"] = ((C / C.shift(30) - 1) * (C / C.shift(240) - 1) * 1e8)

    # ---- 日内/时间特征 (UTC) ----
    hour = DT.dt.hour()
    dow = DT.dt.weekday() - 1                    # 与 pandas dayofweek 对齐: 周一=0
    e["hour_sin"] = (hour * 2 * np.pi / 24).sin()
    e["hour_cos"] = (hour * 2 * np.pi / 24).cos()
    e["dow_sin"] = (dow * 2 * np.pi / 7).sin()
    e["dow_cos"] = (dow * 2 * np.pi / 7).cos()
    # 距当日开盘的累计收益 (组内首值广播)
    e["ret_day"] = (C / C.first().over(DT.dt.truncate("1d")) - 1) * 100

    # ================================================================
    # regime 特征 (针对阴跌坏月: BTC 2026-08 低位做多失效)
    # 全部只使用 t 及以前的滚动/滞后信息, 无未来泄漏
    # ================================================================
    dn_flag = (C < C.shift(1)).cast(pl.Int8)     # 当根收阴
    dn_grp = (dn_flag.diff().fill_null(0).ne(0).cast(pl.Int32).cum_sum())
    dn_run = dn_flag.cum_count().over(dn_grp).cast(pl.Float32)   # 当前连跌长度
    e["dn_run_len"] = (dn_flag.cast(pl.Float32) * dn_run / 50).clip(0, 1)
    e["dn_run_max_240"] = (dn_run.rolling_max(240) / 50).clip(0, 1)
    # 阴跌净深度: 仅 240 根净跌幅为负时的绝对值
    lr_240 = e["lr_240"]
    e["dn_net_240"] = pl.when(lr_240 < 0).then(-lr_240).otherwise(0.0)
    # 阴跌加速: 近期 120 根 vs 前期 120 根 (2*lr_120 - lr_240 > 0 表示近期更弱)
    e["dn_accel_240"] = lr_240 - 2 * e["lr_120"]
    # 创新低频率: 120 根内跌破 120 根前低的占比
    low_break = (C < L.rolling_min(120).shift(1)).cast(pl.Float32)
    e["low_break_cnt_120"] = low_break.rolling_mean(120)
    # 反弹失败: 最近 10 根内出现过 5 根阳线主导窗口, 且当前创新低
    green5 = (C > O).cast(pl.Float32).rolling_mean(5) > 0.6
    bounce_fail = (green5.rolling_max(10) & low_break.cast(pl.Boolean)).cast(pl.Float32)
    e["bounce_fail_240"] = (bounce_fail.rolling_sum(240) / 20).clip(0, 1)
    # 阴跌+低位状态标志: 240根净跌且位置低 (超卖陷阱区)
    e["regime_dn_bear"] = ((lr_240 < 0) & (e["pos_120"] < 0.3)).cast(pl.Float32)
    # 阴跌中波动扩张比: 阴跌时 rvol_60 相对 rvol_30 放大 (恐慌加剧)
    e["dn_rvol_ratio"] = ((lr_240 < 0) * (e["rvol_60"] / (e["rvol_30"] + EPS))).clip(0, 3)

    # ================================================================
    # 多周期 / 高周期K形态 (针对"任意时刻实时聚合高周期K")
    # 在时刻 t, 用 1 分钟数据实时聚合成"截至 t 的最后一根高周期K"
    # (未完成K, 收盘价=C[t]), 提取其K线形态与多周期共振。
    # 全部 rolling 递推、从当前价 C[t] 出发, 无简单 resample 的
    # "前一根完整K" 滞后, 无未来泄漏。
    # ================================================================
    # -- 未完成 15 分K: 当前 15 分钟块首开盘 O0, 实体/振幅/进度 --
    O0_15 = O.first().over(DT.dt.truncate("15m"))      # 块内首根开盘
    H15 = H.rolling_max(15); L15 = L.rolling_min(15)
    e["ht15_body"] = ((C - O0_15) / (H15 - L15 + EPS)).clip(-1.5, 1.5)
    e["ht15_range"] = ((H15 - L15) / (L15 + EPS)) * 1e4  # bps
    e["ht15_prog"] = (DT.dt.minute().mod(15) + 1) / 15.0  # 当前15分K进度
    # -- 未完成 5 分K --
    O0_5 = O.first().over(DT.dt.truncate("5m"))
    H5 = H.rolling_max(5); L5 = L.rolling_min(5)
    e["ht5_body"] = ((C - O0_5) / (H5 - L5 + EPS)).clip(-1.5, 1.5)
    # -- 高周期突破: C 是否站上/跌破前一根高周期K的区间 --
    e["ht15_break"] = pl.when(C > H15.shift(1)).then(1.0).otherwise(
        pl.when(C < L15.shift(1)).then(-1.0).otherwise(0.0))
    e["ht60_break"] = pl.when(C > H.rolling_max(60).shift(1)).then(1.0).otherwise(
        pl.when(C < L.rolling_min(60).shift(1)).then(-1.0).otherwise(0.0))
    # -- 多周期方向共振: 1/5/15/30 分涨跌同号的比例 --
    def _sgn(w):
        return pl.when(C > C.shift(w)).then(1.0).otherwise(-1.0)
    e["ht_align_1_30"] = (_sgn(1) + _sgn(5) + _sgn(15) + _sgn(30)) / 4.0
    # -- 高周期K实体方向 vs 多周期: 15分K实体占 30分K区间 的归一位置 --
    H30 = H.rolling_max(30); L30 = L.rolling_min(30)
    e["ht15_in_ht30"] = ((C - L30) / (H30 - L30 + EPS)).clip(0, 1)
    # -- 近5分动量占比: 最近5分净涨跌占 30分 涨跌的份额(短周期接力) --
    lr_5 = (C / C.shift(5)).log()
    e["ht5_share_30"] = (lr_5 / ((C / C.shift(30)).log() + EPS)).clip(-5, 5)

    out = df.select([expr.alias(name) for name, expr in e.items()])
    return out.cast(pl.Float32)


def build_label(c, horizon):
    """label: 1 若 close[t+horizon] > close[t] (预测未来horizon根后涨), 否则 0。"""
    return (c.shift(-horizon) > c).cast(pl.Int8)


def build_ret_future(c, horizon):
    """未来horizon根的真实对数收益(仅用于分析/确认阈值, 不是特征)。"""
    return (c.shift(-horizon) / c).log().cast(pl.Float32)
