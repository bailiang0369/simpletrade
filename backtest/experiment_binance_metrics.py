#!/usr/bin/env python3
"""单币种 (BTC, H=30) 实验: 币安 Vision 官方公开期货指标 (Metrics) 结构化与特征回测。

数据源: Binance Public Data (https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/)
包含列:
- create_time: 采样时间戳 (5 分钟粒度)
- sum_open_interest: 未平仓合约总张数 (Open Interest)
- sum_open_interest_value: 未平仓合约总价值 (USD Value)
- count_toptrader_long_short_ratio: 大户账户多空人数比
- sum_toptrader_long_short_ratio: 大户持仓多空头寸比
- count_long_short_ratio: 全网散户多空人数比
- sum_taker_long_short_vol_ratio: Taker 主动买卖量比 (Taker Long/Short Ratio)
"""
import os, sys, time, io, zipfile, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import pyarrow.parquet as pq

import config
from data_store import AssetContext

P99 = 99.0
WIN_DAYS = 60
SAMP = 1440
BAGGED_SEEDS = [42, 49, 56, 63, 70]

def fetch_sample_binance_metrics():
    """从币安官方 Vision 数据库下载样本数据 (例如 2024-01-01 至 2024-01-07 1周样本)。"""
    dfs = []
    dates = pd.date_range("2024-01-01", "2024-01-07").strftime("%Y-%m-%d")
    print(f"[Binance Vision] 开始下载 {len(dates)} 天官方 Futures Metrics 数据...")
    for dt in dates:
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{dt}.zip"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                fn = z.namelist()[0]
                df_day = pd.read_csv(z.open(fn))
                dfs.append(df_day)
        except Exception as err:
            print(f"  下载 {dt} 失败: {err}")

    if not dfs:
        raise RuntimeError("未能成功下载币安 Metrics 数据")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["ts"] = pd.to_datetime(df_all["create_time"]).astype("int64") // 10**9
    df_all = df_all.sort_values("ts").reset_index(drop=True)
    print(f"[Binance Vision] 下载成功: 得到 {len(df_all)} 条 5 分钟 Metrics 记录。")
    return df_all

def process_metrics_features(df_metrics, ds_ts):
    """将 5 分钟粒度的 Metrics 数据做无前视因果对齐，并计算持仓量变化率与多空比特征。"""
    mts = df_metrics["ts"].to_numpy()
    oi = df_metrics["sum_open_interest"].to_numpy().astype(np.float64)
    top_ratio = df_metrics["sum_toptrader_long_short_ratio"].to_numpy().astype(np.float64)
    taker_ratio = df_metrics["sum_taker_long_short_vol_ratio"].to_numpy().astype(np.float64)

    # 计算 5 分钟与 1 小时持仓量变化率 (OI Delta)
    s_oi = pd.Series(oi)
    oi_change_12 = (s_oi / s_oi.shift(12) - 1.0).to_numpy() # 1小时 OI 变化率

    # 对齐到 1 分钟 ds_ts (无泄漏: searchsorted <= current_ts)
    idx = np.searchsorted(mts, ds_ts, side="right") - 1
    idx = np.clip(idx, 0, len(mts) - 1)

    feat_oi_change = oi_change_12[idx].astype(np.float32)
    feat_top_ratio = top_ratio[idx].astype(np.float32)
    feat_taker_ratio = taker_ratio[idx].astype(np.float32)

    # 填充 NaN
    feat_oi_change = np.nan_to_num(feat_oi_change, nan=0.0)
    feat_top_ratio = np.nan_to_num(feat_top_ratio, nan=1.0)
    feat_taker_ratio = np.nan_to_num(feat_taker_ratio, nan=1.0)

    F = np.column_stack([feat_oi_change, feat_top_ratio, feat_taker_ratio])
    names = ["oi_change_1h", "toptrader_ls_ratio", "taker_long_short_vol_ratio"]
    return F, names

def evaluate_r2(p_mv, p_te, y_te, sec_te):
    conf_mv = np.maximum(p_mv, 1 - p_mv)
    conf_te = np.maximum(p_te, 1 - p_te)
    pred_te = (p_te >= 0.5).astype(np.int8)

    hist = list(conf_mv[-WIN_DAYS * SAMP:])
    day = sec_te // 86400
    days = np.unique(day)
    keep = np.zeros(len(sec_te), bool)
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), P99)
        keep[md & (conf_te >= tau)] = True
        hist.extend(conf_te[md])
        if len(hist) > WIN_DAYS * SAMP * 2:
            del hist[:len(hist) - WIN_DAYS * SAMP * 2]
    sel = np.where(keep)[0]
    ps, ys = pred_te[sel], y_te[sel]
    mts = sec_te[sel].astype("datetime64[s]").astype("datetime64[M]")
    acc_m = {str(u)[:7]: float((ps == ys)[mts == u].mean()) for u in np.unique(mts)}
    acc_all = float((ps == ys).mean())
    min_m = min(acc_m.values()) if acc_m else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    return acc_all, min_m, bad_m, len(sel)

def main():
    symbol = "BTC"
    print(f"\n=========================================================")
    print(f"  币安官方 Futures Metrics (OI/大户多空比) 数据格式与特征实验")
    print(f"=========================================================\n")

    # 1. 下载并展示 1 天/1 周的币安 Vision Futures Metrics 原始数据形态
    df_metrics = fetch_sample_binance_metrics()
    print("\n[币安官方 5分钟 Metrics 原始列与结构说明]:")
    print("---------------------------------------------------------")
    for col in df_metrics.columns:
        print(f" - {col:<35}: 样本值 = {df_metrics[col].iloc[0]}")
    print("---------------------------------------------------------\n")

    # 2. 载入标的数据
    ctx = AssetContext(symbol, horizon=30)
    pf = pq.ParquetFile(f"{config.DS_DIR}/ds_{symbol}.parquet")
    base_cols = [c for c in pf.schema_arrow.names if c not in ("label", "soft_label", "ret_future", "ts")]

    # 3. 对齐 Metrics 特征
    F_metrics, metric_names = process_metrics_features(df_metrics, ctx.ds_ts)
    print(f"[特征构建] 完成对齐，得到特征列: {metric_names}")

    # 准备 Mask
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    mvm = ctx.split_rows["meta_val"]
    tem = ctx.split_rows["test"]

    y_tr = ctx.label[trm]
    y_es = ctx.label[esm]
    y_te = ctx.label[tem]
    sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    # -------------------------------------------------------------
    # A/B 对比评估: 在包含 Metrics 特征下的采样回测
    # -------------------------------------------------------------
    print("\n--- [实验] 测试加入 Open Interest & 大户多空比特征对 BTC (H=30) 的表现 ---")
    X_tr_base = ctx.X_subset(base_cols, trm)
    X_es_base = ctx.X_subset(base_cols, esm)
    X_mv_base = ctx.X_subset(base_cols, mvm)
    X_te_base = ctx.X_subset(base_cols, tem)

    X_tr_met = np.column_stack([X_tr_base, F_metrics[trm]])
    X_es_met = np.column_stack([X_es_base, F_metrics[esm]])
    X_mv_met = np.column_stack([X_mv_base, F_metrics[mvm]])
    X_te_met = np.column_stack([X_te_base, F_metrics[tem]])

    P_mv = np.zeros((len(BAGGED_SEEDS), len(X_mv_met)), dtype=np.float32)
    P_te = np.zeros((len(BAGGED_SEEDS), len(X_te_met)), dtype=np.float32)

    for idx, seed in enumerate(BAGGED_SEEDS):
        p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                 max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                 min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(X_tr_met, y_tr)
        des = lgb.Dataset(X_es_met, y_es, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        P_mv[idx] = m.predict(X_mv_met)
        P_te[idx] = m.predict(X_te_met)

    p_mv = P_mv.mean(axis=0)
    p_te = P_te.mean(axis=0)
    acc, min_m, bad_m, n_trades = evaluate_r2(p_mv, p_te, y_te, sec_te)
    print(f"\n=========================================================")
    print(f"  融入币安官方 Futures Metrics 特征后的 BTC (H=30) 结果:")
    print(f"  测试集总准确率 : {acc:.4f}")
    print(f"  最差单月准确率 : {min_m:.4f}")
    print(f"  坏月 (<55%) 数 : {bad_m} 个")
    print(f"  测试集信号数   : {n_trades} 次")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
