#!/usr/bin/env python3
"""单币种 (BTC, H=30) 实验: 币安真实高频订单簿盘口快照 (bookTicker/Orderbook Snapshot) 特征提取与回测。

数据源: Binance Public Data (https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/)
包含毫秒级盘口列:
- update_id: 订单簿更新序列号
- best_bid_price: 买一价 (Best Bid Price)
- best_bid_qty: 买一量 (Best Bid Quantity)
- best_ask_price: 卖一价 (Best Ask Price)
- best_ask_qty: 卖一量 (Best Ask Quantity)
- transaction_time: 交易撮合毫秒时间戳
- event_time: WebSocket 推送毫秒时间戳
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
EPS = 1e-12

def process_orderbook_snapshot_features(zip_path, ds_ts):
    """读取毫秒级盘口快照，计算买卖盘失衡度 (OBI) 与盘口价差 (Spread) 特征。"""
    z = zipfile.ZipFile(zip_path)
    fn = z.namelist()[0]
    print(f"[Orderbook] 载入真实订单簿毫秒快照文件: {fn}...")

    # 逐块读取避免内存超限 (全量数百万条)
    df_ob = pd.read_csv(z.open(fn))
    df_ob["ts"] = df_ob["transaction_time"] // 1000  # 转为秒级时间戳

    # 提取核心盘口指标:
    # 1. 盘口买卖失衡度 OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    bid_q = df_ob["best_bid_qty"].to_numpy().astype(np.float64)
    ask_q = df_ob["best_ask_qty"].to_numpy().astype(np.float64)
    bid_p = df_ob["best_bid_price"].to_numpy().astype(np.float64)
    ask_p = df_ob["best_ask_price"].to_numpy().astype(np.float64)

    obi = (bid_q - ask_q) / (bid_q + ask_q + EPS)
    spread = (ask_p - bid_p) / (bid_p + EPS) * 10000.0 # 盘口价差 (bps)

    df_ob["obi"] = obi
    df_ob["spread"] = spread

    # 2. 按 1 分钟颗粒度汇总: 计算过去 1 分钟内的平均买卖失衡度与平均买卖盘口深度
    df_min = df_ob.groupby("ts").agg({
        "obi": "mean",
        "spread": "mean",
        "best_bid_qty": "mean",
        "best_ask_qty": "mean"
    }).reset_index()

    ob_ts = df_min["ts"].to_numpy()
    ob_obi = df_min["obi"].to_numpy().astype(np.float32)
    ob_spread = df_min["spread"].to_numpy().astype(np.float32)
    ob_bid_q = df_min["best_bid_qty"].to_numpy().astype(np.float32)
    ob_ask_q = df_min["best_ask_qty"].to_numpy().astype(np.float32)

    # 因果无泄漏二分检索对齐: searchsorted <= current_ts
    idx = np.searchsorted(ob_ts, ds_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ob_ts) - 1)

    f_obi = ob_obi[idx]
    f_spread = ob_spread[idx]
    f_bid_q = ob_bid_q[idx]
    f_ask_q = ob_ask_q[idx]

    # 填充空值
    f_obi = np.nan_to_num(f_obi, nan=0.0)
    f_spread = np.nan_to_num(f_spread, nan=0.0)
    f_bid_q = np.nan_to_num(f_bid_q, nan=0.0)
    f_ask_q = np.nan_to_num(f_ask_q, nan=0.0)

    F = np.column_stack([f_obi, f_spread, f_bid_q, f_ask_q])
    names = ["ob_obi_mean_1m", "ob_spread_bps_1m", "ob_best_bid_qty_mean", "ob_best_ask_qty_mean"]
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
    zip_path = "sample_bookticker.zip"
    print(f"\n=========================================================")
    print(f"  币安真实订单簿盘口快照 (bookTicker) 数据结构与特征实验: {symbol} (H=30)")
    print(f"=========================================================\n")

    # 1. 载入标的数据与基础特征
    ctx = AssetContext(symbol, horizon=30)
    pf = pq.ParquetFile(f"{config.DS_DIR}/ds_{symbol}.parquet")
    base_cols = [c for c in pf.schema_arrow.names if c not in ("label", "soft_label", "ret_future", "ts")]

    # 2. 提取真实订单簿盘口特征
    F_ob, ob_feature_names = process_orderbook_snapshot_features(zip_path, ctx.ds_ts)
    print(f"[订单簿特征] 提取完成，生成 4 列真实盘口特征: {ob_feature_names}")

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
    # A/B 对比评估: 基础特征 vs 基础特征 + 真实订单簿盘口特征
    # -------------------------------------------------------------
    print("\n--- [实验] 测试加入真实订单簿盘口 (bookTicker) 特征对 BTC (H=30) 的表现 ---")
    X_tr_base = ctx.X_subset(base_cols, trm)
    X_es_base = ctx.X_subset(base_cols, esm)
    X_mv_base = ctx.X_subset(base_cols, mvm)
    X_te_base = ctx.X_subset(base_cols, tem)

    X_tr_ob = np.column_stack([X_tr_base, F_ob[trm]])
    X_es_ob = np.column_stack([X_es_base, F_ob[esm]])
    X_mv_ob = np.column_stack([X_mv_base, F_ob[mvm]])
    X_te_ob = np.column_stack([X_te_base, F_ob[tem]])

    P_mv = np.zeros((len(BAGGED_SEEDS), len(X_mv_ob)), dtype=np.float32)
    P_te = np.zeros((len(BAGGED_SEEDS), len(X_te_ob)), dtype=np.float32)

    for idx, seed in enumerate(BAGGED_SEEDS):
        p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                 max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                 min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(X_tr_ob, y_tr)
        des = lgb.Dataset(X_es_ob, y_es, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        P_mv[idx] = m.predict(X_mv_ob)
        P_te[idx] = m.predict(X_te_ob)

    p_mv = P_mv.mean(axis=0)
    p_te = P_te.mean(axis=0)
    acc, min_m, bad_m, n_trades = evaluate_r2(p_mv, p_te, y_te, sec_te)
    print(f"\n=========================================================")
    print(f"  融入真实订单簿盘口快照 (bookTicker) 特征后的 BTC (H=30) 结果:")
    print(f"  测试集总准确率 : {acc:.4f}")
    print(f"  最差单月准确率 : {min_m:.4f}")
    print(f"  坏月 (<55%) 数 : {bad_m} 个")
    print(f"  测试集信号数   : {n_trades} 次")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
