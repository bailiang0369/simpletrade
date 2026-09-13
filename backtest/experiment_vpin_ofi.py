#!/usr/bin/env python3
"""单币种 (BTC) 单周期 (H=30) 测试: VPIN 与 多窗口 OFI 代理特征实验。

不影响 ETH 或其他周期，仅在 BTC (H=30) 上进行单币种 A/B 对比测试。
- OFI (Order Flow Imbalance): (buy_vol - sell_vol) 的多窗口 (15, 30, 60, 120, 240) 归一化失衡额与变化率
- VPIN (Volume-Synchronized Probability of Toxicity): 基于等成交量/时间桶的订单毒性概率代理
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import lightgbm as lgb
import pyarrow.parquet as pq

import config
from data_store import AssetContext

P99 = 99.0
WIN_DAYS = 60
SAMP = 1440
BAGGED_SEEDS = [42, 49, 56, 63, 70]

def compute_vpin_ofi_features(df_raw):
    """计算单个币种的 VPIN 与 OFI 代理特征 (Polars 版)。"""
    TB = pl.col("buy_vol")
    TS = pl.col("sell_vol")
    V_tot = TB + TS + 1e-12
    imb = TB - TS

    e = {}
    # 1. 多窗口 Bar-level OFI (Order Flow Imbalance)
    for w in [15, 30, 60, 120, 240]:
        e[f"ofi_{w}"] = imb.rolling_sum(w) / V_tot.rolling_sum(w)
        # OFI 加速度/变化率
        e[f"ofi_diff_{w}"] = e[f"ofi_{w}"] - e[f"ofi_{w}"].shift(w)

    # 2. VPIN 代理 (Volume-Synchronized Probability of Toxicity)
    # VPIN = sum(|buy_vol - sell_vol|) / (N * V_bucket)
    abs_imb = imb.abs()
    for w in [30, 60, 120, 240]:
        e[f"vpin_{w}"] = abs_imb.rolling_sum(w) / V_tot.rolling_sum(w)

    out = df_raw.select([expr.alias(name) for name, expr in e.items()])
    return out.cast(pl.Float32)

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
    print(f"  单币种单周期测试: {symbol} (H=30) VPIN & OFI 特征实验")
    print(f"=========================================================\n")

    # 1. 载入原始特征矩阵与标的数据
    ctx = AssetContext(symbol, horizon=30)

    # 获取现有所有基础特征列 (基线)
    pf = pq.ParquetFile(f"{config.DS_DIR}/ds_{symbol}.parquet")
    base_cols = [c for c in pf.schema_arrow.names if c not in ("label", "soft_label", "ret_future", "ts")]

    # 2. 构建 VPIN 与 OFI 新特征
    raw_df = pl.read_parquet(f"{config.DS_DIR}/raw_{symbol}.parquet").sort("ts")
    vpin_ofi_df = compute_vpin_ofi_features(raw_df)
    vpin_ofi_feats = list(vpin_ofi_df.columns)
    print(f"[VPIN/OFI] 构建特征 {len(vpin_ofi_feats)} 列: {vpin_ofi_feats}")

    # 对齐到 ds 行
    ds_ts = ctx.ds_ts
    raw_ts = ctx.raw_ts
    ri = ctx.ds_to_raw.astype(int)
    X_vpin_ofi = vpin_ofi_df.to_numpy()[ri] # (N, len_feats)

    # 准备数据集 Mask
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    mvm = ctx.split_rows["meta_val"]
    tem = ctx.split_rows["test"]

    y_tr = ctx.label[trm]
    y_es = ctx.label[esm]
    y_mv = ctx.label[mvm]
    y_te = ctx.label[tem]
    sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    # -------------------------------------------------------------
    # 实验 A: 基线特征模型 (不含 VPIN/OFI)
    # -------------------------------------------------------------
    print("\n--- [实验 A] 运行 BTC 基线模型 (不含 VPIN/OFI) ---")
    X_tr_A = ctx.X_subset(base_cols, trm)
    X_es_A = ctx.X_subset(base_cols, esm)
    X_mv_A = ctx.X_subset(base_cols, mvm)
    X_te_A = ctx.X_subset(base_cols, tem)

    P_mv_A = np.zeros((len(BAGGED_SEEDS), len(X_mv_A)), dtype=np.float32)
    P_te_A = np.zeros((len(BAGGED_SEEDS), len(X_te_A)), dtype=np.float32)

    for idx, seed in enumerate(BAGGED_SEEDS):
        p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                 max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                 min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(X_tr_A, y_tr)
        des = lgb.Dataset(X_es_A, y_es, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        P_mv_A[idx] = m.predict(X_mv_A)
        P_te_A[idx] = m.predict(X_te_A)

    p_mv_A = P_mv_A.mean(axis=0)
    p_te_A = P_te_A.mean(axis=0)
    acc_A, min_A, bad_A, n_A = evaluate_r2(p_mv_A, p_te_A, y_te, sec_te)
    print(f"==> 基线 A 结果: Total Acc={acc_A:.4f}, Min Month={min_A:.4f}, Bad Months={bad_A}, Trades={n_A}")

    # -------------------------------------------------------------
    # 实验 B: 基线特征 + VPIN/OFI 特征
    # -------------------------------------------------------------
    print("\n--- [实验 B] 运行 BTC 增加 VPIN/OFI 代理特征模型 ---")
    X_tr_B = np.column_stack([X_tr_A, X_vpin_ofi[trm]])
    X_es_B = np.column_stack([X_es_A, X_vpin_ofi[esm]])
    X_mv_B = np.column_stack([X_mv_A, X_vpin_ofi[mvm]])
    X_te_B = np.column_stack([X_te_A, X_vpin_ofi[tem]])

    P_mv_B = np.zeros((len(BAGGED_SEEDS), len(X_mv_B)), dtype=np.float32)
    P_te_B = np.zeros((len(BAGGED_SEEDS), len(X_te_B)), dtype=np.float32)

    for idx, seed in enumerate(BAGGED_SEEDS):
        p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                 max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                 min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(X_tr_B, y_tr)
        des = lgb.Dataset(X_es_B, y_es, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        P_mv_B[idx] = m.predict(X_mv_B)
        P_te_B[idx] = m.predict(X_te_B)

    p_mv_B = P_mv_B.mean(axis=0)
    p_te_B = P_te_B.mean(axis=0)
    acc_B, min_B, bad_B, n_B = evaluate_r2(p_mv_B, p_te_B, y_te, sec_te)
    print(f"==> 实验 B (加 VPIN/OFI) 结果: Total Acc={acc_B:.4f}, Min Month={min_B:.4f}, Bad Months={bad_B}, Trades={n_B}")

    diff = acc_B - acc_A
    print("\n=========================================================")
    print(f"  单币种 BTC (H=30) VPIN/OFI 增量对比:")
    print(f"  基线 A 准确率  : {acc_A:.4f}")
    print(f"  实验 B 准确率  : {acc_B:.4f}")
    print(f"  准确率边际变化: {diff:+.4f} ({diff*100:+.2f} pp)")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
