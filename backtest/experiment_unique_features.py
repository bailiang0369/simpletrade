#!/usr/bin/env python3
"""单币种 (BTC, H=30) 独特特征实验:
测试以下 4 组全新设计的高微观表达特征:
1. Price-VWAP 偏离度 Z-score: (Close - VWAP_w) / rolling_std(w)
2. 量价背离度 (Price-CVD Divergence): Z(Price Return_w) - Z(CVD_w)
3. 影线不对称衰竭度 (Wick Asymmetry): (Upper Wick - Lower Wick) / (High - Low + EPS)
4. 波动率压缩比率 (Volatility Compression Ratio): rvol_15 / (rvol_120 + EPS)
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
EPS = 1e-12

def compute_unique_features(df_raw):
    """计算全新设计的 4 组独创微观结构特征 (Polars 版)。"""
    C = pl.col("close")
    O = pl.col("open")
    H = pl.col("high")
    L = pl.col("low")
    TB = pl.col("buy_vol")
    TS = pl.col("sell_vol")
    V_tot = TB + TS + EPS

    e = {}

    # 1. Price-VWAP 偏离度 Z-score
    # VWAP_w = sum(Close * V_tot) / sum(V_tot)
    for w in [30, 60, 120]:
        vwap_w = (C * V_tot).rolling_sum(w) / (V_tot.rolling_sum(w) + EPS)
        std_w = C.rolling_std(w, ddof=1) + EPS
        e[f"vwap_z_{w}"] = (C - vwap_w) / std_w

    # 2. 量价背离度 (Price-CVD Divergence)
    # Z(Price_lr_w) - Z(CVD_w)
    for w in [30, 60, 120]:
        lr_w = (C / C.shift(w)).log()
        cvd_w = (TB - TS).rolling_sum(w) / (V_tot.rolling_sum(w) + EPS)

        z_p = (lr_w - lr_w.rolling_mean(240)) / (lr_w.rolling_std(240, ddof=1) + EPS)
        z_c = (cvd_w - cvd_w.rolling_mean(240)) / (cvd_w.rolling_std(240, ddof=1) + EPS)
        e[f"price_cvd_div_{w}"] = z_p - z_c

    # 3. 影线不对称衰竭度 (Wick Asymmetry Ratio)
    # upper_wick = High - max(Open, Close), lower_wick = min(Open, Close) - Low
    up_wick = H - pl.max_horizontal([O, C])
    lo_wick = pl.min_horizontal([O, C]) - L
    rng = (H - L) + EPS
    wick_asym = (up_wick - lo_wick) / rng
    for w in [15, 30, 60]:
        e[f"wick_asym_{w}"] = wick_asym.rolling_mean(w)

    # 4. 波动率压缩比率 (Volatility Compression Ratio)
    lr = (C / C.shift(1)).log()
    rvol_15 = lr.rolling_std(15, ddof=1)
    rvol_60 = lr.rolling_std(60, ddof=1)
    rvol_240 = lr.rolling_std(240, ddof=1)
    e["vol_compress_15_120"] = rvol_15 / (rvol_60 + EPS)
    e["vol_compress_60_240"] = rvol_60 / (rvol_240 + EPS)

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
    print(f"  单币种单周期测试: {symbol} (H=30) 独创特征组 A/B 测试")
    print(f"=========================================================\n")

    # 1. 载入标的数据与基础特征
    ctx = AssetContext(symbol, horizon=30)
    pf = pq.ParquetFile(f"{config.DS_DIR}/ds_{symbol}.parquet")
    base_cols = [c for c in pf.schema_arrow.names if c not in ("label", "soft_label", "ret_future", "ts")]

    # 2. 计算独创特征组
    raw_df = pl.read_parquet(f"{config.DS_DIR}/raw_{symbol}.parquet").sort("ts")
    uniq_df = compute_unique_features(raw_df)
    uniq_feats = list(uniq_df.columns)
    print(f"[独创特征] 构建特征 {len(uniq_feats)} 列: {uniq_feats}")

    # 对齐到 ds 行
    ri = ctx.ds_to_raw.astype(int)
    X_uniq = uniq_df.to_numpy()[ri]

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
    # 实验 A: 基线特征模型
    # -------------------------------------------------------------
    print("\n--- [实验 A] BTC 基线模型 (不含独创特征) ---")
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
    # 实验 B: 基线特征 + 独创特征组
    # -------------------------------------------------------------
    print("\n--- [实验 B] BTC 增加独创特征组模型 ---")
    X_tr_B = np.column_stack([X_tr_A, X_uniq[trm]])
    X_es_B = np.column_stack([X_es_A, X_uniq[esm]])
    X_mv_B = np.column_stack([X_mv_A, X_uniq[mvm]])
    X_te_B = np.column_stack([X_te_A, X_uniq[tem]])

    P_mv_B = np.zeros((len(BAGGED_SEEDS), len(X_mv_B)), dtype=np.float32)
    P_te_B = np.zeros((len(BAGGED_SEEDS), len(X_te_B)), dtype=np.float32)

    models_B = []
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
        models_B.append(m)

    p_mv_B = P_mv_B.mean(axis=0)
    p_te_B = P_te_B.mean(axis=0)
    acc_B, min_B, bad_B, n_B = evaluate_r2(p_mv_B, p_te_B, y_te, sec_te)
    print(f"==> 实验 B (加独创特征组) 结果: Total Acc={acc_B:.4f}, Min Month={min_B:.4f}, Bad Months={bad_B}, Trades={n_B}")

    # 特征贡献度分析 (Gain Importance)
    all_feature_names = base_cols + uniq_feats
    importances = np.zeros(len(all_feature_names))
    for m in models_B:
        importances += m.feature_importance(importance_type="gain")
    importances /= len(models_B)
    top_indices = np.argsort(-importances)[:15]

    print("\n[Top 15 最关键特征贡献度 (Gain)]")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {all_feature_names[idx]:<25}: {importances[idx]:.2f}")

    diff = acc_B - acc_A
    print("\n=========================================================")
    print(f"  单币种 BTC (H=30) 独创特征增量对比:")
    print(f"  基线 A 准确率  : {acc_A:.4f}")
    print(f"  实验 B 准确率  : {acc_B:.4f}")
    print(f"  准确率边际变化: {diff:+.4f} ({diff*100:+.2f} pp)")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
