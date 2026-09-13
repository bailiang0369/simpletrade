#!/usr/bin/env python3
"""二阶段 Meta-Learner 融合 (Ridge / Logistic Blending)。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
import config
from data_store import AssetContext

P99 = 99.0
WIN_DAYS = 90
SAMP = 1440

def rank_normalize(P):
    R = np.zeros_like(P, dtype=np.float64)
    n = P.shape[1]
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R

def eval_r2(symbol, p_mv, p_te):
    ctx = AssetContext(symbol, horizon=30)
    y_mv = ctx.y("meta_val")
    y_te = ctx.y("test")
    sec_mv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
    mts_te = sec_te.astype("datetime64[s]").astype("datetime64[M]")

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
    ps, ys, ms = pred_te[sel], y_te[sel], mts_te[sel]
    n_days = np.unique(day).size

    acc_m = {}
    for u in np.unique(ms):
        m = ms == u
        acc_m[str(u)[:7]] = float((ps == ys)[m].mean())
    acc_all = float((ps == ys).mean())
    min_m = min(acc_m.values()) if acc_m else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    return acc_all, min_m, bad_m, len(sel), acc_m

def main():
    print("=== 二阶段 Meta-Learner Stacking 融合实验 ===")
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        y_mv = ctx.y("meta_val")

        # 读取各模型预测矩阵 (5 LGB seeds)
        p_lgb_mv = np.load(f"{config.DS_DIR}/JOINT_{symbol}_lgb_meta_val_P.npy")
        p_lgb_te = np.load(f"{config.DS_DIR}/JOINT_{symbol}_lgb_test_P.npy")

        r_lgb_mv = rank_normalize(p_lgb_mv)
        r_lgb_te = rank_normalize(p_lgb_te)

        # 1. 基线: Equal-Weight Rank Mean
        p_base_mv = r_lgb_mv.mean(axis=0)
        p_base_te = r_lgb_te.mean(axis=0)
        acc_base, min_base, bad_base, n_base, _ = eval_r2(symbol, p_base_mv, p_base_te)
        print(f"\n[{symbol}] 基线 (Equal Rank Mean): Total Acc={acc_base:.4f}, Min Month={min_base:.4f}, Bad Months={bad_base}")

        # 2. Meta-Learner: Ridge Regression 在 meta_val 上对 5 个 seed 拟合权重
        X_meta_tr = r_lgb_mv.T  # (n_samples, 5)
        X_meta_te = r_lgb_te.T

        meta_ridge = Ridge(alpha=100.0)
        meta_ridge.fit(X_meta_tr, y_mv)
        p_ridge_mv = meta_ridge.predict(X_meta_tr)
        p_ridge_te = meta_ridge.predict(X_meta_te)
        acc_ridge, min_ridge, bad_ridge, n_ridge, _ = eval_r2(symbol, p_ridge_mv, p_ridge_te)
        print(f"[{symbol}] Ridge Meta-Learner:   Total Acc={acc_ridge:.4f}, Min Month={min_ridge:.4f}, Bad Months={bad_ridge}")

        # 3. Meta-Learner: Logistic Regression
        meta_lr = LogisticRegression(C=0.01)
        meta_lr.fit(X_meta_tr, y_mv)
        p_lr_mv = meta_lr.predict_proba(X_meta_tr)[:, 1]
        p_lr_te = meta_lr.predict_proba(X_meta_te)[:, 1]
        acc_lr, min_lr, bad_lr, n_lr, _ = eval_r2(symbol, p_lr_mv, p_lr_te)
        print(f"[{symbol}] Logistic Meta-Learner:Total Acc={acc_lr:.4f}, Min Month={min_lr:.4f}, Bad Months={bad_lr}")

if __name__ == "__main__":
    main()
