#!/usr/bin/env python3
"""分析 pool20 各 family 独立/组合在 ETH 上的表现, 寻找最佳子集集成。"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from itertools import combinations

FAMS = ["lgb", "xgb", "cat"]


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def eval_daily(p, y, sec_arr):
    n = len(p); pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    day = sec_arr // 86400; days = np.unique(day); sm = np.zeros(n, bool)
    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
        sub = np.where(md)[0]
        sm[sub[np.argsort(-conf[sub])[:kd]]] = True
    sel = np.where(sm)[0]
    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq}
    min_k = min(int((mts == u).sum()) for u in uniq)
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    n_days = np.unique(sec_arr // 86400).size
    return (float((pred[sel] == y[sel]).mean()), min_a, min_k, nbad,
            float(sel.size) / n_days, float(sel.size) / n, acc_m)


def main():
    symbol = "ETH"
    ctx = AssetContext(symbol, horizon=30)
    for split in ("meta_val", "test"):
        Ps = {f: np.load(f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy") for f in FAMS}
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        print(f"\n===== {symbol} {split} =====")
        # 单 family
        for f in FAMS:
            P = Ps[f]; n = P.shape[1]
            p = rank_mean(P, n)
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
            print(f"[{f:3s} alone] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad}")
        # 组合 (2个及以上 family)
        for r in (2, 3):
            for combo in combinations(FAMS, r):
                P = np.concatenate([Ps[f] for f in combo], axis=0)
                n = P.shape[1]
                p = rank_mean(P, n)
                acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
                print(f"[{'/'.join(combo)}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f}")
        del Ps; gc.collect()
    print("\n===== ETH test 2025-12 月详情 =====")


if __name__ == "__main__":
    main()
