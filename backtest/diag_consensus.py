#!/usr/bin/env python3
"""诊断: 坏月(BTC test 2026-02/05) vs 好月的选中样本中, ensemble分歧(spread)有差异吗?
若坏月选中样本分歧显著更高, 分歧度门控才有信息量; 否则此方向无效。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
SAMP = 1440


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    Pmean = [Ps[i].mean(axis=0) for i in range(3)]
    n = Pmean[0].shape[0]
    P = rank_mean(np.stack(Pmean, axis=0), n)
    conf = np.maximum(P, 1 - P)
    pred = (P >= 0.5).astype(np.int8)
    spread = np.std([pm for pm in Pmean], axis=0)   # 3 family P 的 std
    return conf, pred, spread


def r2_mask(conf, sec, seed_conf, pctl):
    conf = conf.ravel(); sec = sec.ravel()
    day = sec // 86400
    days = np.unique(day)
    hist = list(seed_conf)
    keep = np.zeros(len(sec), bool)
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), pctl)
        keep[md & (conf >= tau)] = True
        hist.extend(conf[md])
        if len(hist) > 400 * SAMP * 2:
            del hist[:len(hist) - 400 * SAMP * 2]
    return keep


def main():
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        conf, pred, spread = load(symbol, "test")
        y = ctx.y("test")
        sec = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
        mts = sec.astype("datetime64[s]").astype("datetime64[M]")
        # meta seed for R2
        mconf, _, _ = load(symbol, "meta_val")
        seed = mconf[-180 * SAMP:]
        mask = r2_mask(conf, sec, seed, 99.0)
        sel = np.where(mask)[0]
        print(f"\n===== {symbol} test 选中样本: 各月 spread 与 acc =====")
        for u in np.unique(mts[sel]):
            m = mts[sel] == u
            s = sel[m]
            acc = float((pred[s] == y[s]).mean())
            sp = spread[s]
            n = len(s)
            mkey = str(u)[:7]
            flag = "  <-- 坏月" if acc < 0.55 else ""
            print(f"{mkey:<10} n={n:>4} acc={acc:.4f}  spread均值={sp.mean():.4f} 中位={np.median(sp):.4f} p75={np.percentile(sp,75):.4f}{flag}")
        # 全体: 坏 vs 好的 spread 分布
        bad = (mts[sel].astype('datetime64[M]') < np.datetime64('2026-01-01'))
        print(f"  [对照组] 2025H2 acc... (略)")
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()