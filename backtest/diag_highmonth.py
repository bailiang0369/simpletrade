#!/usr/bin/env python3
"""诊断 BTC 高准月(2025-10/12) vs 坏月(2026-02/05): 方向分布 + 月度真实涨跌。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
P99 = 99.0
WIN_DAYS = 90
SAMP = 1440


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def main():
    symbol = "BTC"
    ctx = AssetContext(symbol, horizon=30)
    p = load_fused(symbol, "test")
    y = ctx.y("test")
    sec = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
    mts = sec.astype("datetime64[s]").astype("datetime64[M]")
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)

    mv = load_fused(symbol, "meta_val")
    mv_conf = np.maximum(mv, 1 - mv)
    hist = list(mv_conf[-WIN_DAYS * SAMP:])
    day = sec // 86400
    days = np.unique(day)
    keep = np.zeros(len(sec), bool)
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), P99)
        keep[md & (conf >= tau)] = True
        hist.extend(conf[md])
        if len(hist) > WIN_DAYS * SAMP * 2:
            del hist[:len(hist) - WIN_DAYS * SAMP * 2]
    sel = np.where(keep)[0]
    sel_sec = sec[sel]

    targets = ["2025-10", "2025-12", "2026-02", "2026-05"]
    # 月度盘面方向: 用 raw close, 按月取首末收盘
    raw_ts = ctx.raw_ts
    raw_c = ctx.c
    print(f"\n=== {symbol}: 高准月 vs 坏月 方向分布 & 行情方向 ===")
    for m in targets:
        mm = np.datetime64(m, "M").astype("datetime64[M]")
        month_sel = mts[sel] == mm
        s = sel[month_sel]
        ps = pred[s]; ys = y[s]
        acc = float((ps == ys).mean())
        long_n = int((ps == 1).sum()); short_n = int((ps == 0).sum())
        long_ok = int(((ps == 1) & (ys == 1)).sum())
        short_ok = int(((ps == 0) & (ys == 0)).sum())
        long_acc = long_ok / max(long_n, 1); short_acc = short_ok / max(short_n, 1)
        up_r = int((ys == 1).sum()); dn_r = int((ys == 0).sum())
        # 该月首末 raw close (基于该月第一/最后一个 test 样本 ds_ts 前后最近 raw)
        msec = sel_sec[month_sel]
        t0, t1 = int(msec.min()), int(msec.max())
        i0 = np.searchsorted(raw_ts, t0); i1 = np.searchsorted(raw_ts, t1) - 1
        c0 = raw_c[max(0, i0)]; c1 = raw_c[max(0, min(i1, len(raw_c) - 1))]
        mret = (c1 / c0 - 1) * 100 if c0 > 0 else 0.0
        print(f"\n{m}:  n={len(s)} acc={acc:.4f} | 月份价格 {c0:.0f}->{c1:.0f} ({mret:+.1f}%)")
        print(f"   做多 {long_n} (acc {long_acc:.4f})  做空 {short_n} (acc {short_acc:.4f})")
        print(f"   未来30min 真实: 涨 {up_r} / 跌 {dn_r}")
    del ctx
    gc.collect()


if __name__ == "__main__":
    main()