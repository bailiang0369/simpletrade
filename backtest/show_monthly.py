#!/usr/bin/env python3
"""以 R2(每日滚动阈值, 盘前定当天阈值, 无泄露) 列出 ETH/BTC test 每月 信号量+准确率。"""
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
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        data = {}
        for split in ("meta_val", "test"):
            p = load_fused(symbol, split)
            y = ctx.y(split)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            data[split] = dict(conf=np.maximum(p, 1 - p), pred=(p >= 0.5).astype(np.int8),
                               y=y, mts=mts, sec=sec)
            del p
            gc.collect()
        mv, te = data["meta_val"], data["test"]
        # 盘前历史 = meta_val 尾部 90 天
        hist = list(mv["conf"][-WIN_DAYS * SAMP:])
        day = te["sec"] // 86400
        days = np.unique(day)
        keep = np.zeros(len(te["sec"]), bool)
        for dd in days:
            md = day == dd
            tau = np.percentile(np.asarray(hist), P99)
            keep[md & (te["conf"] >= tau)] = True
            hist.extend(te["conf"][md])
            if len(hist) > WIN_DAYS * SAMP * 2:
                del hist[:len(hist) - WIN_DAYS * SAMP * 2]
        sel = np.where(keep)[0]
        ps, ys, ms = te["pred"][sel], te["y"][sel], te["mts"][sel]
        n_days = np.unique(day).size
        print(f"\n===== {symbol}  R2 每日滚动阈值 (P99), 总选中 {len(sel)} / 日均 {len(sel)/n_days:.2f} =====")
        print(f"{'月份':<10}{'信号数':>8}{'准确率':>8}{'是否<55':>8}")
        for u in np.unique(ms):
            m = ms == u
            n = int(m.sum())
            acc = float((ps == ys)[m].mean())
            flag = "  <-- 坏月" if acc < 0.55 else ""
            print(f"{str(u)[:7]:<10}{n:>8}{acc:>8.4f}{flag:>8}")
        acc_all = float((ps == ys).mean())
        print(f"{'总 acc':<10}{'':>8}{acc_all:>8.4f}")
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()