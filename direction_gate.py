#!/usr/bin/env python3
"""方向分离门控: 做多/做空各维护近N天(当天之前)已结算命中率,
<THR 则当天掐掉该方向。全程无泄露。结算用"实际下的单"(含门控)。
高效: 预分配 hist 数组, 逐样本/逐日 numpy 向量化, 无 vstack/list 热点。
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
SAMP = 1440
MAXH = 1000 * SAMP


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def summarize(sel, pred, y, mts):
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0, 0.0, 0.0, {})
    ps, ys, ms = pred[sel], y[sel], mts[sel]
    acc = float((ps == ys).mean())
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in np.unique(ms)}
    min_a = min(acc_m.values())
    nbad = sum(1 for v in acc_m.values() if v < 0.55)
    long_n = int((ps == 1).sum()); short_n = int((ps == 0).sum())
    return (acc, min_a, nbad, len(ps), long_n, short_n, acc_m)


def run_gate(conf, pred, y, sec, seed_conf, pctl, N, THR):
    """返回 keep mask。day级循环; 每日先定R2阈值, 再用方向历史门控; 结算实际单。"""
    day = sec // 86400
    days = np.unique(day)
    # hist 使用数组缓冲
    hist = np.empty(MAXH, dtype=np.float64)
    hist[:len(seed_conf)] = seed_conf
    hptr = len(seed_conf)
    # 方向历史: 定长每日聚合数组
    hday = np.zeros(len(days), dtype=np.float64)
    hLok = np.zeros(len(days), dtype=np.float64); hLn = np.zeros(len(days), dtype=np.float64)
    hRok = np.zeros(len(days), dtype=np.float64); hRn = np.zeros(len(days), dtype=np.float64)
    nhist = 0
    keep = np.zeros(len(sec), bool)
    dpos = {d: i for i, d in enumerate(days)}
    for k, dd in enumerate(days):
        md = day == dd
        tau = np.percentile(hist[:hptr], pctl)
        # R2 原始
        orig = md & (conf >= tau)
        day_sel = orig.copy()
        # 方向门控 (近N天, 不含当天)
        if nhist > 0:
            wl = (dd - hday[:nhist]) <= N
            for direc, o_ok, o_n in ((1, hLok[:nhist], hLn[:nhist]), (0, hRok[:nhist], hRn[:nhist])):
                ntot = o_n[wl].sum()
                if ntot >= 1 and (o_ok[wl].sum() / ntot) < THR:
                    day_sel &= ~(md & (pred == direc))
        keep[day_sel] = True
        # 结算实际单
        s = np.where(day_sel)[0]
        hday[nhist] = dd
        hLok[nhist] = int(((pred[s] == 1) & (y[s] == 1)).sum()); hLn[nhist] = int((pred[s] == 1).sum())
        hRok[nhist] = int(((pred[s] == 0) & (y[s] == 0)).sum()); hRn[nhist] = int((pred[s] == 0).sum())
        nhist += 1
        # 结算当天conf加入hist
        c = conf[md]
        hist[hptr:hptr + len(c)] = c
        hptr += len(c)
    return keep


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
        seed = mv["conf"][-180 * SAMP:]
        print(f"\n===== {symbol} 方向分离门控 P99 (meta_val) =====", flush=True)
        best = None
        for N in (3, 7, 14, 30):
            for THR in (0.52, 0.55, 0.58):
                keep = run_gate(mv["conf"], mv["pred"], mv["y"], mv["sec"], seed, 99.0, N, THR)
                sel = np.where(keep)[0]
                acc, min_a, nbad, nn, ln, sn, _ = summarize(sel, mv["pred"], mv["y"], mv["mts"])
                print(f"  N={N:>2} THR={THR:.2f}: acc={acc:.4f} min={min_a:.4f} bad={nbad} n={nn} L={ln}/S={sn}", flush=True)
                cand = (nbad == 0, min_a, acc)
                if best is None or cand > best[0]:
                    best = (cand, N, THR)
        (_, _, _), Nb, THRb = best
        print(f"  >> meta_val 最优: N={Nb} THR={THRb:.2f}", flush=True)
        keep = run_gate(te["conf"], te["pred"], te["y"], te["sec"], seed, 99.0, Nb, THRb)
        sel = np.where(keep)[0]
        acc, min_a, nbad, nn, ln, sn, acc_m = summarize(sel, te["pred"], te["y"], te["mts"])
        print(f"  [test] N={Nb} THR={THRb:.2f}: acc={acc:.4f} min={min_a:.4f} bad={nbad} n={nn} L={ln}/S={sn}", flush=True)
        print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()}, flush=True)
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()