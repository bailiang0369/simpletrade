#!/usr/bin/env python3
"""ensemble 分歧度门控(不砍方向): 用三个family预测是否同向(consensus)过滤。
坏月亏损样本多为模型间分歧大; 只保留强共识样本, 逐样本过滤, 不砍整方向。
meta_val 定 MIN_VOTE, test 见真章。无泄露。
"""
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


def load_conf_consensus(symbol, split):
    """返回 conf(融合rank), pred(融合), consensus(同向模型数 1..3)。
    每个family P 是 (5seed, N); 先对seed取平均得 (N,) per family, 再合成。"""
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    # 每个 family: (5,N) -> (N,) seed平均
    Pmean = [Ps[i].mean(axis=0) for i in range(3)]
    n = Pmean[0].shape[0]
    P = rank_mean(np.stack(Pmean, axis=0), n)     # (3,N) -> (N,)
    conf = np.maximum(P, 1 - P)
    pred = (P >= 0.5).astype(np.int8)
    # 每 family 的预测方向 (seed均值后阈值)
    dirs = np.stack([(pm >= 0.5).astype(np.int8) for pm in Pmean], axis=0)  # (3,N)
    vote = dirs.sum(axis=0)
    consensus = np.maximum(vote, 3 - vote)
    del Ps, Pmean, dirs, P
    gc.collect()
    return conf, pred, consensus


def summarize(sel, pred, y, mts):
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0, 0.0, {})
    ps, ys, ms = pred[sel], y[sel], mts[sel]
    acc = float((ps == ys).mean())
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in np.unique(ms)}
    min_a = min(acc_m.values())
    nbad = sum(1 for v in acc_m.values() if v < 0.55)
    return (acc, min_a, nbad, len(ps), acc_m)


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
        data = {}
        for split in ("meta_val", "test"):
            conf, pred, cons = load_conf_consensus(symbol, split)
            y = ctx.y(split)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            data[split] = dict(conf=conf, pred=pred, cons=cons, y=y, mts=mts, sec=sec)
            gc.collect()
        mv, te = data["meta_val"], data["test"]
        seed = mv["conf"][-180 * SAMP:]
        mv_mask = r2_mask(mv["conf"], mv["sec"], seed, 99.0)
        te_mask = r2_mask(te["conf"], te["sec"], seed, 99.0)
        print(f"\n===== {symbol} ensemble分歧度门控 =====", flush=True)
        # meta_val 选区 MIN_VOTE in {1,2,3} (1=全量)
        best = None
        for MINV in (1, 2, 3):
            sel = np.where(mv_mask & (mv["cons"] >= MINV))[0]
            acc, min_a, nbad, nn, _ = summarize(sel, mv["pred"], mv["y"], mv["mts"])
            print(f"  mv MINV={MINV}: acc={acc:.4f} min={min_a:.4f} bad={nbad} n={nn}", flush=True)
            cand = (nbad == 0, min_a, acc, nn)
            if best is None or cand > best[0]:
                best = (cand, MINV)
        (_, _, _, _), MINVb = best
        print(f"  >> meta_val 最优 MINV={MINVb}", flush=True)
        # 基线 (MINV=1)
        sel0 = np.where(te_mask)[0]
        b0 = summarize(sel0, te["pred"], te["y"], te["mts"])
        print(f"  [test 基线     ] acc={b0[0]:.4f} min={b0[1]:.4f} bad={b0[2]} n={b0[3]}", flush=True)
        print("    逐月:", {k: round(v,3) for k,v in b0[4].items()}, flush=True)
        # 门控
        selg = np.where(te_mask & (te["cons"] >= MINVb))[0]
        bg = summarize(selg, te["pred"], te["y"], te["mts"])
        print(f"  [test +共识     ] acc={bg[0]:.4f} min={bg[1]:.4f} bad={bg[2]} n={bg[3]}", flush=True)
        print("    逐月:", {k: round(v,3) for k,v in bg[4].items()}, flush=True)
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()