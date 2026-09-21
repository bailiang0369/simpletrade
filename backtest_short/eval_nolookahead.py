#!/usr/bin/env python3
"""无前视评估 (事件合约 H=15): 支持 LGB/XGB/CAT 多族集成 + 滚动分位数阈值。

严格无前视: 第 d 天阈值 τ 仅由 <d 天的历史置信度分布算出, 当天第一分钟即可用。
方向: conf>=τ 触发, P>=0.5 做多 / P<0.5 做空。
"""
import os, sys
sys.path.insert(0, "/workspace")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
FAMS = ["lgb", "xgb", "cat"]


def get_models(sym, split, fams=FAMS, seeds=SEEDS):
    """返回 (list_of_arrays, list_of_names)。缺失文件跳过。"""
    arrs, names = [], []
    for fam in fams:
        for s in seeds:
            fp = f"{config.DS_DIR}/SHORT_{sym}_h15_{fam}_seed{s}_{split}_P.npy"
            if os.path.exists(fp):
                arrs.append(np.load(fp).astype(np.float64))
                names.append(f"{fam}{s}")
    return arrs, names


def rank_ens(arrays):
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def eval_nolookahead(sym, split, arrays, q, win_days=30, cold=30):
    ctx = AssetContext(sym, horizon=15, ds_name=f"ds_{sym}_h15")
    p = rank_ens(arrays)
    y = ctx.y(split)
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    day_of = sec // 86400
    days = np.unique(day_of)
    n = len(p)
    sel = np.zeros(n, dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di - win_days):di]
        if len(prior) < cold:
            continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0:
                continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True
    acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / len(days)
    return acc * 100, tpd, int(sel.sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="lgb,xgb,cat")
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--cold", type=int, default=30)
    a = ap.parse_args()
    fams = a.fams.split(",")

    print(f"无前视 | 集成={a.fams} | win={a.win} cold={a.cold} | 多空独立, 阈值仅用历史")
    for split in ["test", "meta_val"]:
        print(f"\n===== {split} =====")
        for sym in ["ETH", "BTC"]:
            arrs, names = get_models(sym, split, fams)
            ctx = AssetContext(sym, horizon=15, ds_name=f"ds_{sym}_h15")
            nd = len(np.unique(ctx.times(split).astype("datetime64[s]").astype(np.int64) // 86400))
            print(f"\n--- {sym} ({len(arrs)}模型, {nd}天) ---")
            for q in [98, 99, 99.2, 99.5, 99.8]:
                acc, tpd, n = eval_nolookahead(sym, split, arrs, q, a.win, a.cold)
                ok = " ⭐" if acc >= 65 and tpd >= 14 else ("・" if acc >= 65 else "")
                print(f"  q={q:<5}: acc={acc:5.2f}%  tpd={tpd:5.1f}  n={n:<5}{ok}")
    print()


if __name__ == "__main__":
    main()