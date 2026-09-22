#!/usr/bin/env python3
"""无前视评估 GRU/LGB/集成 (事件合约 H=15): 滚动分位数阈值, 多空独立。

用法:
  python eval_gru.py {gru|lgb|lgbgru}
"""
import os, sys
sys.path.insert(0, "/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from data_store import AssetContext

H = 15
SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]


def rank1d(p):
    return np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1)


def no_lookahead(day_of, days, pred, conf, P_conf_signal, q, win=30, cold=30):
    """滚动分位阈值, 只做 pred 规定的方向, conf>=τ。返回 bool mask 和方向(用于acc)."""
    n = len(conf)
    sel = np.zeros(n, dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di - win):di]
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
    return sel


def load_p(sym, split, mode):
    if mode == "gru":
        return np.load(f"{config.DS_DIR}/SHORT_{sym}_h{H}_gru_{split}_P.npy").astype(np.float64), "gru"
    if mode == "lgb":
        P = np.stack([np.load(f"{config.DS_DIR}/SHORT_{sym}_h{H}_lgb_seed{s}_{split}_P.npy")
                      for s in SEEDS], axis=0).astype(np.float64)
        return np.mean([rank1d(r) for r in P], axis=0), "lgb"
    if mode == "lgbgru":
        rg = rank1d(np.load(f"{config.DS_DIR}/SHORT_{sym}_h{H}_gru_{split}_P.npy").astype(np.float64))
        P = np.stack([rank1d(np.load(f"{config.DS_DIR}/SHORT_{sym}_h{H}_lgb_seed{s}_{split}_P.npy"))
                      for s in SEEDS], axis=0).astype(np.float64)
        rl = P.mean(axis=0)
        return np.mean([rg, rl], axis=0), "lgbgru"
    raise ValueError(mode)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gru"
    print(f"无前视 | {mode} | 滚动30天分位, 冷启动30, 多空独立")
    for split in ["test", "meta_val"]:
        print(f"\n===== {split} =====")
        for sym in ["ETH", "BTC"]:
            try:
                p, _ = load_p(sym, split, mode)
                p_file = f"{config.DS_DIR}/SHORT_{sym}_h{H}_{mode.split('G')[0] if 'G' not in mode else 'gru'}_{split}"
            except FileNotFoundError:
                print(f"  {sym}: 无{mode}预测, 跳过")
                continue
            ctx = AssetContext(sym, H, ds_name=f"ds_{sym}_h{H}")
            y = ctx.y(split)
            conf = np.abs(p - 0.5) * 2
            pred = (p >= 0.5).astype(np.int8)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            day_of = sec // 86400
            days = np.unique(day_of)
            nd = len(days)
            res = []
            for q in [98.5, 99, 99.2, 99.5, 99.8]:
                sel = no_lookahead(day_of, days, pred, conf, None, q, 30, 30)
                n = int(sel.sum())
                acc = (pred[sel] == y[sel]).mean() * 100 if n else 0.0
                tpd = n / nd
                ok = "★" if acc >= 65 and tpd >= 14 else ""
                res.append(f"q={q}: {acc:5.1f}%/{tpd:4.1f}t {ok}")
            print(f"  {split} {sym}[{nd}d]: " + " | ".join(res))
    print()


if __name__ == "__main__":
    main()