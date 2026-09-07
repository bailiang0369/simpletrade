#!/usr/bin/env python3
"""二次集成: ETH/BTC 的 单独训练 与 联合训练 两套 pool20 预测 rank 融合, 看是否加固单月下限。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
SEEDS = 5


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


def load_Ps(symbol, tag, split):
    Ps = [np.load(f"{config.DS_DIR}/{tag}_{symbol}_{f}_{split}_P.npy") if tag == "JOINT"
          else np.load(f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy")
          for f in FAMS]
    return np.concatenate(Ps, axis=0)


def main():
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        for split in ("meta_val", "test"):
            y = ctx.y(split)
            sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            n = len(y)
            solo = load_Ps(symbol, "", split)
            joint = load_Ps(symbol, "JOINT", split)
            # 各套 rank 融合
            p_solo = rank_mean(solo, n)
            p_joint = rank_mean(joint, n)
            print(f"\n===== {symbol} {split} =====")
            for name, p in (("solo ", p_solo), ("joint", p_joint)):
                acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
                print(f"[{name}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad}")
            # 二次融合: 两套 rank 平均
            p2 = (rank_mean(solo, n) + rank_mean(joint, n)) / 2
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p2, y, sec_arr)
            print(f"[2nd-avg] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f}")
            print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
            del solo, joint; gc.collect()


if __name__ == "__main__":
    main()
