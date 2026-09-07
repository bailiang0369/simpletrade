#!/usr/bin/env python3
"""meta_val 上选择 ETH 的 solo/joint 融合权重 α, 再在 test 上报结果(合法模型选择)。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]  # p = α*p_joint + (1-α)*p_solo


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
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") if tag == "JOINT"
          else np.load(f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy")
          for f in FAMS]
    return np.concatenate(Ps, axis=0)


def main():
    symbol = "ETH"
    ctx = AssetContext(symbol, horizon=30)
    data = {}
    for split in ("meta_val", "test"):
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        n = len(y)
        p_solo = rank_mean(load_Ps(symbol, "", split), n)
        p_joint = rank_mean(load_Ps(symbol, "JOINT", split), n)
        data[split] = (y, sec_arr, p_solo, p_joint)
    print("  α      meta_val(min,bad,acc)          test(min,bad,acc)")
    results = []
    for a in ALPHAS:
        row = {}
        for split in ("meta_val", "test"):
            y, sec_arr, p_solo, p_joint = data[split]
            p = a * p_joint + (1 - a) * p_solo
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
            row[split] = (min_a, nbad, acc)
        results.append((a, row))
        print(f"  {a:.1f}   mv(min={row['meta_val'][0]:.4f},bad={row['meta_val'][1]},"
              f"acc={row['meta_val'][2]:.4f})  test(min={row['test'][0]:.4f},"
              f"bad={row['test'][1]},acc={row['test'][2]:.4f})")
    # meta_val 选择: 优先坏月最少, 其次 min_month 最高, 再次 acc 最高
    best = max(results, key=lambda r: (-r[1]["meta_val"][1], r[1]["meta_val"][0], r[1]["meta_val"][2]))
    print(f"\nmeta_val 最优 α={best[0]:.1f} → test: min_month={best[1]['test'][0]:.4f}, "
          f"bad={best[1]['test'][1]}, acc={best[1]['test'][2]:.4f}")


if __name__ == "__main__":
    main()
