#!/usr/bin/env python3
"""15模型集成(3family×5seed) 无前视因果阈值评估: 只跑指定周期."""
import os, sys
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]


def load_P(sym, h, split):
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{sym}_h{h}_{f}_{split}_P.npy")
          for f in FAMILIES]
    P = np.stack(Ps, axis=0)
    if P.ndim == 3:
        P = P.reshape(-1, P.shape[-1])  # (15, n)
    return P.astype(np.float64)


def rank_ens(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def quantile_signal(sec, day_of, conf, pred, days, q, win_days=30, cold=30):
    day_confs = {int(d): conf[day_of == d] for d in days}
    day_list = days.astype(int).tolist()
    sel = np.zeros(len(sec), dtype=bool)
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - win_days):i]
        if len(prior) < cold:
            continue
        hist = np.concatenate([day_confs[d2] for d2 in prior])
        tau = float(np.percentile(hist, q))
        m = day_of == d
        sel[m] = conf[m] >= tau
    return sel


def main():
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for sym in ["ETH", "BTC"]:
        ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
        split = "test"
        P = load_P(sym, H, split)
        n_models = P.shape[0]
        p = rank_ens(P)
        y = ctx.y(split)
        sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(p - 0.5) * 2
        pred = (p >= 0.5).astype(np.int8)
        day_of = sec // 86400
        days = np.unique(day_of)

        print(f"\n=== {sym} h{H} [{n_models}模型集成] test {len(days)}天 ===")
        for q in [98, 99, 99.2, 99.5, 99.8]:
            sel = quantile_signal(sec, day_of, conf, pred, days, q)
            n = int(sel.sum())
            acc = float((pred[sel] == y[sel]).mean()) if n else 0
            print(f"  q={q:<5}: n={n:6d} ({n/len(days):5.1f}/天) acc={acc*100:5.2f}%")

        # 每日top1% 有前视参考
        selt = np.zeros(len(sec), dtype=bool)
        for d in days:
            m = day_of == d
            k = max(1, int(np.ceil(int(m.sum()) * 0.01)))
            sub = np.where(m)[0]
            selt[sub[np.argsort(-conf[sub])[:k]]] = True
        print(f"  [top1有前视] n={selt.sum()} ({selt.sum()/len(days):.1f}/天) "
              f"acc={(pred[selt]==y[selt]).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
