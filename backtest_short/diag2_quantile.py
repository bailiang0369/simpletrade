#!/usr/bin/env python3
"""正确口径诊断: 为什么无前视(因果阈值)后准确率下降 + 用 meta_val 无前视调最优分位.

修复: conf bin 准确率 = (pred==y).mean() 而非 y.mean()
方案: 在 meta_val 段扫描分位点(历史conf的Pq), 找准确率-数量均衡点,
      然后在 test 段用相同分位点验证 -> 全程无前视.
"""
import os, sys
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext

HORIZONS = [3, 5, 10, 15]
FAMILIES = ["lgb", "xgb", "cat"]


def load_P(sym, h, split):
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{sym}_h{h}_{f}_{split}_P.npy")
          for f in FAMILIES]
    return np.stack(Ps, axis=0).astype(np.float64)


def rank_ens(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def prob_ens(P):
    return P.mean(axis=0)


def conf_bin_acc(conf, pred, y):
    """conf 分箱: (pred==y) 准确率, 看单调性"""
    qs = np.percentile(conf, np.linspace(0, 100, 11))
    rows = []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (conf >= lo) & (conf <= hi)
        if m.sum() < 100:
            continue
        rows.append((lo, hi, m.sum(), float((pred[m] == y[m]).mean())))
    return rows


def daily_top1(sec, day_of, conf, pred, y, days):
    sel = np.zeros(len(sec), dtype=bool)
    for d in days:
        m = day_of == d
        k = max(1, int(np.ceil(int(m.sum()) * 0.01)))
        sub = np.where(m)[0]
        sel[sub[np.argsort(-conf[sub])[:k]]] = True
    return sel


def quantile_signal(sec, day_of, conf, pred, y, days, q, win_days=30, cold=30):
    """朴素因果: 每日阈值=前win_days天历史conf的Pq分位, 无前视"""
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
    for sym in ["ETH", "BTC"]:
        for H in HORIZONS:
            ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
            # meta_val 调参, test 验证
            Pmv = load_P(sym, H, "meta_val")
            Pte = load_P(sym, H, "test")
            ymv = ctx.y("meta_val"); yte = ctx.y("test")
            smv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
            ste = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

            for label, ens_fn in [("rank", rank_ens), ("prob", prob_ens)]:
                pmv = ens_fn(Pmv); pte = ens_fn(Pte)
                cmv = np.abs(pmv - 0.5) * 2; cte = np.abs(pte - 0.5) * 2
                prmv = (pmv >= 0.5).astype(np.int8); prte = (pte >= 0.5).astype(np.int8)
                dmv = smv // 86400; dte = ste // 86400
                days_mv = np.unique(dmv); days_te = np.unique(dte)

                # meta_val 上扫描分位
                best = None
                print(f"\n=== {sym} h{H} [{label}] ===")
                print(f"  meta_val 天数={len(days_mv)} test 天数={len(days_te)}")
                for q in [95, 96, 97, 98, 99, 99.2, 99.5, 99.8]:
                    sel_mv = quantile_signal(smv, dmv, cmv, prmv, ymv, days_mv, q)
                    n_mv = int(sel_mv.sum())
                    acc_mv = float((prmv[sel_mv] == ymv[sel_mv]).mean()) if n_mv else 0
                    per_day = n_mv / max(len(days_mv), 1)
                    # test 上用相同分位
                    sel_te = quantile_signal(ste, dte, cte, prte, yte, days_te, q)
                    n_te = int(sel_te.sum())
                    acc_te = float((prte[sel_te] == yte[sel_te]).mean()) if n_te else 0
                    print(f"    q={q:<5} mv: n={n_mv:6d} ({per_day:5.1f}/天) acc={acc_mv*100:5.2f}%"
                          f" | test: n={n_te:6d} ({n_te/max(len(days_te),1):5.1f}/天) acc={acc_te*100:5.2f}%")
                # top1 参考(有前视)
                sel_t1 = daily_top1(ste, dte, cte, prte, yte, days_te)
                print(f"    [top1 有前视] n={sel_t1.sum()} acc={(prte[sel_t1]==yte[sel_t1]).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
