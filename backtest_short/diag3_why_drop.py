#!/usr/bin/env python3
"""诊断: 为什么无前视(因果阈值)后准确率大幅下降, 以及如何恢复到 top1% 水平.

核心机制检验:
  1) 阈值漂移: 每日历史阈值 tau_hist(前30天P99) vs 当日实际P99(tau_day,即top1%切点).
     tau_hist < tau_day -> 因果阈值比当日top1%更松, 选入更多低置信样本.
  2) 信号密度: 因果P99 每天出多少信号 vs top1% 的15个/天.
  3) 准确率-置信度单调性: 高conf样本是否真的更准(conf分箱).
  4) 修复: 在 meta_val 上无前视扫描 q 与 win_days, 冻结最优参数, test 验证,
     目标是密度≈top1%(15/天/币)时准确率接近甚至超过 top1% 有前视水平.

仅评估已完全 5-seed 训练的周期, 避免读正在写的 P 文件.
用法: python backtest_short/diag3_why_drop.py [h3 [h5 ...]]
"""
import os, sys
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
HORIZONS = [int(a) for a in sys.argv[1:]] or [3]


def load_P(sym, h, split):
    """容错加载: (3,n) 单seed 或 (3,5,n) 5-seed -> (n_models, n)."""
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{sym}_h{h}_{f}_{split}_P.npy")
          for f in FAMILIES]
    # 统一到 (n_models, n): 3个family都是5-seed才有意义
    shapes = {p.ndim for p in Ps}
    if shapes != {2}:
        raise SystemExit(f"SKIP {sym} h{h} {split}: 未全部5-seed, shapes={[p.shape for p in Ps]}")
    return np.stack(Ps, axis=0).reshape(-1, Ps[0].shape[-1]).astype(np.float64)


def rank_ens(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def causal_sel(sec, day_of, conf, days, q, win_days):
    """无前视: 每日阈值 = 前 win_days 天历史 conf 的 Pq. 返回 (sel, taus, counts)."""
    day_confs = {int(d): conf[day_of == d] for d in days}
    day_list = days.astype(int).tolist()
    sel = np.zeros(len(sec), dtype=bool)
    taus, cnts = [], []
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - win_days):i]
        if len(prior) < win_days:  # 冷启动: 不足 win_days 天不发
            taus.append(np.nan); cnts.append(0)
            continue
        hist = np.concatenate([day_confs[d2] for d2 in prior])
        tau = float(np.percentile(hist, q))
        m = day_of == d
        c = int((conf[m] >= tau).sum())
        sel[m] = conf[m] >= tau
        taus.append(tau); cnts.append(c)
    return sel, np.asarray(taus), np.asarray(cnts)


def daily_top1_sel(day_of, conf, days):
    sel = np.zeros(len(conf), dtype=bool)
    for d in days:
        m = day_of == d
        k = max(1, int(np.ceil(int(m.sum()) * 0.01)))
        sub = np.where(m)[0]
        sel[sub[np.argsort(-conf[sub])[:k]]] = True
    return sel


def analyze(sym, H):
    ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
    Pmv, Pte = load_P(sym, H, "meta_val"), load_P(sym, H, "test")
    pmv, pte = rank_ens(Pmv), rank_ens(Pte)
    ymv, yte = ctx.y("meta_val"), ctx.y("test")
    smv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    ste = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
    cmv, cte = np.abs(pmv - 0.5) * 2, np.abs(pte - 0.5) * 2
    prmv, prte = (pmv >= 0.5).astype(np.int8), (pte >= 0.5).astype(np.int8)
    dmv, dte = smv // 86400, ste // 86400
    days_mv, days_te = np.unique(dmv), np.unique(dte)

    print(f"\n######## {sym} h{H} | 15模型集成 | meta_val {len(days_mv)}天 test {len(days_te)}天 ########")

    # 1) top1 有前视参考
    selt1 = daily_top1_sel(dte, cte, days_te)
    print(f"[top1 有前视] n={int(selt1.sum())} ({selt1.sum()/len(days_te):.1f}/天) "
          f"acc={(prte[selt1]==yte[selt1]).mean()*100:.2f}%")

    # 2) 阈值漂移: 历史P99 vs 当日P99, 及每日信号数
    sel30, tau30, cnt30 = causal_sel(ste, dte, cte, days_te, 99.0, 30)
    tau_day = np.array([float(np.percentile(cte[dte == d], 99.0)) for d in days_te])
    drift = tau_day - tau30
    n30 = int(sel30.sum())
    print(f"[因果P99/30天] n={n30} ({n30/len(days_te):.1f}/天) acc={(prte[sel30]==yte[sel30]).mean()*100:.2f}%")
    print(f"[漂移诊断] 当日P99 - 历史P99: 中位={np.nanmedian(drift):+.4f} "
          f"均值={np.nanmean(drift):+.4f} | 历史阈值更松的天数占比={np.nanmean(drift>0)*100:.0f}%")
    print(f"[密度诊断] 因果信号数/天: 中位={np.median(cnt30):.0f} 均值={np.mean(cnt30):.1f} "
          f"(top1%参考=15/天)")

    # 3) 修复: meta_val 无前视扫描 (win_days, q) -> 冻结 -> test 验证
    best = None
    print("  [meta_val扫描]  win_days  q        n/天  acc%  |  test: n/天 acc%")
    for win in [30, 60, 90]:
        for q in [99.0, 99.2, 99.5, 99.7, 99.8, 99.9]:
            sel_mv, _, cnt = causal_sel(smv, dmv, cmv, days_mv, q, win)
            n_mv = int(sel_mv.sum())
            if n_mv == 0:
                continue
            acc_mv = (prmv[sel_mv] == ymv[sel_mv]).mean()
            sel_te, _, _ = causal_sel(ste, dte, cte, days_te, q, win)
            n_te = int(sel_te.sum())
            acc_te = (prte[sel_te] == yte[sel_te]).mean() if n_te else 0
            print(f"  w={win:<3} q={q:<5} mv: {n_mv/len(days_mv):5.1f} {acc_mv*100:5.2f}%"
                  f" | test: {n_te/max(len(days_te),1):5.1f} {acc_te*100:5.2f}%")
            # 目标: meta_val 密度接近15/天(±30%) 且 acc 最高 -> 冻结
            per_day = n_mv / len(days_mv)
            if 10 <= per_day <= 20:
                score = acc_mv
                if best is None or score > best[0]:
                    best = (score, win, q, acc_te, n_te)
    if best:
        print(f"  => meta_val最优(密度10-20/天): win={best[1]} q={best[2]} "
              f"mv_acc={best[0]*100:.2f}% | test_acc={best[3]*100:.2f}% n={best[4]}")
    return {"sym": sym, "h": H}


def main():
    for H in HORIZONS:
        for sym in ["ETH", "BTC"]:
            try:
                analyze(sym, H)
            except SystemExit as e:
                print(e)


if __name__ == "__main__":
    main()
