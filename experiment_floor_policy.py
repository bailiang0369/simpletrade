"""mv 离线策略对比: 在固定 1% 覆盖率前提下, 测试不同选择机制对逐月下限的影响。

只用已存的 mv 预测数组 + AssetContext(仅读 label/ret/times), 不重训、不碰 test。
对每个策略报告: 整体 acc / 最差月 acc / 最差月 n / n_月份_ac55 / 日均单数。
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

def load():
    ctx = AssetContext("ETH", horizon=30)
    p_rk = np.load(f"{config.DS_DIR}/ETH_floor_mv_p.npy")
    std = np.load(f"{config.DS_DIR}/ETH_floor_mv_std.npy")
    y = ctx.y("meta_val"); retf = ctx.retf("meta_val")
    t = ctx.times("meta_val")
    sec = np.asarray(t).astype("datetime64[s]").astype(np.int64)
    return p_rk, std, y, retf, sec

def month_stats(sel, pred, y, sec):
    m = sec[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(m)
    accs = {str(u): float((pred[sel] == y[sel])[m == u].mean()) for u in uniq}
    ns = {str(u): int((m == u).sum()) for u in uniq}
    min_a = min(accs.values()); min_k = next(ns[k] for k, a in accs.items() if a == min_a)
    n_bad = sum(1 for a in accs.values() if a < 0.55)
    return float((pred[sel] == y[sel]).mean()), min_a, min_k, n_bad, ns

def run(name, sel, pred, y, retf, sec):
    if sel.size == 0:
        return
    sd = len(sec)
    n_days = np.unique(sec // 86400).size
    tpd = sel.size / n_days
    acc, min_a, min_k, n_bad, _ = month_stats(sel, pred, y, sec)
    cr = sel.size / sd
    ret = retf[sel].mean() * 1e4
    print(f"  {name:<34} cov={cr:.4f} tpd={tpd:5.1f} acc={acc:.4f} "
          f"min_month={min_a:.4f}(n={min_k}) bad_months(<55)={n_bad} ret={ret:.1f}bp")

def main():
    p, std, y, retf, sec = load()
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    n = len(p); k = max(1, int(round(n * 0.01)))
    day = sec // 86400
    days = np.unique(day)
    print(f"mv rows={n} 全局n_days={days.size} 全局k(1%)={k}\n")

    # P0 基线: 全局 rank 置信 top-1%
    sel0 = np.argsort(-conf)[:k]
    run("P0 全局top1%(基线)", sel0, pred, y, retf, sec)

    # P1 每日 top-1% (操作口径: 每天交易
    sel_d = np.zeros(n, bool)
    for d in days:
        m = day == d
        kd = max(1, int(round(int(m.sum()) * 0.01)))
        sub = np.where(m)[0]
        sel_d[sub[np.argsort(-conf[sub])[:kd]]] = True
    run("P1 每日top1%", np.where(sel_d)[0], pred, y, retf, sec)

    # P1 月度明细
    sel1 = np.where(sel_d)[0]; m1 = sec[sel1]
    um = np.unique(m1.astype("datetime64[s]").astype("datetime64[M]"))
    line = "    P1 逐月: "
    for u in um:
        mm = (m1.astype("datetime64[s]").astype("datetime64[M]")) == u
        line += f"{str(u)[:7]}={float((pred[sel1]==y[sel1])[mm].mean()):.3f}({int(mm.sum())}) "
    print(line)

    # P1c: 每日用 ceil 保证每天>=其该日floor行数, 以满足 tpd>=14
    sel_c = np.zeros(n, bool)
    for d in days:
        m = day == d
        kd = max(1, int(np.ceil(int(m.sum()) * 0.01)))
        sub = np.where(m)[0]
        sel_c[sub[np.argsort(-conf[sub])[:kd]]] = True
    run("P1c 每日top1%(ceil)", np.where(sel_c)[0], pred, y, retf, sec)

    # P2 每日top1% + 低置信日守门(跳过低质量日)
    #   每天内, 若当天 top1% 命中样本的均值置信低于 bar, 该日不交易
    for bar in (0.90, 0.93, 0.95, 0.97):
        sm = np.zeros(n, bool)
        for d in days:
            m = day == d
            sub = np.where(m)[0]
            if sub.size == 0: continue
            kd = max(1, int(round(int(m.sum()) * 0.01)))
            top = sub[np.argsort(-conf[sub])[:kd]]
            if conf[top].mean() >= bar:
                sm[top] = True
        run(f"P2 每日top1%+日置信门>{bar}", np.where(sm)[0], pred, y, retf, sec)

    # P3 全局 topk, 但按 (conf - lam*std) 排序(惩罚高分歧)
    for lam in (0.2, 0.5, 1.0):
        sc = conf - lam * std
        sel = np.argsort(-sc)[:k]
        run(f"P3 全局 topk by conf-{lam}*std", sel, pred, y, retf, sec)

    # P4: 每月局部 top-1% => 样本量稳定, 消解全局clump
    mts = sec.astype("datetime64[s]").astype("datetime64[M]")
    sm = np.zeros(n, bool)
    muniq = np.unique(mts)
    for mu in muniq:
        mm = mts == mu
        km = max(1, int(round(int(mm.sum()) * 0.01)))
        sub = np.where(mm)[0]
        sm[sub[np.argsort(-conf[sub])[:km]]] = True
    run("P4 每月局部top1%(样本量稳定)", np.where(sm)[0], pred, y, retf, sec)

    # P5: 复合: 每日 top1%, 但置信用跨日回望的"滚动可信"(近似跳月保留) -- 简化: 日内最好且全局 top1k 双条件
    #     即: 全局候选集合(conf 前 2k) 且 每天最多取该日分到的配额(=该日1%) -> 压住clump, 且挤掉低质日
    cand = np.argsort(-conf)[: 2 * k]
    sm = np.zeros(n, bool)
    for d in days:
        m = day == d
        sub = np.where(m)[0]
        inter = sub[np.isin(sub, cand)]
        kd = max(1, int(round(int(m.sum()) * 0.01)))
        if inter.size:
            sm[inter[np.argsort(-conf[inter])[:min(kd, inter.size)]]] = True
    run("P5 全局候2k∩每日配额", np.where(sm)[0], pred, y, retf, sec)

    print("\n说明: cov 为实际覆盖率(部分策略会低于1%); 目标: min_month>=0.55 且 acc 尽量高。")


if __name__ == "__main__":
    main()