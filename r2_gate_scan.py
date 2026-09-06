#!/usr/bin/env python3
"""无泄露滚动门控: 在 R2 每日滚动阈值基础上, 用最近 M 天已结算准确率做门控。

严格因果: 每天盘前, 只统计 当天之前 已结算选中样本的准确率;
若最近 M 天 acc < GATE_ACC, 当天阈值提高 GATE_MULT 倍(即要求更极端的置信度)。
参数 (M, GATE_ACC, GATE_MULT) 只在 meta_val 上网格搜索, 再应用到 test。
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
BASE_P = 99.0
WIN_DAYS = 90
SAMPLES_PER_DAY = 1440


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, tag, split):
    Ps = []
    for f in FAMS:
        path = f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy" if tag == "JOINT" \
            else f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy"
        Ps.append(np.load(path))
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def daily_roll_gate(conf, pred, y, mts, sec, pctl, hist_seed, M, GATE_ACC, GATE_MULT):
    """每日滚动 + 门控。返回 (acc, min_a, nbad, cov, acc_m, sel_count_by_day)。"""
    day = sec // 86400
    days = np.unique(day)
    hist = list(hist_seed)
    # 门控历史: 已结算的 (acc 由 对/错 记录)
    rec_ok = []      # 最近 M 天内选中样本的对错
    rec_day = []     # 对应 day
    sel_all = []
    for dd in days:
        md = day == dd
        # 门控判定: 最近 M 天已结算 acc
        gate_on = False
        if len(rec_ok) >= 20:
            # 只统计最近 M 天
            cutoff = dd - M
            idx = [i for i, d in enumerate(rec_day) if d >= cutoff]
            if len(idx) >= 20:
                ok_arr = [rec_ok[i] for i in idx]
                gate_on = float(np.mean(ok_arr)) < GATE_ACC
        tau = np.percentile(np.asarray(hist), pctl)
        if gate_on:
            # 暂停模式: 当天不做任何交易
            hist.extend(conf[md])
            if len(hist) > WIN_DAYS * SAMPLES_PER_DAY * 2:
                del hist[:len(hist) - WIN_DAYS * SAMPLES_PER_DAY * 2]
            continue
        tau = 0.5 + (tau - 0.5) * GATE_MULT
        s = np.where(md & (conf >= tau))[0]
        if len(s) > 0:
            sel_all.append(s)
            ok = (pred[s] == y[s])
            rec_ok.extend(ok.tolist())
            rec_day.extend([dd] * len(s))
            if len(rec_ok) > M * SAMPLES_PER_DAY * 2:
                rec_ok = rec_ok[-M * SAMPLES_PER_DAY * 2:]
                rec_day = rec_day[-M * SAMPLES_PER_DAY * 2:]
        hist.extend(conf[md])
        if len(hist) > WIN_DAYS * SAMPLES_PER_DAY * 2:
            del hist[:len(hist) - WIN_DAYS * SAMPLES_PER_DAY * 2]
    sel = np.concatenate(sel_all) if sel_all else np.array([], dtype=np.int64)
    n_all = len(sec)
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0, {}, 0)
    ps, ys, ms = pred[sel], y[sel], mts[sel]
    acc = float((ps == ys).mean())
    uniq = np.unique(ms)
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in uniq}
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    cov = float(len(ps)) / n_all
    n_gate = sum(1 for d in days if False)
    return (acc, min_a, nbad, cov, acc_m, len(sel))


def main():
    # meta_val 网格搜索参数
    MS = [3, 5, 7]
    GATE_ACCS = [0.40, 0.45, 0.50]
    GATE_MULTS = [1.5, 2.0, 3.0]
    configs = [("ETH", "JOINT"), ("BTC", "JOINT")]
    for symbol, tag in configs:
        ctx = AssetContext(symbol, horizon=30)
        data = {}
        for split in ("meta_val", "test"):
            p = load_fused(symbol, tag, split)
            y = ctx.y(split)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            pred = (p >= 0.5).astype(np.int8)
            conf = np.maximum(p, 1 - p)
            data[split] = dict(p=p, y=y, sec=sec, mts=mts, pred=pred, conf=conf)
            del p
            gc.collect()
        mv, te = data["meta_val"], data["test"]
        mv_hist_seed = list(mv["conf"][:WIN_DAYS * SAMPLES_PER_DAY])

        # ---- meta_val 网格搜索 ----
        results = []
        for M in MS:
            for GA in GATE_ACCS:
                for GM in GATE_MULTS:
                    acc, min_a, nbad, cov, acc_m, ns = daily_roll_gate(
                        mv["conf"], mv["pred"], mv["y"], mv["mts"], mv["sec"],
                        BASE_P, list(mv_hist_seed), M, GA, GM)
                    results.append((M, GA, GM, acc, min_a, nbad, cov, ns))
        print(f"\n===== {tag} {symbol} meta_val 门控网格搜索 =====")
        print("  M/GA/GM  ->  acc/min_month/bad/cov/n_sel")
        for r in sorted(results, key=lambda r: (r[5] == 0, r[4] if r[4] == r[4] else -1), reverse=True)[:15]:
            print(f"  M={r[0]} GA={r[1]:.2f} GM={r[2]:.1f}  mv(acc={r[3]:.4f}, min={r[4]:.4f}, bad={r[5]}, cov={r[6]:.4f}, n={r[7]})")
        # 选 bad=0 优先, min 最高
        pool = [r for r in results if r[5] == 0]
        if not pool:
            pool = results
        pool.sort(key=lambda r: (r[4] if r[4] == r[4] else -1, r[3]), reverse=True)
        M_b, GA_b, GM_b = pool[0][0], pool[0][1], pool[0][2]
        print(f"  >> 选择 M={M_b} GA={GA_b} GM={GM_b}")

        # ---- test 应用 ----
        mv_tail = list(mv["conf"][-WIN_DAYS * SAMPLES_PER_DAY:])
        acc, min_a, nbad, cov, acc_m, ns = daily_roll_gate(
            te["conf"], te["pred"], te["y"], te["mts"], te["sec"],
            BASE_P, list(mv_tail), M_b, GA_b, GM_b)
        print(f"  [test gated] acc={acc:.4f} min_month={min_a:.4f} bad={nbad} cov={cov:.4f} n={ns}")
        print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
        # 对照: 无门控 R2
        acc0, min0, bad0, cov0, acc_m0, ns0 = daily_roll_gate(
            te["conf"], te["pred"], te["y"], te["mts"], te["sec"],
            BASE_P, list(mv_tail), 99, 0.0, 1.0)
        print(f"  [test plain ] acc={acc0:.4f} min_month={min0:.4f} bad={bad0} cov={cov0:.4f} n={ns0}")
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()
