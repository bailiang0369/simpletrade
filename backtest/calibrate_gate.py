#!/usr/bin/env python3
"""通用后处理验证: 置信度校准(isotonic) + 分档位保底门控。与形态无关, 非过拟合。

组合对比 (均无泄露, meta_val 定参, test 应用):
  A. baseline  R2 每日滚动 (现状)
  B. +calibr   在 meta_val 拟合 conf->真命中 的 isotonic 映射, test 用校准conf选样
  C. +gate     在校准基础上, 某conf档位最近90天历史命中率<55% 则掐掉该档
目标: 顶1% 总准>=0.65 且 单月下限>=0.55。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
WIN_DAYS = 90
SAMP = 1440
PS = [98.5, 98.75, 99.0, 99.25, 99.5]


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def summarize(sel, pred, y, mts, sec):
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0, 0.0, {})
    ps, ys, ms = pred[sel], y[sel], mts[sel]
    acc = float((ps == ys).mean())
    n_days = np.unique(sec // 86400).size
    cov = float(len(ps)) / len(sec)
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in np.unique(ms)}
    min_a = min(acc_m.values())
    nbad = sum(1 for v in acc_m.values() if v < 0.55)
    return (acc, min_a, nbad, cov, len(ps) / n_days, acc_m)


def daily_roll(conf, pred, y, mts, sec, pctl, hist_seed):
    day = sec // 86400
    days = np.unique(day)
    hist = list(hist_seed)
    sel_list = []
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), pctl)
        s = np.where(md & (conf >= tau))[0]
        sel_list.append(s)
        hist.extend(conf[md])
        if len(hist) > WIN_DAYS * SAMP * 2:
            del hist[:len(hist) - WIN_DAYS * SAMP * 2]
    return np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)


def daily_roll_gate(conf, pred, y, mts, sec, cal, pctl, hist_seed, GATE_ACC):
    """每日滚动 + 校准conf + 档位门控。 bucket_ok/bucket_n 按 conf 档(0-100)累计历史对错。"""
    day = sec // 86400
    days = np.unique(day)
    cconf = cal(conf) if cal is not None else conf
    hist = list(hist_seed)
    b_ok = [0] * 101
    b_n = [0] * 101
    sel_list = []
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), pctl)
        s = np.where(md & (cconf >= tau))[0]
        if len(s) > 0:
            b = np.clip((cconf[s] * 100).astype(int), 0, 100)
            hit = np.array([b_ok[bb] / max(b_n[bb], 1) for bb in b])
            known = np.array([b_n[bb] >= 20 for bb in b])
            s = s[~(known & (hit < GATE_ACC))]
        sel_list.append(s)
        if len(s) > 0:
            b = np.clip((cconf[s] * 100).astype(int), 0, 100)
            for bb, o in zip(b, (pred[s] == y[s])):
                b_ok[bb] += int(o)
                b_n[bb] += 1
        hist.extend(cconf[md])
        if len(hist) > WIN_DAYS * SAMP * 2:
            del hist[:len(hist) - WIN_DAYS * SAMP * 2]
    return np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)


def main():
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        data = {}
        for split in ("meta_val", "test"):
            p = load_fused(symbol, split)
            y = ctx.y(split)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            data[split] = dict(conf=np.maximum(p, 1 - p), pred=(p >= 0.5).astype(np.int8),
                               y=y, mts=mts, sec=sec)
            del p
            gc.collect()
        mv, te = data["meta_val"], data["test"]

        # isotonic 校准
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(mv["conf"], (mv["pred"] == mv["y"]).astype(float))
        cal = lambda c: iso.predict(c)
        mv_cal = cal(mv["conf"])
        mv_tail = list(mv_cal[-WIN_DAYS * SAMP:])

        # meta_val 选 pctl
        print(f"\n===== {symbol} meta_val 扫描 (校准conf) =====")
        cands = []
        for pctl in PS:
            sel = daily_roll(mv_cal, mv["pred"], mv["y"], mv["mts"], mv["sec"], pctl,
                             list(mv_cal[:WIN_DAYS * SAMP]))
            acc, min_a, nbad, cov, tpd, _ = summarize(sel, mv["pred"], mv["y"], mv["mts"], mv["sec"])
            print(f"  P={pctl:5.2f} acc={acc:.4f} min={min_a:.4f} bad={nbad} cov={cov:.4f}")
            cands.append((pctl, nbad, (cov - 0.01) < 0.004, min_a))
        cands.sort(key=lambda r: (r[1] == 0, r[2], r[3]), reverse=True)
        pbest = cands[0][0]
        print(f"  >> choose P={pbest}")

        # A baseline: 未校准 conf, meta_val尾部
        mv_tail_raw = list(mv["conf"][-WIN_DAYS * SAMP:])
        sel0 = daily_roll(te["conf"], te["pred"], te["y"], te["mts"], te["sec"], pbest, list(mv_tail_raw))
        a = summarize(sel0, te["pred"], te["y"], te["mts"], te["sec"])
        print(f"[A baseline ] acc={a[0]:.4f} min={a[1]:.4f} bad={a[2]} cov={a[3]:.4f} tpd={a[4]:.2f}")
        print("  逐月:", {k: round(v,3) for k,v in a[5].items()})

        # B +calibr
        selc = daily_roll(cal(te["conf"]), te["pred"], te["y"], te["mts"], te["sec"], pbest, list(mv_tail))
        b = summarize(selc, te["pred"], te["y"], te["mts"], te["sec"])
        print(f"[B +calibr ] acc={b[0]:.4f} min={b[1]:.4f} bad={b[2]} cov={b[3]:.4f} tpd={b[4]:.2f}")
        print("  逐月:", {k: round(v,3) for k,v in b[5].items()})

        # C +gate
        selg = daily_roll_gate(te["conf"], te["pred"], te["y"], te["mts"], te["sec"],
                               cal, pbest, list(mv_tail), 0.50)
        c = summarize(selg, te["pred"], te["y"], te["mts"], te["sec"])
        print(f"[C +gate   ] acc={c[0]:.4f} min={c[1]:.4f} bad={c[2]} cov={c[3]:.4f} tpd={c[4]:.2f}")
        print("  逐月:", {k: round(v,3) for k,v in c[5].items()})
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()