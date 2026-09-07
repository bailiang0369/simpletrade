#!/usr/bin/env python3
"""按日统计 test 段 top1% 交易的信号数与准确率分布(R2 逐日滚动阈值, 与生产口径一致)。

输出: 每日信号数 / 每日准确率 的分位数(min/p5/p10/p25/p50/p75/p90/p95/max),
以及坏日/零信号日占比、信号数-准确率相关性。
用法: python probe_daily_stats.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM

FAMS = ("lgb", "xgb", "cat")


def load_pmean(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P, dtype=np.float64)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def pct(a, qs=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    return [float(np.percentile(a, q)) for q in qs]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    for s in ("ETH", "BTC"):
        ctx = AssetContext(s, horizon=30)
        pmv = load_pmean(s, "meta_val")
        pt = load_pmean(s, "test")
        ts_test = ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        conf = np.abs(pt - 0.5) * 2
        r = ESM.r2_eval(conf, (pt >= 0.5).astype(np.int8), ts_test, seed_conf)
        sel = r["sel"]
        ysel = ctx.y("test")[sel]
        pred = r["pred"]
        ok = (pred == ysel).astype(int)
        day = ts_test[sel] // 86400
        all_days = np.unique(ts_test // 86400)          # test 段全部日历日
        n_days = len(all_days)
        dmap = {d: i for i, d in enumerate(all_days)}
        idx = np.array([dmap[d] for d in day])
        cnt = np.bincount(idx, minlength=n_days)
        acc = np.zeros(n_days)
        np.add.at(acc, idx, ok)
        acc = np.divide(acc, np.maximum(cnt, 1), out=np.zeros(n_days), where=cnt > 0)
        have = cnt > 0
        cq, aq = pct(cnt[have]), pct(acc[have])
        qnames = ["min", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "max"]
        print(f"\n######## {s} test 按日统计 ########", flush=True)
        print(f"  总信号= {len(sel)}   总准确率= {(pred == ysel).mean():.4f}   "
              f"日历日= {n_days}   有信号日= {int(have.sum())} ({int(have.sum())/n_days:.1%})   零信号日= {int((~have).sum())} ({int((~have).sum())/n_days:.1%})", flush=True)
        print(f"  每日信号数(仅交易日) {', '.join(f'{n}={v:.2f}' for n, v in zip(qnames, cq))}", flush=True)
        print(f"  每日信号数(含零信号日) {', '.join(f'{n}={v:.2f}' for n, v in zip(qnames, pct(cnt)))}", flush=True)
        print(f"  每日准确率(仅交易日) {', '.join(f'{n}={v:.4f}' for n, v in zip(qnames, aq))}", flush=True)
        # 坏日统计
        n50 = int((acc[have] < 0.5).sum())
        n40 = int(((acc[have] < 0.4) & (cnt[have] >= 20)).sum())
        n20 = int(((acc[have] < 0.3) & (cnt[have] >= 10)).sum())
        print(f"  准确率<50% 的日: {n50} ({n50/int(have.sum()):.1%})   "
              f"<40%且信号>=20 的日: {n40}   <30%且信号>=10 的日: {n20}", flush=True)
        # 信号数与准确率相关性
        print(f"  每日信号数 vs 准确率 Spearman r= {spearman(cnt[have], acc[have]):+.3f}", flush=True)
        # 每日信号数直方简表
        h = np.histogram(cnt[have], bins=[0, 10, 20, 50, 100, 200, 500, 1e9])[0]
        lab = ["1-10", "10-20", "20-50", "50-100", "100-200", "200-500", "500+"]
        print("  每日信号数分布: " + "  ".join(f"{l}={c}" for l, c in zip(lab, h)), flush=True)
        # 列前几个准确率最低的日(带日期), 供人工核查
        sel_day = all_days[have]
        idx = np.argsort(acc[have])[:5]
        dts = np.datetime64(int(sel_day[0]) * 86400, "s").astype("datetime64[D]")
        worst = [(str(dts + int(sel_day[i] - sel_day[0]))[:10], int(cnt[have][i]), round(float(acc[have][i]), 3))
                 for i in idx]
        print(f"  准确率最差的5日(日期, 信号数, 准确率): {worst}", flush=True)


if __name__ == "__main__":
    main()
