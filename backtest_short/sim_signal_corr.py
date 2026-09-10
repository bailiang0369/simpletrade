#!/usr/bin/env python3
"""四周期(3/5/10/15min)因果信号之间的相关性分析:
- 共享1分钟网格: 有信号且看涨=+1, 看跌=-1, 无信号=0
- 两两 Pearson 相关
- 方向一致率: 双方同分钟都有信号时, 方向相同的比例
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from backtest_short.sim_causal_3m import causal_signals
from live.signals import greedy_sparse

HORIZONS = [3, 5, 10, 15]


def grid_signals(H):
    """返回 (grid_ts, series): 1分钟网格上的信号方向序列(±1/0), 以及原始信号列表。"""
    ts_all, dir_all = [], []
    for sym in ["ETH", "BTC"]:
        ts, pr, _ = causal_signals(sym, H)
        idx = greedy_sparse(ts.astype(np.int64), H)
        ts_all.append(ts[idx]); dir_all.append(pr[idx])
    ts = np.concatenate(ts_all); dr = np.concatenate(dir_all)
    o = np.argsort(ts); ts, dr = ts[o], dr[o]
    # 去重(同一分钟只留一条)
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, dr = ts[keep], dr[keep]
    # 1分钟网格
    t0, t1 = ts.min(), ts.max()
    grid = np.arange(t0, t1 + 60, 60)
    pos = np.searchsorted(grid, ts)
    series = np.zeros(len(grid), np.int8)
    series[pos] = np.where(dr >= 0.5, 1, -1)
    return grid, series, ts, dr


def main():
    data = {H: grid_signals(H) for H in HORIZONS}
    Hs = list(HORIZONS)
    n = len(Hs)
    corr = np.zeros((n, n))
    agree = np.zeros((n, n))
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gi, si = data[Hs[i]][0], data[Hs[i]][1]
            gj, sj = data[Hs[j]][0], data[Hs[j]][1]
            # 对齐公共时间范围
            lo = max(gi.min(), gj.min()); hi = min(gi.max(), gj.max())
            mi = (gi >= lo) & (gi <= hi); mj = (gj >= lo) & (gj <= hi)
            a, b = si[mi].astype(np.float64), sj[mj].astype(np.float64)
            # 1) 全网格 Pearson (含0=无信号)
            if a.std() > 0 and b.std() > 0:
                corr[i, j] = np.corrcoef(a, b)[0, 1]
            # 2) 双方都有信号的分钟
            both = (a != 0) & (b != 0)
            if both.sum() > 0:
                agree[i, j] = float((a[both] == b[both]).mean())
                overlap[i, j] = float(both.sum()) / max((a != 0).sum(), 1)
    print("== 两两 Pearson 相关 (含无信号分钟, 整体时序相关性) ==")
    print(f"{'':>8}" + "".join(f"{h:>8}min" for h in Hs))
    for i in range(n):
        print(f"{Hs[i]:>5}min" + "".join(f"{corr[i,j]:>9.3f}" for j in range(n)))
    print("\n== 方向一致率 (双方同分钟都有信号时的同向比例) ==")
    print(f"{'':>8}" + "".join(f"{h:>8}min" for h in Hs))
    for i in range(n):
        print(f"{Hs[i]:>5}min" + "".join(f"{agree[i,j]*100:>8.1f}%" for j in range(n)))
    print("\n== 时间重叠率 (B周期的信号中, 有多少与A同时刻存在) ==")
    print(f"{'':>8}" + "".join(f"{h:>8}min" for h in Hs))
    for i in range(n):
        print(f"{Hs[i]:>5}min" + "".join(f"{overlap[i,j]*100:>8.1f}%" for j in range(n)))

    # 汇总: 每对周期信号数
    print("\n== 信号数量(因果+去重) ==")
    for H in Hs:
        print(f"  {H}min: {len(data[H][2])}")


if __name__ == "__main__":
    main()