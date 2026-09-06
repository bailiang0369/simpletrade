#!/usr/bin/env python3
"""funding 与未来30min方向的关系探查 —— 零训练, 决定是否值得投入。
指标:
  1) 全样本点列相关 corr(funding, 未来30min对数收益)
  2) 分5桶看 未来P(up) 是否随funding单调变化(>0.52 或 <0.48 才算有价值)
  3) 极端|funding| 桶是否均值回归(反向)
用法: python experiment_funding_probe.py ETH
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

H = config.HORIZON_MIN  # 30


def run(symbol):
    ctx = AssetContext(symbol)
    f = ctx.o  # 复用数组分配, 下面直接读 raw
    del ctx
    import pyarrow.parquet as pq
    t = pq.read_table(f"data/datasets/raw_{symbol}.parquet", columns=["ts", "close", "funding"])
    close = t["close"].to_numpy().astype(np.float64)
    fr = t["funding"].to_numpy().astype(np.float64)
    n = len(close)
    # 未来30min方向 (末尾无未来, 丢弃)
    fut = close[H:] / close[:-H] - 1.0
    up = fut > 0.0
    f_cur = fr[:-H]
    c = np.corrcoef(f_cur, fut)[0, 1]
    # 分5桶按 funding 值
    qs = np.quantile(f_cur, [0.2, 0.4, 0.6, 0.8])
    print(f"\n### {symbol}  n={n-H}")
    print(f"corr(funding, 未来30min收益) = {c:+.5f}")
    labs = ["Q1(最负)", "Q2", "Q3", "Q4", "Q5(最正)"]
    edges = [f_cur.min()] + list(qs) + [f_cur.max()]
    print(f"{'桶':<12}{'count':>10}{'P(up)':>8}{'均值funding':>12}")
    for i in range(5):
        m = (f_cur >= edges[i]) & (f_cur <= edges[i + 1])
        mm = m & (i == 0 or True)  # 桶边界重叠仅首桶含min
        sel = mm if i == 0 else ((f_cur > edges[i]) & (f_cur <= edges[i + 1]))
        # Q5 含 max; Q1 含 min; 区间用半开方便
        sel = ((f_cur >= edges[i]) if i == 0 else (f_cur > edges[i])) & (f_cur <= edges[i + 1])
        print(f"{labs[i]:<12}{int(sel.sum()):>10}{float(up[sel].mean()):>8.4f}{float(f_cur[sel].mean()):>12.4f}")
    # 极端|funding| 桶 (前0.5% 最极端)
    for tag, sl in (("|funding|前0.5%极值", np.abs(f_cur) >= np.quantile(np.abs(f_cur), 0.995)),
                    ("|funding|中位区", (np.abs(f_cur) < np.quantile(np.abs(f_cur), 0.5)))):
        print(f"{tag:<18} count={int(sl.sum()):>8}  P(up)={float(up[sl].mean()):.4f}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETH"
    run(sym)