#!/usr/bin/env python3
"""固定注(flat bet)资金管理模拟: 每笔投入 = f * 初始本金(固定, 不复利)。

与复利的区别:
- 期末权益 = INIT + b*(0.85*胜局 - 败局), 只取决于总胜局数, 与顺序无关(线性)
- 回撤是绝对亏损额(相对初始本金), 连败不会自我放大, 可承受比复利略高的单注比例
- 风险 = 每注额 × 连败长度; 故关键约束是"最坏连败下不伤本金"

输出: 真实序列确定性期末+MDD, 及 MC(8000路径, 悲观-4pp)的期末分布/MDD分布/大亏概率。
用法: python probe_flat_bet.py
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
BAY = 0.85
INIT = 1000.0
N_SIM = 8000
FRACS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08)


def load_pmean(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(float) / (n - 1)
    return R.mean(axis=0)


def greedy_sparse(ts, gap=1800):
    keep = np.zeros(len(ts), bool)
    last = -10**18
    for i, t in enumerate(ts):
        if t >= last + gap:
            keep[i] = True
            last = t
    return keep


def signal_seq(symbol):
    """返回 (独立事件胜负序列 win(0/1), 胜率 p)。"""
    ctx = AssetContext(symbol, horizon=30)
    pmv = load_pmean(symbol, "meta_val"); pt = load_pmean(symbol, "test")
    ts_test = ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64)
    seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
    r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8), ts_test, seed_conf)
    sel = r["sel"]; ysel = ctx.y("test")[sel]; pred = r["pred"]; ts = ts_test[sel]
    keep = greedy_sparse(ts)
    win = (pred[keep] == ysel[keep]).astype(int)
    return win, win.mean()


def sim_deterministic(win, f):
    """真实序列固定注: 期末与顺序无关, MDD 与顺序有关。"""
    b = f * INIT
    net = np.where(win == 1, BAY * b, -b)
    eq = INIT + np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    mdd_init = float(((peak - eq) / INIT).max())      # 回撤/初始本金
    return float(eq[-1]), mdd_init, int(win.sum()), len(win)


def mc_flat(n, p, f):
    """MC: 固定注 f*INIT。返回 (期末中位, MDD中位, MDD p90, 期末<50%概率, 期末<10%概率)。"""
    b = f * INIT
    wins = np.random.default_rng(7).random((N_SIM, n)) < p
    net = np.where(wins, BAY * b, -b)
    eq = INIT + np.cumsum(net, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (peak - eq) / INIT
    fin = eq[:, -1]
    return (np.median(fin), np.median(mdd), np.quantile(mdd, 0.90),
            float((fin < 0.5 * INIT).mean()), float((fin < 0.1 * INIT).mean()))


def main():
    for s in ("ETH", "BTC"):
        win, p = signal_seq(s)
        n = len(win)
        print(f"\n######## {s}: 独立事件 {n} 笔, 实测胜率 {p:.4f} ########", flush=True)
        print("  --- 真实序列确定性固定注(期末与顺序无关) ---", flush=True)
        print(f"  {'f':>5} | {'每注(U)':>7} | {'总胜/总':>8} | {'期末':>7} | {'回撤/本金':>9}", flush=True)
        for f in FRACS:
            eq, mdd, wcnt, nn = sim_deterministic(win, f)
            print(f"  {f:5.1%} | {f*INIT:7.0f} | {wcnt:>4}/{nn:<3} | {eq:7.0f} | {mdd:9.1%}", flush=True)
        print("  --- MC 悲观档(胜率 -4pp) 8000路径 ---", flush=True)
        print(f"   {'f':>5} | {'每注(U)':>7} | {'期末中位':>8} | {'MDD中位':>8} | {'MDD p90':>8} | {'期末<50%':>8} | {'期末<10%':>8}", flush=True)
        for f in FRACS:
            fin, md50, md90, r50, r10 = mc_flat(n, p - 0.04, f)
            print(f"   {f:5.1%} | {f*INIT:7.0f} | {fin:8.0f} | {md50:8.1%} | {md90:8.1%} | {r50:8.1%} | {r10:8.1%}", flush=True)
        # 连败统计(悲观档)
        wins = np.random.default_rng(7).random((N_SIM, n)) < (p - 0.04)
        runs = [np.diff(np.where(np.concatenate([[0], w, [0]]))[0]).max() - 1 for w in wins[:2000]]
        print(f"  悲观档 2000条路径最大连败: 中位 {int(np.median(runs))} / p90 {int(np.quantile(runs, 0.90))} / 最大 {int(max(runs))}", flush=True)


if __name__ == "__main__":
    main()
