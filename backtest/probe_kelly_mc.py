#!/usr/bin/env python3
"""蒙特卡洛资金管理: 在模型胜率 p 假设下, 随机生成 30min窗口去重信号序列,
统计不同下注比例 f 的 期末权益/最大回撤分布/大亏概率。比确定性路径更贴近实盘离散。

- 每注: 下注 f*当前权益, 赢 +0.85*f*eq, 输 -f*eq (币安事件合约, 返奖0.85)
- 信号数: test 段实际窗口数(ETH 762 / BTC 716)
- 对每币跑 两档胜率: 实测 p 与悲观 p-0.04
- 向量化 8000 条路径
用法: python probe_kelly_mc.py
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
FRACS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15)


def load_pmean(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(float) / (n - 1)
    return R.mean(axis=0)


def greedy_sparse(ts, gap=1800):
    """每30min最多1单: 贪心保留间距>=gap 的信号(滚动窗口首尾相接=独立事件)。"""
    keep = np.zeros(len(ts), bool)
    last = -10**18
    for i, t in enumerate(ts):
        if t >= last + gap:
            keep[i] = True
            last = t
    return keep


def n_signals(symbol):
    ctx = AssetContext(symbol, horizon=30)
    pmv = load_pmean(symbol, "meta_val"); pt = load_pmean(symbol, "test")
    ts_test = ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64)
    seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
    r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8), ts_test, seed_conf)
    return int(greedy_sparse(ts_test[r["sel"]]).sum())


def mc(n, p, f):
    """返回 (期末中位数, MDD中位数, MDD p90, 期末<50%本金概率, 期末<10%概率)"""
    wins = np.random.default_rng(7).random((N_SIM, n)) < p          # (N_SIM, n)
    g = np.where(wins, 1.0 + BAY * f, 1.0 - f)                      # 每注权益乘子
    eq = INIT * np.cumprod(g, axis=1)                               # 复利路径
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = 1.0 - eq / peak
    fin = eq[:, -1]
    return (np.median(fin), np.median(mdd), np.quantile(mdd, 0.90),
            float((fin < 0.5 * INIT).mean()), float((fin < 0.1 * INIT).mean()))


def main():
    for s in ("ETH", "BTC"):
        n = n_signals(s)
        print(f"\n######## {s}: 30min窗口信号数={n} ########", flush=True)
        # 实测胜率(去重后) 由确定性脚本给出: ETH 0.6220 / BTC 0.6061; 此处重新算
        ctx = AssetContext(s, horizon=30)
        pmv = load_pmean(s, "meta_val"); pt = load_pmean(s, "test")
        ts_test = ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8), ts_test, seed_conf)
        sel = r["sel"]; ysel = ctx.y("test")[sel]; pred = r["pred"]; ts = ts_test[sel]
        # 修正: 滚动30min窗口(相邻信号重叠29min), 仅间隔>=30min为独立事件(与 sim 口径一致),
        # 不再用固定桶 ts//1800(那会把07:30~07:59误当同一事件)
        keep = greedy_sparse(ts)
        p_real = float((pred[keep] == ysel[keep]).mean())
        for tag, p in (("实测", p_real), ("悲观-4pp", p_real - 0.04)):
            print(f"  --- 胜率 {tag} = {p:.4f} (盈亏平衡 0.5405) ---", flush=True)
            print("   f    | 期末中位 | MDD中位 | MDD p90 | 期末<50% | 期末<10%", flush=True)
            for f in FRACS:
                fin, mdd50, mdd90, r50, r10 = mc(n, p, f)
                print(f"  {f:5.1%} | {fin:8.0f} | {mdd50:6.1%} | {mdd90:6.1%} | {r50:7.1%} | {r10:6.1%}", flush=True)


if __name__ == "__main__":
    main()
