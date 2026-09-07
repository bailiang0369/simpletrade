#!/usr/bin/env python3
"""资金管理模拟: 币安事件合约(30min涨跌, 赢返0.85) 在 test 段真实信号序列上的
不同下注比例(占当前权益)的 期末权益/最大回撤/收益回撤比。

- 胜率/信号序列: 生产 JOINT 模型 R2 协议选出的 top1% 真实 pred vs label
- 信号重叠: 1分钟级滚动30min窗口, 相邻信号重叠29min; 按间隔>=30min贪心稀疏化
  (greedy_sparse, 每30min最多1注=首尾相接的独立事件), 不按固定周期桶去重
- 每注下注额 = f * 当前权益(复利); 赢 +0.85*bet, 输 -bet
用法: python probe_kelly_sim.py
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
BAY = 0.85          # 赢的净返奖倍数
INIT = 1000.0
FRACS = (0.02, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20)


def load_pmean(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(float) / (n - 1)
    return R.mean(axis=0)


def sim(w, f):
    eq = INIT; peak = INIT; mdd = 0.0
    for rw in w:
        bet = f * eq
        eq += (BAY if rw else -1.0) * bet
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
    return eq, mdd


def greedy_sparse(ts, win, gap=1800):
    """按时间顺序贪心: 每次开单后跳过 gap(30min), 保留信号间距>=gap -> 窗口首尾相接, 近似独立事件。"""
    keep = np.zeros(len(ts), bool)
    last = -10**18
    for i, t in enumerate(ts):
        if t >= last + gap:
            keep[i] = True
            last = t
    return win[keep]


def main():
    for s in ("ETH", "BTC"):
        ctx = AssetContext(s, horizon=30)
        pmv = load_pmean(s, "meta_val"); pt = load_pmean(s, "test")
        ts_test = ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        conf = np.abs(pt - 0.5) * 2
        r = ESM.r2_eval(conf, (pt >= 0.5).astype(np.int8), ts_test, seed_conf)
        sel = r["sel"]; ysel = ctx.y("test")[sel]; pred = r["pred"]; ts = ts_test[sel]
        win = (pred == ysel).astype(int)
        w = greedy_sparse(ts, win, 1800)                 # 每30min最多1单(币安约束), 独立事件
        p = w.mean()
        f_kelly = (BAY * p - (1 - p)) / BAY
        print(f"\n######## {s} test 模拟(修正: 滚动30min窗口, 每30min限1单) ########", flush=True)
        print(f"  信号: 总 {len(win)} 笔(1min级,相邻重叠29min), 30min间隔稀疏化后 {len(w)} 笔独立事件, 胜率 {p:.4f}", flush=True)
        print(f"  盈亏平衡胜率 = {1/(1+BAY):.4f}   单注期望 = {BAY*p-(1-p):+.4f} (每1U)", flush=True)
        print(f"  full Kelly f* = {f_kelly:.1%}   half Kelly = {f_kelly/2:.1%}", flush=True)
        print("  下注比例f | 期末权益 | 最大回撤 | 收益/回撤 | 期末/期初", flush=True)
        for f in FRACS + (f_kelly, f_kelly / 2):
            eq, mdd = sim(w, f)
            print(f"  {f:6.1%} | {eq:8.0f} | {mdd:7.1%} | {eq/INIT/mdd:6.1f}x | {eq/INIT:6.2f}x", flush=True)


if __name__ == "__main__":
    main()
