#!/usr/bin/env python3
"""方案3: 统计时序基线探底。检验 H=15 方向可预测性的下限。

对每个单变量信号, 计算其在"信号强"样本上的方向命中率 vs 无条件基准 (base)。
重点看能不能显著超过 base, 及其随信号强度/置信度分档的变化。
纯向量化, 秒级完成。
"""
import os, sys
sys.path.insert(0, "/workspace")
import numpy as np
import pandas as pd
import config

H = 15
SYMBOLS = ["ETH", "BTC"]


def load(sym):
    raw = pd.read_parquet(f"{config.DS_DIR}/raw_{sym}.parquet")
    c = raw["close"].to_numpy(np.float64)
    buy = raw["buy_vol"].to_numpy(np.float64)
    sell = raw["sell_vol"].to_numpy(np.float64)
    fnd = raw["funding"].to_numpy(np.float64)
    return c, buy, sell, fnd


def probe(sym):
    c, buy, sell, fnd = load(sym)
    n = len(c)
    # 未来 H 方向
    future_lr = np.full(n, np.nan)
    future_lr[:n - H] = np.log(c[H:] / c[:-H])
    y = (future_lr > 0).astype(np.float64)
    base = np.nanmean(y)
    # 1-min 收益
    lr1 = np.full(n, np.nan); lr1[1:] = np.log(c[1:] / c[:-1])
    # 各窗口累计收益
    def cumret(w):
        r = np.full(n, np.nan); r[w:] = np.log(c[w:] / c[:-w]); return r
    # 波动 (收益绝对值的滚动均方根)
    def rvol(w):
        m = np.full(n, np.nan)
        sq = lr1 ** 2
        m[w:] = np.sqrt(np.convolve(np.nan_to_num(sq), np.ones(w)/w, mode='full')[:n])[w:]
        return m
    signals = {
        "mom_15": cumret(15), "mom_30": cumret(30), "mom_60": cumret(60), "mom_120": cumret(120),
    }
    # 均值回复: 价格相对短均线的 z (负=低于均线, 正=高于)
    for w in [30, 60, 120, 240]:
        ma = np.convolve(c, np.ones(w)/w, mode='full')[:n]
        # 用 NaN shift 对齐: ma 前 w 个不准
        z = (c - ma) / (ma + 1e-9)
        signals[f"z_{w}"] = z
    # 波动集聚: 高波动后方向
    for w in [15, 30, 60]:
        signals[f"rvol_{w}"] = rvol(w)
    # 买卖压力
    vlb = buy / (sell + 1e-9)
    vlb_norm = vlb / (np.abs(c).mean()*0+1)
    for w in [15, 60]:
        vlb_w = np.convolve(np.nan_to_num(vlb), np.ones(w)/w, mode='full')[:n]
        signals[f"vlb_{w}"] = vlb_w

    print(f"\n===== {sym} H={H} 统计探底 | base(无条件上涨)={base*100:.2f}% =====")
    print(f"样本 {n} | 信号分两端(强端/弱端)看命中率")
    import datetime as dt
    # 只看训练段 (<=2024-06-30) 之后的测试行为? 探底用全段即可
    for name, s in signals.items():
        m = ~np.isnan(s) & ~np.isnan(y)
        s = s[m]; yy = y[m]
        up_acc = (yy == 1).mean() * 100  # 全样本
        # 强信号端 (s 最大 5%)
        k = max(1, int(len(s) * 0.05))
        hi = np.argsort(-s)[:k]
        lo = np.argsort(s)[:k]
        hi_acc = yy[hi].mean() * 100          # 强信号下涨率
        lo_acc = yy[lo].mean() * 100          # 弱(负)信号下涨率 (跌率=100-lo)
        hi_dn = 100 - hi_acc                  # 强信号下跌率
        # 偏离度: max离base
        dev = max(abs(hi_acc-base*100), abs(lo_acc-base*100))
        flag = " >" if dev > 3 else ""
        print(f"  {name:<10}: 全:{up_acc:5.2f}% | 强端涨:{hi_acc:5.2f}% 强端跌:{hi_dn:5.2f}% | 弱端涨:{lo_acc:5.2f}% | 偏离base:{dev:4.1f}pp{flag}")


if __name__ == "__main__":
    for s in SYMBOLS:
        probe(s)