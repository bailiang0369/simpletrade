#!/usr/bin/env python3
"""实证验证 label 定义: 每个1分钟行预测的是"以该行为起点的未来30分钟"滚动窗口,
不是固定30min周期(如 07:30-08:00)。
打印 test 段一段连续分钟: ts、预测窗口 [ts, ts+30min]、close[ts]、close[ts+30]、涨跌、label。
用法: python probe_verify_window.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext


def fmt(s):
    return str(np.datetime64(int(s), "s"))


def main():
    ctx = AssetContext("ETH", horizon=30)
    m = ctx.split_rows["test"]
    ts = ctx.ds_ts[m].astype(np.int64)
    pos = ctx.ds_to_raw[m]
    lab = ctx.label[m]

    # 从 test 段起点找一段分钟连续的 ds 行(中间无剔除缺口)
    n = len(ts)
    start = None
    for i in range(n - 10):
        if ts[i + 1] - ts[i] == 60 and ts[i + 2] - ts[i + 1] == 60 and ts[i + 3] - ts[i + 2] == 60:
            start = i
            break
    if start is None:
        print("未找到连续4分钟的区段")
        return

    print(f"######## ETH test 段连续 {start}~{start+3} 分钟 (ds行) ########", flush=True)
    print(f"{'ds行':>4} | {'信号时间(UTC)':>17} | {'预测窗口 [t, t+30min]':>26} | {'close[t]':>9} | "
          f"{'close[t+30]':>11} | {'30min涨跌':>9} | label", flush=True)
    for i in range(start, start + 4):
        p = pos[i]
        p30 = p + 30
        r = (ctx.c[p30] - ctx.c[p]) / ctx.c[p] * 100.0
        print(f"{i:>4} | {fmt(ts[i]):>17} | [{fmt(ts[i])}, {fmt(ts[i]+1800)}] | "
              f"{ctx.c[p]:>9.1f} | {ctx.c[p30]:>11.1f} | {r:>+8.2f}% | {int(lab[i])}", flush=True)

    print("\n结论: 每行的预测窗口都以该行 ts 为起点后推30min(滚动), 相邻行窗口只错开1分钟,", flush=True)
    print("      并非都指向同一个固定周期末(如 08:00)。label=1 表示 close[t+30]>close[t]。", flush=True)


if __name__ == "__main__":
    main()
