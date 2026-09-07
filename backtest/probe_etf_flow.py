#!/usr/bin/env python3
"""ETF 资金流探针: "上一交易日 BTC ETF 净流入" 能否区分 当日30min 上涨率(base rate)?

结构性问题背景: train 段(2020-01~2024-06-30)几乎无 ETF 数据(BTC ETF 2024-01-11 上市,
train 尾部仅~10%覆盖; ETH ETF 2024-07-23 上市,train 段完全不存在)。
因此模型无法从训练中学到 ETF->收益映射; 即使特征可加, 也是日频慢变量,
只能移动"整日 base rate", 不能区分日内哪根30min涨。
本探针直接验证必要条件: 2024+ 数据中, 前一日流入方向/大小 vs 当日30min上涨率是否有区分度。

无泄漏: 对某 UTC 日的样本, 只用"严格早于该日"的最后一个 ETF 交易日数据
(ETF 净流入于美盘收盘后公布, "早于当日"是保守安全口径)。

用法: python probe_etf_flow.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os, json
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_store import AssetContext

ETF_JSON = "/tmp/btc_etf.json"      # tftc.io 抓取的 BTC ETF 日净流入
# 有"前一日流"的最早 UTC 日(2024-01-11 上市) -> epoch day
MIN_DATE = int(datetime(2024, 1, 12, tzinfo=timezone.utc).timestamp()) // 86400


def load_flow():
    d = json.load(open(ETF_JSON))
    days = []
    for x in d["days"]:
        ts = datetime.fromisoformat(x["date"]).replace(tzinfo=timezone.utc)
        ed = int(ts.timestamp()) // 86400
        f = x.get("netFlowUsd")
        if f is None:
            continue
        days.append((ed, float(f)))
    days.sort()
    eds = np.array([a for a, _ in days], dtype=np.int64)
    flows = np.array([b for _, b in days], dtype=np.float64)
    return eds, flows


def day_stats(symbol, eds, flows):
    """对目标币: 每个 UTC 日(有前日流)的 30min 上涨率/平均收益/样本数。"""
    ctx = AssetContext(symbol, horizon=30)
    ts = ctx.ds_ts
    lab = ctx.label.astype(np.float64)
    rf = ctx.ret_future.astype(np.float64)
    # 仅取 ETF 时代(有前日流)样本: 样本 UTC 日 > 首个 ETF 日
    day = ts // 86400
    sel = day >= MIN_DATE
    day = day[sel]; lab = lab[sel]; rf = rf[sel]; ts = ts[sel]
    prev_day = np.searchsorted(eds, day, side="left") - 1   # 严格早于当日的最后 ETF 日
    ok = prev_day >= 0
    prev_day = prev_day[ok]; day = day[ok]; lab = lab[ok]; rf = rf[ok]
    flow = flows[prev_day]
    # 按 UTC 日聚合
    u, inv = np.unique(day, return_inverse=True)
    n = len(u)
    up = np.zeros(n); ret = np.zeros(n); cnt = np.zeros(n)
    np.add.at(up, inv, lab); np.add.at(ret, inv, rf); np.add.at(cnt, inv, 1.0)
    up /= np.maximum(cnt, 1); ret /= np.maximum(cnt, 1)
    # 对应每日的"前日流"
    fday = np.array([flows[np.searchsorted(eds, d, side="left") - 1] for d in u])
    return u, up, ret, cnt, fday


def report(symbol, u, up, ret, cnt, fday):
    print(f"\n===== {symbol} (ETF 时代 {u.min()}-{u.max()}, 天数={len(u)}) =====", flush=True)
    # 方向分组: 前日流入 vs 流出
    pos = fday > 0; neg = fday < 0
    print(f"  全样本: 日均上涨率 {up.mean():.4f}  日均收益 {ret.mean()*100:.4f}%  日样本数均值 {cnt.mean():.0f}")
    for name, m in (("前日流入", pos), ("前日流出", neg)):
        if m.sum() == 0:
            print(f"  [{name}] 无样本"); continue
        print(f"  [{name}] 天数={m.sum():3d}  日均上涨率 {up[m].mean():.4f}  日均收益 {ret[m].mean()*100:+.4f}%", flush=True)
    if pos.sum() and neg.sum():
        print(f"  >>> 流入-流出 上涨率差 {up[pos].mean()-up[neg].mean():+.4f}  收益差 {(ret[pos].mean()-ret[neg].mean())*100:+.4f}%", flush=True)
    # 大小分档(三分位)
    q1, q2 = np.quantile(fday, [1/3, 2/3])
    for name, m in (("前日大幅流出", fday <= q1), ("前日中性", (fday > q1) & (fday <= q2)), ("前日大幅流入", fday > q2)):
        if m.sum() == 0:
            continue
        print(f"  [{name}] 阈值≤{q1:.0f}/{q2:.0f}  天数={m.sum():3d}  日均上涨率 {up[m].mean():.4f}  日均收益 {ret[m].mean()*100:+.4f}%", flush=True)
    # 极端日: |流| > 90 分位
    q90 = np.quantile(np.abs(fday), 0.9)
    big = np.abs(fday) > q90
    print(f"  [|流|>P90={q90:.0f}] 天数={big.sum():3d}  日均上涨率 {up[big].mean():.4f}  日均收益 {ret[big].mean()*100:+.4f}%", flush=True)
    # 逐月流入/流出上涨率差(稳健性)
    month = u // 100
    print("  逐月(流入日上涨率 - 流出日上涨率):")
    line = []
    for mm in np.unique(month):
        m = (month == mm) & pos; n = (month == mm) & neg
        if m.sum() >= 5 and n.sum() >= 5:
            d = up[m].mean() - up[n].mean()
            line.append(f"{mm}:{d:+.3f}({m.sum()}/{n.sum()})")
    print("   " + "  ".join(line), flush=True)


def main():
    eds, flows = load_flow()
    print(f"ETF 数据: {len(eds)} 个交易日  {str(np.datetime64(int(eds[0])*86400,'s').astype('datetime64[D]'))} ~ {str(np.datetime64(int(eds[-1])*86400,'s').astype('datetime64[D]'))}", flush=True)
    print(f"净流入 正/负: {(flows>0).sum()}/{(flows<0).sum()}  中位 {np.median(flows):.0f} M", flush=True)
    for s in ("BTC", "ETH"):
        u, up, ret, cnt, fday = day_stats(s, eds, flows)
        report(s, u, up, ret, cnt, fday)


if __name__ == "__main__":
    main()
