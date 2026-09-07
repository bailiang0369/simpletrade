#!/usr/bin/env python3
"""Phase1-步骤5: 信号独立性筛选 + 信号落盘 (人工跟单用)。

职责:
  1. IndependenceFilter: per-(symbol,horizon) 状态机, 保证新信号与上一个独立信号
     时间间隔 >= Nmin(分钟)。这是"滚动窗口重叠"下把相邻 1min 信号拆成独立事件的关键
     (对应离线 greedy_sparse 语义)。
  2. emit_signal: 命中即写 live/signals_recent.json(最新) + live/signals.jsonl(追加)。

格式:
  {ts: UTC秒, iso: UTC ISO串, symbol, horizon, direction: 1/0(p>0.5看涨), p, conf, thresh}
"""
import json
import os
import time
from datetime import datetime, timezone

LIVE_DIR = os.path.dirname(os.path.abspath(__file__))
RECENT = os.path.join(LIVE_DIR, "signals_recent.json")
JSONL = os.path.join(LIVE_DIR, "signals.jsonl")

NMIN = {30: 30, 60: 60}   # 每周期独立性最小间隔(分钟), 对应 freeze_bundle.NMIN / greedy_sparse


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class IndependenceFilter:
    """按 (symbol,horizon) 独立跟踪上次发信号时刻; 间隔不足 Nmin 则丢弃。"""

    def __init__(self, nmin=NMIN):
        self._last = {}           # (symbol,horizon) -> ts(秒)
        self._nmin = nmin

    def accept(self, symbol, horizon, ts):
        """返回 True 表示该信号可作为独立事件; False 表示跳过。"""
        key = (symbol, horizon)
        gap_min = NMIN.get(horizon, horizon)
        last = self._last.get(key)
        if last is not None and (ts - last) < gap_min * 60:
            return False
        self._last[key] = ts
        return True


def flush_recent(records):
    """records: 本次 run_once 全部(最多 4 条)已确认信号, 覆盖写 recent。"""
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(RECENT, "w", encoding="utf-8") as f:
        json.dump({"updated_iso": _iso(time.time()), "signals": records},
                  f, ensure_ascii=False, indent=2)


def append_jsonl(record):
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def emit_signal(symbol, horizon, ts, p, conf, thresh):
    """单个已独立信号 -> 落盘并返回 dict。"""
    rec = {
        "ts": int(ts),
        "iso": _iso(ts),
        "symbol": symbol,
        "horizon": horizon,
        "direction": 1 if p >= 0.5 else 0,
        "p": round(float(p), 6),
        "conf": round(float(conf), 6),
        "thresh": None if thresh is None else round(float(thresh), 6),
    }
    append_jsonl(rec)
    return rec


def greedy_sparse(times, gap_min):
    """离线等价: 贪心挑选(按时间序)间隔 >= gap_min 的独立事件, 返回选中索引。
    times: 已升序的秒级时间戳; gap_min: 分钟。"""
    sel = []
    last = None
    for i, t in enumerate(times):
        if last is None or (t - last) >= gap_min * 60:
            sel.append(i)
            last = t
    return sel