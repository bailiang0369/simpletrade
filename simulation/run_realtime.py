#!/usr/bin/env python3
"""实时模拟(模拟盘): 接入币安实时 1min K线, 复用实盘信号引擎, 只产信号不下单。

与 live/run_once.py 的区别: 数据源从离线 raw parquet 换成币安公开行情 REST
(data-api.binance.vision/api/v3/klines, 官方公开端点, 无需 API key),
其余(特征 compute_X -> 池推理 SessionRankFuser -> 滚动阈值 RollingThreshold
-> 独立性 IndependenceFilter -> 落盘)完全复用 live/ 同一套代码。

落盘位置: 输出重定向到 simulation/ (signals_recent.json + signals.jsonl),
不触碰 live/ 的实盘信号文件。独立性状态持久化到 .indep_state.json,
保证 cron/多次触发之间 30/60min 独立间隔仍然生效。

用法:
  python simulation/run_realtime.py             # 跑一次(适合 cron 每分钟/每5分钟触发)
  python simulation/run_realtime.py --watch 60  # 常驻: 每 60s tick 一次
  python simulation/run_realtime.py --limit 1500
"""
import argparse
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(ROOT, "live")
for p in (ROOT, LIVE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from features_online import compute_X, bars_frame
from predict import SessionRankFuser, RollingThreshold
import signals as sig_mod
from signals import IndependenceFilter, emit_signal, flush_recent

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLES_DIR = os.path.join(LIVE, "bundles")
SYMBOLS = ["ETH", "BTC"]
HORIZONS = [30, 60]
WARMUP = 480          # 特征/自Z/滚动窗需要的最短 bar 数
LOOKBACK = 1500       # 每个预测时刻向前取多少根 bar(含跨资产), 覆盖全部最长滚窗
BINANCE_API = "https://data-api.binance.vision/api/v3/klines"

# 输出重定向: 模拟盘信号写 simulation/, 与实盘 live/ 隔离
sig_mod.RECENT = os.path.join(SIM_DIR, "signals_recent.json")
sig_mod.JSONL = os.path.join(SIM_DIR, "signals.jsonl")
INDEP_STATE = os.path.join(SIM_DIR, ".indep_state.json")


# ---------- 币安实时 kline ----------
def _get(url, retries=3):
    """带重试的 GET(沙箱代理偶发 SSL EOF, 重试即恢复)。"""
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    raise last


def fetch_klines(symbol, limit=LOOKBACK):
    """拉取最近 limit 根 1min kline -> polars bars df(与 raw parquet 同构)。

    币安 kline 数组: [openTime(ms), o, h, l, c, volume, closeTime, quoteVol,
    trades, takerBuyBase, takerBuyQuote, ignore]。buy_vol=takerBuyBase(主动买),
    sell_vol=volume-takerBuyBase(主动卖); funding 实时无数据置 0。
    币安单次最多返回 1000 根, limit>1000 时分两段(endTime)拼接。
    只保留已收盘 bar(openTime+60s <= now), 与"只看到 t 及以前"语义一致。
    """
    raw = _get(f"{BINANCE_API}?symbol={symbol}USDT&interval=1m&limit={min(limit, 1000)}")
    if limit > 1000 and raw:
        end = raw[0][0] - 1                       # 更早段的截止时刻(ms)
        raw = _get(f"{BINANCE_API}?symbol={symbol}USDT&interval=1m"
                   f"&limit={min(limit - len(raw), 1000)}&endTime={end}") + raw
    if not raw:
        return None
    now_ms = int(time.time() * 1000)
    rows = [k for k in raw if k[0] + 60_000 <= now_ms]      # 丢弃未收盘 bar
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    ts = (arr[:, 0].astype(np.int64) // 1000)               # ms -> 秒
    buy = arr[:, 9]
    vol = arr[:, 5]
    return bars_frame(ts, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4],
                      buy, np.maximum(vol - buy, 0.0), funding=None)


def _with_warmup(bars, n=LOOKBACK):
    if bars is None or len(bars) < WARMUP:
        return None
    return bars[-n:]


# ---------- 独立性状态持久化(cron 多次触发间保持 30/60min 间隔) ----------
def _load_indep_state():
    if os.path.exists(INDEP_STATE):
        try:
            st = json.load(open(INDEP_STATE))
            return {tuple(k.split("|")): v for k, v in st.items()}
        except Exception:
            pass
    return {}


def _save_indep_state(last_map):
    st = {"|".join(k): v for k, v in last_map.items()}
    with open(INDEP_STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


# ---------- 单次 tick: 拉实时数据 -> 计算信号 -> 落盘 ----------
def tick(limit=LOOKBACK):
    bars = {}
    win = {}
    for s in SYMBOLS:
        try:
            bars[s] = fetch_klines(s, limit=limit)
        except Exception as e:
            print(f"[realtime] {s}: 拉取失败 {e}", flush=True)
            bars[s] = None
        win[s] = _with_warmup(bars[s])
        if win[s] is None:
            print(f"[realtime] 跳过 {s}: bars 不足 {WARMUP}", flush=True)
    if all(w is None for w in win.values()):
        return 0

    indep = IndependenceFilter()
    indep._last = _load_indep_state()
    # 统一当前时刻: 两币最新已收盘 bar 的 ts 较小者
    cur_ts = min(int(win[s]["ts"].to_numpy()[-1])
                 for s in SYMBOLS if win[s] is not None)

    emitted = []
    for symbol in SYMBOLS:
        if win[symbol] is None:
            continue
        for horizon in HORIZONS:
            pool_dir = os.path.join(BUNDLES_DIR, f"h{horizon}")
            probe = os.path.join(pool_dir, "JOINT_lgb_seed42.txt")
            if not os.path.isfile(probe):
                print(f"[realtime] 跳过 {symbol}/h{horizon}: 无权重 {probe}", flush=True)
                continue
            seed_files = [f for f in os.listdir(pool_dir)
                          if f.startswith("seed_conf") and symbol in f]
            seed_conf = []
            if seed_files:
                seed_conf = json.load(open(os.path.join(pool_dir, seed_files[0])))
            fuser = SessionRankFuser(pool_dir)
            thr = RollingThreshold(seed_conf)
            other = "ETH" if symbol == "BTC" else "BTC"
            if win[other] is None:
                print(f"[realtime] 跳过 {symbol}/{horizon}: 对手盘 {other} 数据不足", flush=True)
                continue
            X = compute_X(win[symbol], win[other], symbol)   # [n, 58]

            last_p, last_conf = None, None
            for i in range(max(0, len(X) - WARMUP), len(X)):
                last_p, last_conf = fuser.fused_at(X[i:i + 1].reshape(1, -1))
            p, conf = last_p, last_conf
            tau = thr.threshold()
            tag = f"{symbol}/h{horizon}"
            if tau is None:
                print(f"[realtime] {tag}: 冷启动, 不发信号", flush=True)
                continue
            print(f"[realtime] {tag}: p={p:.4f} conf={conf:.4f} tau={tau:.4f}", flush=True)
            if conf < tau:
                continue
            if not indep.accept(symbol, horizon, cur_ts):
                print(f"[realtime] {tag}: 独立性间隔不足, 丢弃", flush=True)
                continue
            rec = emit_signal(symbol, horizon, cur_ts, p, conf, tau)
            emitted.append(rec)
            print(f"[realtime] SIGNAL {tag} t={rec['iso']} dir={rec['direction']} "
                  f"p={rec['p']} conf={rec['conf']} thresh={tau:.4f}", flush=True)

    _save_indep_state(indep._last)
    flush_recent(emitted)
    print(f"[realtime] done, {len(emitted)} signal(s) written", flush=True)
    return len(emitted)


def main():
    ap = argparse.ArgumentParser(description="模拟盘: 币安实时K线 -> 预测信号(不下单)")
    ap.add_argument("--watch", type=int, default=0,
                    help="常驻模式: 每 N 秒 tick 一次; 默认 0 = 跑一次即退出")
    ap.add_argument("--limit", type=int, default=LOOKBACK,
                    help=f"每次拉取最近多少根 1min bar (默认 {LOOKBACK})")
    a = ap.parse_args()
    if a.limit < WARMUP:
        a.limit = WARMUP
    if a.watch and a.watch <= 0:
        a.watch = 0
    tick(limit=a.limit)
    while a.watch:
        time.sleep(a.watch)
        tick(limit=a.limit)


if __name__ == "__main__":
    main()
