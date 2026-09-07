#!/usr/bin/env python3
"""实时模拟(模拟盘): WebSocket 订阅币安实时 1min K线, 复用实盘信号引擎, 只产信号不下单。

数据源(实时): wss://data-stream.binance.vision:9443 组合流
  btcusdt@kline_1m + ethusdt@kline_1m (币安官方公开数据流, 无需 API key)。
  每收到一根"已收盘"(k.x=true) 的 1min bar 即增量触发该币信号计算。
启动时仅用 REST (data-api.binance.vision/api/v3/klines) 一次性回填历史 warmup
  (特征窗口必需), 之后全程走 WebSocket 推送, 不做轮询。

信号流水线: 特征 compute_X -> 池推理 SessionRankFuser -> 滚动阈值 RollingThreshold
-> 独立性 IndependenceFilter -> 落盘, 与 live/run_once.py 完全同一套代码, 不下单。

落盘: simulation/signals_recent.json + signals.jsonl (与实盘 live/ 隔离);
独立性状态持久化 .indep_state.json, 进程重启后 30/60min 间隔仍生效。

用法:
  python simulation/run_realtime.py             # WebSocket 常驻, 收盘 bar 驱动
  python simulation/run_realtime.py --limit 1500  # 冷启动回填 bar 数
"""
import argparse
import json
import os
import sys
import time
import urllib.parse

import requests
import websocket

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
LOOKBACK = 1500       # 预测时刻向前取的 bar 数(含跨资产), 覆盖全部最长滚窗
REST_API = "https://data-api.binance.vision/api/v3/klines"
WS_URL = "wss://data-stream.binance.vision:9443/stream"
STREAM_SYMBOL = {"ETH": "ethusdt", "BTC": "btcusdt"}

# 输出重定向: 模拟盘信号写 simulation/, 与实盘 live/ 隔离
sig_mod.RECENT = os.path.join(SIM_DIR, "signals_recent.json")
sig_mod.JSONL = os.path.join(SIM_DIR, "signals.jsonl")
INDEP_STATE = os.path.join(SIM_DIR, ".indep_state.json")

BARS = {}          # symbol -> polars bars df(时间升序, 启动回填后由 WS 增量 append)


# ---------- 冷启动回填(仅启动时一次性, 用 REST) ----------
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


def bootstrap(limit=LOOKBACK):
    """REST 拉最近 limit 根已收盘 1min bar 作 warmup, 返回 {symbol: df}。

    币安 kline: [openTime(ms), o, h, l, c, volume, closeTime, quoteVol, trades,
    takerBuyBase, ...]; buy_vol=takerBuyBase(主动买), sell_vol=volume-buy(主动卖);
    funding 实时无数据置 0。单次最多 1000 根, limit>1000 分两段(endTime)拼接。
    """
    out = {}
    for s in SYMBOLS:
        raw = _get(f"{REST_API}?symbol={s}USDT&interval=1m&limit={min(limit, 1000)}")
        if limit > 1000 and raw:
            end = raw[0][0] - 1
            raw = _get(f"{REST_API}?symbol={s}USDT&interval=1m"
                       f"&limit={min(limit - len(raw), 1000)}&endTime={end}") + raw
        if not raw:
            out[s] = None
            continue
        now_ms = int(time.time() * 1000)
        rows = [k for k in raw if k[0] + 60_000 <= now_ms]   # 只留已收盘 bar
        arr = np.asarray(rows, dtype=np.float64)
        buy = arr[:, 9]
        vol = arr[:, 5]
        out[s] = bars_frame((arr[:, 0].astype(np.int64) // 1000),
                            arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4],
                            buy, np.maximum(vol - buy, 0.0), funding=None)
        print(f"[bootstrap] {s}: {len(out[s])} bars ts[{arr[:,0][0]}..{arr[:,0][-1]}]", flush=True)
    return out


# ---------- WS 收盘 bar 增量接入 ----------
def _append_closed_bar(symbol, k):
    """把一根已收盘 1min bar(k: kline 事件内的 k 对象)append 到缓存并裁剪。"""
    ts = int(k["t"]) // 1000
    df = BARS.get(symbol)
    if df is not None and len(df) and int(df["ts"].to_numpy()[-1]) >= ts:
        return df                                   # 重复/乱序事件, 忽略
    row = bars_frame(np.array([ts], dtype=np.int64),
                     np.array([float(k["o"])]), np.array([float(k["h"])]),
                     np.array([float(k["l"])]), np.array([float(k["c"])]),
                     np.array([float(k["V"])]),          # takerBuyBase = 主动买
                     np.array([max(float(k["v"]) - float(k["V"]), 0.0)]),
                     funding=None)
    df = row if df is None else df.vstack(row)
    if len(df) > LOOKBACK + 5:
        df = df.tail(LOOKBACK + 5)
    BARS[symbol] = df
    return df


# ---------- 信号计算(与 live/run_once.py 同一套引擎) ----------
_flows = {}        # (symbol,horizon) -> pool_dir(权重存在才可用)


def _setup_flows():
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            pool_dir = os.path.join(BUNDLES_DIR, f"h{horizon}")
            probe = os.path.join(pool_dir, "JOINT_lgb_seed42.txt")
            if not os.path.isfile(probe):
                print(f"[realtime] 跳过 {symbol}/h{horizon}: 无权重 {probe}", flush=True)
                continue
            _flows[(symbol, horizon)] = pool_dir


def compute_signals(symbol):
    """对某币最新收盘 bar 时刻, 逐 horizon 算信号; 命中独立信号则落盘。"""
    self_df = BARS.get(symbol)
    if self_df is None or len(self_df) < WARMUP:
        return
    other = "ETH" if symbol == "BTC" else "BTC"
    other_df = BARS.get(other)
    if other_df is None or len(other_df) < WARMUP:
        return
    cur_ts = int(self_df["ts"].to_numpy()[-1])
    indep = IndependenceFilter()
    indep._last = _load_indep_state()
    emitted = []
    for (sym, horizon), pool_dir in list(_flows.items()):
        if sym != symbol:
            continue
        # 每次事件重建 fuser/threshold(与 run_once 一致): 窗口尾部喂入取最后一根
        seed_files = [f for f in os.listdir(pool_dir)
                      if f.startswith("seed_conf") and symbol in f]
        seed_conf = []
        if seed_files:
            seed_conf = json.load(open(os.path.join(pool_dir, seed_files[0])))
        fuser = SessionRankFuser(pool_dir)
        thr = RollingThreshold(seed_conf)
        X = compute_X(self_df.tail(LOOKBACK), other_df.tail(LOOKBACK), symbol)
        last_p, last_conf = None, None
        for i in range(max(0, len(X) - WARMUP), len(X)):
            last_p, last_conf = fuser.fused_at(X[i:i + 1].reshape(1, -1))
        p, conf = last_p, last_conf
        tau = thr.threshold()
        tag = f"{symbol}/h{horizon}"
        if tau is None:
            print(f"[realtime] {tag}: 冷启动, 不发信号", flush=True)
            continue
        print(f"[realtime] {tag} t={cur_ts}: p={p:.4f} conf={conf:.4f} tau={tau:.4f}",
              flush=True)
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
    if emitted:
        flush_recent(emitted)


# ---------- 独立性状态持久化(进程重启间保持 30/60min 间隔) ----------
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


# ---------- WS 连接(组合流) ----------
def _proxy_from_env():
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        raw = os.environ.get(key)
        if raw:
            u = urllib.parse.urlparse(raw)
            if u.hostname:
                return u.hostname, (u.port or 80)
    return None, None


def run_ws():
    streams = "/".join(f"{STREAM_SYMBOL[s]}@kline_1m" for s in SYMBOLS)
    url = f"{WS_URL}?streams={streams}"
    phost, pport = _proxy_from_env()
    print(f"[realtime] connect {url}", flush=True)
    ws = websocket.create_connection(url, http_proxy_host=phost,
                                     http_proxy_port=pport, timeout=30)
    last_ping = time.time()

    def on_message(msg):
        try:
            data = json.loads(msg)
            body = data["data"] if "data" in data else data   # 组合流/单流兼容
            if body.get("e") != "kline":
                return
            k = body["k"]
            if not k.get("x"):                                # 只处理已收盘 bar
                return
            symbol = k["s"].rstrip("USDT").upper()
            if symbol not in BARS:
                return
            _append_closed_bar(symbol, k)
            compute_signals(symbol)
        except Exception as e:
            print(f"[realtime] on_message error: {e}", flush=True)

    try:
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                ws.ping()                                   # 空闲探活(币安20s无数据会断开)
                last_ping = time.time()
                continue
            if msg is None:                                  # 服务端关闭
                print("[realtime] ws closed by server", flush=True)
                break
            on_message(msg)
            if time.time() - last_ping > 20:
                ws.ping()
                last_ping = time.time()
    except Exception as e:
        print(f"[realtime] ws error: {e}", flush=True)
    finally:
        try:
            ws.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="模拟盘: WebSocket 实时K线 -> 预测信号(不下单)")
    ap.add_argument("--limit", type=int, default=LOOKBACK,
                    help=f"冷启动回填 bar 数 (默认 {LOOKBACK})")
    a = ap.parse_args()
    limit = max(a.limit, WARMUP)

    # 1) 启动回填(仅一次): 历史 warmup -> 覆盖最长特征窗
    BARS.update(bootstrap(limit=limit))
    if any(BARS.get(s) is None or len(BARS[s]) < WARMUP for s in SYMBOLS):
        print("[realtime] 冷启动数据不足, 退出", flush=True)
        sys.exit(1)

    # 2) 预检可用模型流(h30 已有权重; h60 无权重则跳过)
    _setup_flows()
    if not _flows:
        print("[realtime] 无任何可用模型流, 退出", flush=True)
        sys.exit(1)

    # 3) WS 常驻: 收盘 bar 驱动信号计算; 断线自动重连
    while True:
        try:
            run_ws()
        except Exception as e:
            print(f"[realtime] 连接异常: {e}", flush=True)
        print("[realtime] 连接断开, 5s 后重连", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
