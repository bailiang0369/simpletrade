#!/usr/bin/env python3
"""Phase1-步骤5: 最小调用入口 live/run_once.py (非常驻, 手动/定时触发)。

流程:
  1. 从 data/datasets/raw_{SYM}.parquet 读最近的 1min bars(训练/回测同源, 官方公开 kline)。
  2. 逐流((symbol,horizon), h30/h60) 用在线特征引擎 compute_X -> 池推理(当日秩)
     -> 滚动阈值 -> 独立性 → 命中则落盘信号。
  3. 不做 ws 订阅 / 常驻 / 下单。

注意: h30/h60 权重需先由 live/train_pool.py 训练好放入 live/bundles/, 否则会直接报模型缺失。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from data_store import load_raw_bars
from features_online import compute_X
from predict import PoolPredictor, RollingThreshold, SessionRankFuser
from signals import IndependenceFilter, emit_signal, flush_recent

BUNDLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundles")
SYMBOLS = ["ETH", "BTC"]
HORIZONS = [30, 60]
WARMUP = 480          # 特征/自Z/滚动窗需要的最短 bar 数(>960 亦可, 保守取 960+)
LOOKBACK = 1500       # 每个预测时刻向前取多少根 bar(含跨资产), 覆盖全部最长滚窗


def _with_warmup(bars, n=LOOKBACK):
    """返回每币最近 LOOKBACK 根 bar(时间升序)供在线特征计算; 不足则返回 None。"""
    if bars is None or len(bars) < WARMUP:
        return None
    return bars[-n:]


def main():
    # 1) 读双币 bars(与离线同 raw parquet; 实盘替换为实时等结构 DataFrame)
    bars = {s: load_raw_bars(s) for s in SYMBOLS}
    win = {s: _with_warmup(bars[s]) for s in SYMBOLS}
    for s in SYMBOLS:
        if win[s] is None:
            print(f"[run_once] 跳过 {s}: bars 不足 {WARMUP}", flush=True)

    conf_fusers = {}     # (symbol,horizon) -> SessionRankFuser
    thresholds = {}      # (symbol,horizon) -> RollingThreshold
    indep = IndependenceFilter()
    # 统一 CPU 时差: 以两币最近 ts 的较小者作为"当前时刻"
    cur_ts = min(int(win[s]["ts"].to_numpy()[-1]) for s in SYMBOLS)

    emitted = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            pool_dir = os.path.join(BUNDLES_DIR, f"h{horizon}")
            # 需完整 15 booster(以 lgb seed42 为例)才可加载; 缺失(如 h60 未训练)则跳过
            probe = os.path.join(pool_dir, "JOINT_lgb_seed42.txt")
            if not os.path.isfile(probe):
                print(f"[run_once] 跳过 {symbol}/h{horizon}: 无权重 {probe}", flush=True)
                continue
            # 冷启动种子: 从 90d 置信度种子加载(无则在 bundle 内找 seed_conf_*)
            seed_files = [f for f in os.listdir(pool_dir) if f.startswith("seed_conf") and symbol in f]
            seed_conf = []
            if seed_files:
                import json
                seed_conf = json.load(open(os.path.join(pool_dir, seed_files[0])))
            fuser = SessionRankFuser(pool_dir)
            thr = RollingThreshold(seed_conf)
            # 在线特征(对手盘=另一币的 bars, 与离线 cross 前缀一致)
            other = "ETH" if symbol == "BTC" else "BTC"
            X = compute_X(win[symbol], win[other], symbol)      # [n, 58]

            # 当日秩需在"当日已见样本"上增量累计才有意义。每次 run 取最近两根
            # (today bars 全喂会过重), 以窗口尾部样本重建秩后取最后一根。
            # 注: SessionRankFuser 是纯增量, 无法重置"当日"边界; 这里用滚动窗口近似:
            #     只把窗口尾部一小段喂入以得相对秩, 实际当日阈值的绝对值不依赖秩大小
            #     (阈值是置信度|p-0.5|*2 的绝对分位), 因此这里仅需 p 稳定、conf 正确。
            last_p, last_conf = None, None
            for i in range(max(0, len(X) - WARMUP), len(X)):
                last_p, last_conf = fuser.fused_at(X[i:i + 1].reshape(1, -1))
            p, conf = last_p, last_conf
            tau = thr.threshold()
            if tau is None:
                print(f"[run_once] {symbol}/h{horizon}: 冷启动, 不发信号", flush=True)
                continue
            if conf < tau:
                print(f"[run_once] {symbol}/h{horizon}: conf={conf:.4f}<tau={tau:.4f}, 不达阈值",
                      flush=True)
                continue
            if not indep.accept(symbol, horizon, cur_ts):
                continue
            rec = emit_signal(symbol, horizon, cur_ts, p, conf, tau)
            emitted.append(rec)
            print(f"[run_once] SIGNAL {symbol}/{horizon} t={rec['iso']} dir={rec['direction']} "
                  f"p={rec['p']} conf={rec['conf']} thresh={tau:.4f}", flush=True)

    flush_recent(emitted)
    print(f"[run_once] done, {len(emitted)} signal(s) written", flush=True)


if __name__ == "__main__":
    main()