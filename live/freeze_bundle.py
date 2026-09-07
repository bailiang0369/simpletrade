#!/usr/bin/env python3
"""Phase1-步骤1: 固化"最终版"为可复现/可在线重放的 Bundle。

职责:
  1. make-spec  导出 get_X 精确特征列序(38 表特征 + 3 extra + 17 跨资产)与协议常量,
                生成 live/spec/feature_spec.json + live/bundles/{h30,h60} 目录骨架。
  2. seed-conf  导出 meta_val 末尾 90 天置信度作为线上滚动阈值滑窗的种子
                (依赖已训练权重跑出 meta_val 融合概率; 未训练则提示去向)。

协议常量取自离线最终版:
  - 融合: JOINT pool20, rank 融合(BAGGED_SEEDS=5), 见 experiment_pool20_joint
  - 阈值: 每一 UTC 日, 前 90 天置信度 |p-0.5|*2 的 99 分位 (experiment_seq_model.r2_eval)
  - 独立性: ≥Nmin, h30=30 / h60=60 (greedy_sparse)
用法:
  python live/freeze_bundle.py make-spec
  python live/freeze_bundle.py seed-conf --symbol ETH --horizon 30
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES, CROSS_FEATURES,
                                BAGGED_SEEDS)

LIVE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(LIVE_DIR, "spec")
BUNDLES_DIR = os.path.join(LIVE_DIR, "bundles")

P99 = 99.0          # 逐日滚动阈值分位 (与 experiment_seq_model.P99 一致)
WIN_DAYS = 90       # 阈值滑窗天数 (experiment_seq_model.WIN_DAYS)
NMIN = {30: 30, 60: 60}   # 每周期信号独立性最小间隔(分钟)
COLD_MIN_DAYS = 30        # 冷启动: 滑窗无用样本天数低于此值则不发信号
HORIZONS = [30, 60]
SYMBOLS = ["ETH", "BTC"]

# 跨资产特征前缀规则: 目标币 ETH -> 源币 BTC_, 目标币 BTC -> 源币 ETH_
CROSS_PREFIX = {"ETH": "BTC_", "BTC": "ETH_"}


def get_column_order(symbol):
    """返回与 validate_eth_quick.get_X 完全一致的 58 列顺序。"""
    prefix = CROSS_PREFIX[symbol]
    cross = [prefix + c for c in CROSS_FEATURES]
    return {"base": list(FEATURES), "extra": list(EXTRA_FEATURE_NAMES),
            "cross": cross, "order": list(FEATURES) + list(EXTRA_FEATURE_NAMES) + cross}


def make_spec():
    os.makedirs(SPEC_DIR, exist_ok=True)
    os.makedirs(BUNDLES_DIR, exist_ok=True)
    spec = {
        "version": "1.0.0",
        "protocol": {
            "pool": "JOINT_pool20",
            "fams": ["lgb", "xgb", "cat"],
            "seeds": [int(s) for s in BAGGED_SEEDS],
            "fusion": "per-family probability -> within-sample percentile rank -> mean",
            "threshold": f"per UTC day, P{P99:.0f} of trailing {WIN_DAYS}d confidence |p-0.5|*2",
            "independence_min_gap_min": NMIN,
            "cold_start_min_days": COLD_MIN_DAYS,
            "time_base": "UTC (ts//86400 aligns to r2_eval)",
        },
        "symbols": SYMBOLS,
        "horizons": HORIZONS,
        "columns": {s: get_column_order(s) for s in SYMBOLS},
        "feature_cardinality": len(FEATURES) + len(EXTRA_FEATURE_NAMES) + len(CROSS_FEATURES),
    }
    out = os.path.join(SPEC_DIR, "feature_spec.json")
    json.dump(spec, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[freeze] spec -> {out}")
    print(f"[freeze] column order per symbol: base={len(FEATURES)} extra={len(EXTRA_FEATURE_NAMES)} "
          f"cross={len(CROSS_FEATURES)} total={spec['feature_cardinality']}")
    # 目录骨架
    for h in HORIZONS:
        os.makedirs(os.path.join(BUNDLES_DIR, f"h{h}"), exist_ok=True)
        print(f"[freeze] bundle dir -> live/bundles/h{h}/")
    EXT = {"lgb": "txt", "xgb": "json", "cat": "cbm"}
    weights = {}
    for h in HORIZONS:
        weights[h] = {}
        for sym in SYMBOLS:
            weights[h][sym] = {}
            for fam in ["lgb", "xgb", "cat"]:
                weights[h][sym][fam] = [
                    f"{BUNDLES_DIR}/h{h}/JOINT_{sym}_{fam}_seed{seed}.{EXT[fam]}"
                    for seed in BAGGED_SEEDS
                ]
    missing = [w for h in HORIZONS for sym in SYMBOLS for fam in ["lgb", "xgb", "cat"]
               for w in [weights[h][sym][fam]]
               if not all(os.path.exists(p) for p in w)]
    if missing:
        print(f"[freeze] 提示: 存在缺失权重, 若需生产权重请运行 live/train_pool.py(可后台续跑)。")
    return spec


def seed_conf(symbol, horizon):
    """导出 meta_val 末尾 WIN_DAYS*1440 个置信度作为线上阈值滑窗种子。

    用与实盘一致的 JOINT 池(validate_eth_quick.get_X 的 58 列特征 + predict.PoolPredictor
    整段秩融合)计算 meta_val 融合概率, 再取末尾 90 天置信度 — 保证种子与线上推理同源,
    不依赖旧 torch 推理通道或预存 npy(本沙箱无 torch 且无 JOINT_*_P.npy)。
    """
    from validate_eth_quick import get_X, compute_extra_raw
    from data_store import AssetContext
    from predict import PoolPredictor
    ctx = AssetContext(symbol, horizon=horizon)
    mask = ctx.split_rows["meta_val"]
    X = get_X(ctx, compute_extra_raw(ctx), mask)     # [n_mv, 58]
    pool = PoolPredictor(os.path.join(BUNDLES_DIR, f"h{horizon}"))
    P = pool.fused(X)                                 # 整段秩融合(== 离线语义)
    conf = np.abs(P - 0.5) * 2
    seed = list(conf[-WIN_DAYS * 1440:])             # 末尾 90 天, 对应 r2_eval seed_conf
    out = os.path.join(BUNDLES_DIR, f"h{horizon}", f"seed_conf_{symbol}_90d.json")
    json.dump(seed, open(out, "w"))
    print(f"[freeze] {symbol} h{horizon}: slide-window seed {len(seed)} samples -> {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-spec")
    sc = sub.add_parser("seed-conf")
    sc.add_argument("--symbol", required=True, choices=SYMBOLS)
    sc.add_argument("--horizon", required=True, type=int)
    a = ap.parse_args()
    if a.cmd == "make-spec":
        make_spec()
    else:
        seed_conf(a.symbol, a.horizon)


if __name__ == "__main__":
    main()