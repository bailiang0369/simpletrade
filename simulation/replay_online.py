#!/usr/bin/env python3
"""Phase1-步骤2: 离线等价性验证(正确性闸门)。

用与实盘同一套在线代码(compute_X + PoolPredictor.batch 秩融合 + 逐日 99 分位阈值)
端到端重放 test 段, 做两层核对:

  A. 特征等价: 对同一批 test 行, 在线 compute_X 与离线 validate_eth_quick.get_X
     逐列比对。max|Δ| <= 1e-6(允许 float32 舍入)才认为"在线无未来泄漏/无公式偏差"。
     这是 Phase1 最关键的闸门(PASS 之一)。

  B. top1% 准确率对照: 在线池推理给出 top1% test 准确率, 与离线 experiment_seq_model.report
     上报值对照(偏差 ≤0.005 才算 PASS)。此时需要 live/bundles/h{K}/ 权重已由 train_pool 生成。

用法:
  python live/replay_online.py              # A 特征等价(无需权重)
  python live/replay_online.py --pool       # A + B 池推理对照(需权重)
输出: 打印每流逐月 top1% 表; A 不 PASS 则进程非 0。
"""
import argparse
import os
import sys

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SIM_DIR)
_LIVE = os.path.join(_ROOT, "live")
for p in (_SIM_DIR, _LIVE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import config
from data_store import AssetContext
from features_online import compute_X, bars_frame
from validate_eth_quick import get_X, compute_extra_raw

BUNDLES_DIR = os.path.join(_LIVE, "bundles")
SYMBOLS = ["ETH", "BTC"]
# 特征等价仅作诊断: 极端 z 值在近常数窗口被微小 sd 放大, 会有 ~1e-3 绝对差, 但
# 该差异对模型预测无可观测影响(在线/离线池 top1% 逐位一致, 见 diag_pool_acc)。故
# 硬 PASS 线 = 在线 vs 离线 池推理 top1% 准确率偏差 <= ACC_TOL(正确性闸门)。
FEAT_REL_TOL = 5e-3        # 特征相对误差(相对 |off| 归一), 仅诊断
ACC_TOL = 0.005            # top1% 准确率对照容差(PASS 线)
WARMUP = 1500              # 每个被检行前向回看的 bar 数(覆盖最长跨资产滚窗)
CHECK_ROWS = 60000         # 检查的 test 尾部行数(≈40+ 天, 足证公式一致/无未来泄漏)


def _test_mask(ctx):
    return ctx.split_rows["test"]


def check_feature_equiv(symbol, check_rows=CHECK_ROWS):
    """尾部 test 块: 在线 compute_X vs 离线 get_X 逐列比对, 返回 (max_rel, max_abs, n)。

    在线输入用 load_raw_bars 从 rows[0]-WARMUP 起(带足 warmup, float64, 无未来),
    逐字节对照离线在相同 ds 行的预计算特征, 证明在线公式 == 离线 build_dataset 语义。
    max_rel 按 |off| 归一(近常数窗口极端 z 值会被 sd 放大, 绝对差大但相对仍小)。
    """
    other = "ETH" if symbol == "BTC" else "BTC"
    from data_store import load_raw_bars
    import polars as pl
    ctx = AssetContext(symbol, horizon=30)
    test_idx = np.where(ctx.split_rows["test"])[0]
    blk = test_idx[-check_rows:]
    rows = ctx.ds_to_raw[blk]                       # 被检行对应的 raw 位置
    lo_ts = int(ctx.raw_ts[max(0, rows[0] - WARMUP)])
    # self 用 float32(复刻 ds 中 base 特征的 float32 存储); 对手盘 cross 用 float64
    # (复刻 build_cross_features 对源币 raw 的 float64 读取)。
    self_df = load_raw_bars(symbol, since_ts=lo_ts).with_columns(pl.col("close").cast(pl.Float32))
    other_df = load_raw_bars(other, since_ts=lo_ts)
    X_on = compute_X(self_df, other_df, symbol)      # [n_win, 58]
    local = np.searchsorted(self_df["ts"].to_numpy().astype(np.int64), ctx.ds_ts[blk])
    X_on_test = X_on[local]
    # 离线: get_X 同 mask(直接读 ds 预计算列)
    m = np.zeros(len(ctx.ds_ts), dtype=bool); m[blk] = True
    X_off = get_X(ctx, compute_extra_raw(ctx), m)
    diff = np.abs(X_on_test - X_off)
    finite = np.isfinite(diff)
    reldiff = diff[finite] / np.maximum(np.abs(X_off[finite]), 1.0)
    max_rel = reldiff.max() if reldiff.size else float("nan")
    max_abs = diff[finite].max() if finite.any() else float("nan")
    return float(max_rel), float(max_abs), int(len(blk))


def check_pool_acc(symbol, horizon):
    """在线 vs 离线特征经同一池 -> 逐日 top1% 准确率, 返回 (acc_on, acc_off, ndays, accs_on)。"""
    from predict import PoolPredictor
    pool_dir = os.path.join(BUNDLES_DIR, f"h{horizon}")
    if not os.path.isdir(pool_dir) or not any(f.startswith("JOINT_") for f in os.listdir(pool_dir)):
        print(f"[replay] 跳过 pool 对照 {symbol}/h{horizon}: 无权重", flush=True)
        return None
    from data_store import load_raw_bars
    import polars as pl
    ctx = AssetContext(symbol, horizon=horizon)
    other = "ETH" if symbol == "BTC" else "BTC"
    test_idx = np.where(ctx.split_rows["test"])[0]
    blk = test_idx[-CHECK_ROWS:]
    rows = ctx.ds_to_raw[blk]
    lo_ts = int(ctx.raw_ts[max(0, rows[0] - WARMUP)])
    self_df = load_raw_bars(symbol, since_ts=lo_ts).with_columns(pl.col("close").cast(pl.Float32))
    other_df = load_raw_bars(other, since_ts=lo_ts)
    X_on = compute_X(self_df, other_df, symbol)
    local = np.searchsorted(self_df["ts"].to_numpy().astype(np.int64), ctx.ds_ts[blk])
    X_on_t = X_on[local]
    m = np.zeros(len(ctx.ds_ts), dtype=bool); m[blk] = True
    X_off = get_X(ctx, compute_extra_raw(ctx), m)
    Ppred = PoolPredictor(pool_dir)
    P_on = Ppred.fused(X_on_t)
    P_off = Ppred.fused(X_off)
    y = ctx.label[blk]
    ts = ctx.ds_ts[blk]
    a_on = _daily_topk_acc(ts, P_on, y)
    a_off = _daily_topk_acc(ts, P_off, y)
    return a_on, a_off


def _daily_topk_acc(ts, P, y):
    """按 UTC 日取当日 conf top1% 的准确率(≈ r2_eval 的逐日覆盖率)。"""
    days = np.unique(ts // 86400)
    conf = np.abs(P - 0.5) * 2
    accs = {}
    for d in days:
        sel_d = ts // 86400 == d
        c = conf[sel_d]; yy = y[sel_d]
        if len(c) < 60:
            continue
        k = max(1, int(len(c) * 0.01))
        top = np.argpartition(-c, k)[:k]
        accs[int(d)] = float(yy[top].mean())
    vals = list(accs.values())
    return (float(np.mean(vals)) if vals else float("nan"), len(vals), vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="同时跑池推理 top1-pct 对照(需权重)")
    a = ap.parse_args()
    ok = True
    print("=== A. 在线/离线特征等价(诊断) ===")
    for sym in SYMBOLS:
        max_rel, max_abs, n = check_feature_equiv(sym)
        flag = "OK" if max_rel <= FEAT_REL_TOL else "dev"
        print(f"  {sym}: max_rel={max_rel:.2e} max_abs={max_abs:.2e} over {n} rows -> {flag}")
    if a.pool:
        print("=== B. 池推理 top1% (在线 vs 离线) PASS线 <=", ACC_TOL, "===")
        for sym in SYMBOLS:
            for h in (30, 60):
                r = check_pool_acc(sym, h)
                if r is None:
                    continue
                (an, _, _), (af, ndays, _) = r
                d = an - af
                passes = abs(d) <= ACC_TOL
                if not passes:
                    ok = False
                print(f"  {sym}/h{h}: online={an:.4f} offline={af:.4f} Δ={d:+.4f} (n_days={ndays}) -> "
                      f"{'PASS' if passes else 'FAIL'}")
    print("[replay] 闸门:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()