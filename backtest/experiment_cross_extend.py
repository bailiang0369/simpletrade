#!/usr/bin/env python3
"""在已验证有效的跨资产特征(12列)上加码, 检验"增强版跨资产特征"能否继续提升30min方向准确率。

增强内容(全部因果, 只用 <= t 的源币/目标币信息):
  1) 更长回看窗口: 源币 lr_480/lr_960, z_240/z_480, rvol_240
  2) 比值特征: ETH/BTC 对数价差的 rolling z-score (spread_z_60/240), 捕捉相对价值均值回归

公平对比: 同一 LGB 流程/早停/种子, 仅特征集不同:
  - 原12列 (当前管线口径)  vs  增强19列
R2 逐日滚动 Top1% 协议与 HANDOFF 基线同口径。

用法: python experiment_cross_extend.py [ETH|BTC]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc, argparse, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM
from experiment_cross_asset import build_cross_asset, fit, ens_predict, OTHER

MAX_TRAIN = 2_000_000
LR = 0.03
BAGGED = [42, 49]
EXTRA_LR = (480, 960)
EXTRA_Z = (240, 480)


def _shift(a, k):
    out = np.empty_like(a)
    out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def build_extended_cross(ctx, other):
    """原12列 + 更长回看 + ETH/BTC 比值z-score, 全部对齐到目标币 ds_ts (最近 <= t 的源币行)。"""
    t = pq.read_table(f"{config.DS_DIR}/raw_{other}.parquet",
                      columns=["ts", "close", "buy_vol", "sell_vol"])
    ots = t["ts"].to_numpy().astype(np.int64)
    oc = t["close"].to_numpy().astype(np.float64)
    ob = t["buy_vol"].to_numpy().astype(np.float64)
    os_ = t["sell_vol"].to_numpy().astype(np.float64)
    del t
    gc.collect()

    # 目标币 close 对齐到源币网格(最近 <= 源币ts), 用于比值特征(因果)
    t_ci = np.searchsorted(ctx.raw_ts, ots, side="right") - 1
    t_ci = np.clip(t_ci, 0, len(ctx.c) - 1)
    tc_at_src = ctx.c[t_ci].astype(np.float64)
    del t_ci
    gc.collect()

    n_o = len(oc)
    lc = np.log(np.maximum(oc, 1e-12))
    ltc = np.log(np.maximum(tc_at_src, 1e-12))
    sp = ltc - lc                                   # 比值对数价差(目标/源)
    del tc_at_src
    gc.collect()

    cols, names = [], []

    def add(nm, arr):
        cols.append(arr.astype(np.float32))
        names.append(f"{other}_{nm}")

    # ---- 源币动量(原12列回看 + 更长回看) ----
    for k in (5, 15, 30, 60, 120, 240) + EXTRA_LR:
        add(f"lr_{k}", lc - _shift(lc, k))
    # ---- 源币 z-score ----
    s = pd.Series(lc)
    for w in (30, 60, 120) + EXTRA_Z:
        mu = s.rolling(w).mean().to_numpy()
        sd = s.rolling(w).std().to_numpy()
        add(f"z_{w}", np.where(sd > 1e-9, (lc - mu) / sd, 0.0))
    # ---- 源币波动率 ----
    lr1 = np.full(n_o, np.nan)
    lr1[1:] = lc[1:] - lc[:-1]
    for w in (60, 240):
        add(f"rvol_{w}", pd.Series(lr1).rolling(w).std().to_numpy() * 100)
    # ---- 源币主动买卖失衡 ----
    d = pd.Series(ob - os_)
    tt = pd.Series(ob + os_)
    for w in (30, 60):
        add(f"cvd_{w}", (d.rolling(w).sum() / (tt.rolling(w).sum() + 1e-12)).to_numpy())
    # ---- 比值价差 z-score (相对价值均值回归) ----
    sps = pd.Series(sp)
    for w in (60, 240):
        mu = sps.rolling(w).mean().to_numpy()
        sd = sps.rolling(w).std().to_numpy()
        add(f"spread_z_{w}", np.where(sd > 1e-9, (sp - mu) / sd, 0.0))
    del s, sps, lc, lr1, d, tt, oc, ob, os_, sp
    gc.collect()

    F = np.stack(cols, axis=1).astype(np.float32)
    del cols
    gc.collect()
    idx = np.searchsorted(ots, ctx.ds_ts, side="right") - 1
    bad = idx < 0
    idx = np.clip(idx, 0, n_o - 1)
    out = F[idx]
    if bad.any():
        out[bad] = 0.0
        print(f"  [warn] {int(bad.sum())} ds 行对齐不到 {other} raw, 置零", flush=True)
    del F, idx, bad, ots
    gc.collect()
    print(f"  增强跨资产特征 {len(names)} 列: {names}", flush=True)
    return out.astype(np.float32), names


def run(symbol):
    other = OTHER[symbol]
    print(f"\n########## {symbol} 跨资产增强验证 ##########", flush=True)
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    F12, _ = build_cross_asset(ctx, other)      # 原12列
    F19, _ = build_extended_cross(ctx, other)   # 增强19列

    results = {}
    for tag, Fextra in (("原12列", F12), ("增强19列", F19)):
        models = []
        for seed in BAGGED:
            m, bi = fit(ctx, Fextra, True, seed)
            print(f"  [{tag}/seed{seed}] iter={bi} ({time.time()-t0:.0f}s)", flush=True)
            models.append(m)
        pmv = ens_predict(ctx, models, "meta_val", Fextra, True)
        pt = ens_predict(ctx, models, "test", Fextra, True)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8),
                        ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
        ysel = ctx.y("test")[r["sel"]]
        acc = ESM.report(tag, r["pred"], ysel, r["mts"])
        results[tag] = acc
        for m in models: del m
        gc.collect()
    print(f"\n  >>> {symbol}: 原12列={results['原12列']:.4f}  增强19列={results['增强19列']:.4f}  "
          f"增量={results['增强19列']-results['原12列']:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run(s)


if __name__ == "__main__":
    main()
