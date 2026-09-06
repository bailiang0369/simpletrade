#!/usr/bin/env python3
"""分解增强跨资产特征: 去掉 spread_z 后, 17列(更长回看 lr_480/960, z_240/480, rvol_240)是否对两币都稳健为正。

背景: experiment_cross_extend.py 显示 ETH +1.01pp / BTC -1.75pp, 方向不一致(=不稳健, 同 funding 教训)。
需要定位罪魁: 是"更长回看"(同质扩展, 更可能稳健)还是"比值价差"(跨币相对值, 可能不稳定)。
本脚本只对比 12列 vs 17列(无 spread), 两币同协议。

用法: python experiment_cross_lr17.py [ETH|BTC]
"""
import os, sys, gc, argparse, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM
from experiment_cross_asset import build_cross_asset, fit, ens_predict, OTHER

BAGGED = [42, 49]
EXTRA_LR = (480, 960)
EXTRA_Z = (240, 480)


def _shift(a, k):
    out = np.empty_like(a)
    out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def build_lr17_cross(ctx, other):
    """12列 + 更长回看(不含 spread_z), 对齐到目标币 ds_ts。"""
    t = pq.read_table(f"{config.DS_DIR}/raw_{other}.parquet",
                      columns=["ts", "close", "buy_vol", "sell_vol"])
    ots = t["ts"].to_numpy().astype(np.int64)
    oc = t["close"].to_numpy().astype(np.float64)
    ob = t["buy_vol"].to_numpy().astype(np.float64)
    os_ = t["sell_vol"].to_numpy().astype(np.float64)
    del t
    gc.collect()
    n_o = len(oc)
    lc = np.log(np.maximum(oc, 1e-12))
    cols, names = [], []

    def add(nm, arr):
        cols.append(arr.astype(np.float32))
        names.append(f"{other}_{nm}")

    for k in (5, 15, 30, 60, 120, 240) + EXTRA_LR:
        add(f"lr_{k}", lc - _shift(lc, k))
    s = pd.Series(lc)
    for w in (30, 60, 120) + EXTRA_Z:
        mu = s.rolling(w).mean().to_numpy()
        sd = s.rolling(w).std().to_numpy()
        add(f"z_{w}", np.where(sd > 1e-9, (lc - mu) / sd, 0.0))
    lr1 = np.full(n_o, np.nan)
    lr1[1:] = lc[1:] - lc[:-1]
    for w in (60, 240):
        add(f"rvol_{w}", pd.Series(lr1).rolling(w).std().to_numpy() * 100)
    d = pd.Series(ob - os_)
    tt = pd.Series(ob + os_)
    for w in (30, 60):
        add(f"cvd_{w}", (d.rolling(w).sum() / (tt.rolling(w).sum() + 1e-12)).to_numpy())
    del s, lc, lr1, d, tt, oc, ob, os_
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
    del F, idx, bad, ots
    gc.collect()
    print(f"  17列跨资产特征 {len(names)} 列: {names}", flush=True)
    return out.astype(np.float32), names


def run(symbol):
    other = OTHER[symbol]
    print(f"\n########## {symbol} 17列(更长回看,无spread)验证 ##########", flush=True)
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    F12, _ = build_cross_asset(ctx, other)
    F17, _ = build_lr17_cross(ctx, other)

    results = {}
    for tag, Fextra in (("原12列", F12), ("17列", F17)):
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
    print(f"\n  >>> {symbol}: 原12列={results['原12列']:.4f}  17列={results['17列']:.4f}  "
          f"增量={results['17列']-results['原12列']:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run(s)


if __name__ == "__main__":
    main()
