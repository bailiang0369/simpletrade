"""批量构建 ETH/BTC 多 horizon 数据集 (h5, h15, h30)。

每个 ds_{SYM}_h{H}.parquet 包含:
  ts, label, soft_label, ret_future + 所有特征列 (float32)
"""
import os, sys, gc, time
sys.path.insert(0, "/workspace")
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
from features import build_features, build_label, build_ret_future

HORIZONS = [5, 15, 30]
SYMBOLS = ["ETH", "BTC"]


def build_one(sym, horizon):
    t0 = time.time()
    raw_p = f"{config.DS_DIR}/raw_{sym}.parquet"
    print(f"\n[{sym} h{horizon}] loading raw...", flush=True)
    df = pl.read_parquet(raw_p).sort("ts")
    print(f"  raw shape: {df.shape}", flush=True)

    # 特征
    feats = build_features(df)
    print(f"  feats shape: {feats.shape} ({feats.shape[1]} cols)", flush=True)

    # 标签
    close = df["close"]
    label = build_label(close, horizon)
    retf = build_ret_future(close, horizon)
    # soft_label: ret_future 压缩到 [0,1]
    soft = (pl.lit(0.5) + pl.lit(0.5) * pl.col("__retf__").clip(-0.02, 0.02) / 0.02).alias("soft_label")
    tmp = df.select([pl.col("ts"), retf.alias("__retf__")])
    soft_col = tmp.with_columns(soft)["soft_label"]

    # 合并: ts + label + ret_future + soft_label + features
    out = df.select([pl.col("ts")]).with_columns([
        label.alias("label"),
        retf.alias("ret_future"),
        soft_col,
    ]).hstack(feats)

    # 剔除 null (warmup + 边界) — 逐行: 所有列非 null
    n_cols = len(out.columns)
    ok_expr = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in out.columns]) == n_cols
    finite_mask = out.select(ok_expr.alias("ok")).to_series(0).to_numpy()
    out = out.filter(ok_expr)
    print(f"  valid rows: {out.shape[0]:,} (drop {len(finite_mask)-out.shape[0]:,})", flush=True)

    out_p = f"{config.DS_DIR}/ds_{sym}_h{horizon}.parquet"
    out.write_parquet(out_p, compression="zstd")
    print(f"  saved → {out_p} ({time.time()-t0:.1f}s)", flush=True)
    gc.collect()
    return out.shape[0]


def main():
    os.makedirs(config.DS_DIR, exist_ok=True)
    for sym in SYMBOLS:
        for h in HORIZONS:
            n = build_one(sym, h)
    print("\n全部构建完成 ✓")


if __name__ == "__main__":
    main()
