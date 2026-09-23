import sys; sys.path.insert(0,'/workspace')
import os, time, gc, numpy as np, polars as pl
import config
from features import build_features, build_label, build_ret_future

T0=time.time()
def now(): return f"{time.time()-T0:.0f}s"

def build_one(sym, horizon):
    t = time.time()
    raw_p = f"{config.DS_DIR}/raw_{sym}.parquet"
    print(f"\n[{sym} h{horizon}] loading raw...", flush=True)
    df = pl.read_parquet(raw_p).sort("ts")
    print(f"  raw: {df.shape}", flush=True)
    feats = build_features(df)
    label = build_label(df["close"], horizon)
    retf = build_ret_future(df["close"], horizon)
    out = df.select([pl.col("ts")]).with_columns([
        label.alias("label"), 
        retf.alias("ret_future")
    ]).hstack(feats)
    # soft label 用 ret_future 列
    out = out.with_columns(
        (0.5 + 0.5 * pl.col("ret_future").clip(-0.02, 0.02) / 0.02).alias("soft_label")
    )
    nc = len(out.columns)
    ok = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in out.columns]) == nc
    out = out.filter(ok)
    out.write_parquet(f"{config.DS_DIR}/ds_{sym}_h{horizon}.parquet", compression="zstd")
    print(f"  saved ds_{sym}_h{horizon}.parquet cols={nc} rows={out.shape[0]:,} ({time.time()-t:.0f}s)", flush=True)
    del df, feats, out; gc.collect()

for sym, h in [("ETH",15), ("ETH",30), ("BTC",15), ("BTC",30)]:
    build_one(sym, h)

print(f"\n✅ ALL DONE ({now()})")
