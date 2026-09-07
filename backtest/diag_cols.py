#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from data_store import AssetContext, load_raw_bars
from features_online import compute_X
from validate_eth_quick import get_X, compute_extra_raw, FEATURES, EXTRA_FEATURE_NAMES, CROSS_FEATURES

for symbol in ["ETH","BTC"]:
    other="BTC" if symbol=="ETH" else "ETH"
    CROSS_PREFIX={"ETH":"BTC_","BTC":"ETH_"}
    ctx=AssetContext(symbol,horizon=30)
    blk=np.where(ctx.split_rows["test"])[0][-60000:]
    rows=ctx.ds_to_raw[blk]
    lo_ts=int(ctx.raw_ts[max(0,rows[0]-1500)])
    other_df=load_raw_bars(other,since_ts=lo_ts)
    self_df=load_raw_bars(symbol,since_ts=lo_ts).with_columns(pl.col("close").cast(pl.Float32))
    X_on=compute_X(self_df,other_df,symbol)
    local=np.searchsorted(self_df["ts"].to_numpy().astype(np.int64), ctx.ds_ts[blk])
    Xt=X_on[local]
    m=np.zeros(len(ctx.ds_ts),bool); m[blk]=True
    X_off=get_X(ctx,compute_extra_raw(ctx),m)
    names=list(FEATURES)+list(EXTRA_FEATURE_NAMES)+[CROSS_PREFIX[symbol]+c for c in CROSS_FEATURES]
    mae=np.abs(Xt-X_off).max(axis=0)
    order=np.argsort(-mae)[:10]
    print("==",symbol,"==")
    for j in order:
        print(f"  {names[j]:24s} max={mae[j]:.3e}")
    print("  rows big(>1e-3):", int((np.abs(Xt-X_off).max(axis=1)>1e-3).sum()))