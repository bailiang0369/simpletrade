#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl, pandas as pd
from data_store import AssetContext, load_raw_bars
from features_online import compute_X
from validate_eth_quick import get_X, compute_extra_raw, CROSS_FEATURES

symbol="ETH"; other="BTC"
CROSS_PREFIX={"ETH":"BTC_","BTC":"ETH_"}
GLB0=38+3
cross_names=[CROSS_PREFIX[symbol]+c for c in CROSS_FEATURES]
COLZ=GLB0+cross_names.index("BTC_z_30")
COLZ60=GLB0+cross_names.index("BTC_z_60")
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
for COL,label in [(COLZ,"z_30"),(COLZ60,"z_60")]:
    D=np.abs(Xt[:,COL]-X_off[:,COL]); k=int(D.argmax())
    oi=int(np.searchsorted(other_df["ts"].to_numpy().astype(np.int64), ctx.ds_ts[blk[k]], side="right")-1)
    oc=other_df["close"].to_numpy().astype(np.float64); lc=np.log(np.maximum(oc,1e-12))
    w=30 if label=="z_30" else 60
    mu_pd=pd.Series(lc).rolling(w).mean().to_numpy()[oi]
    sd_pd=pd.Series(lc).rolling(w).std().to_numpy()[oi]
    win=lc[oi-w+1:oi+1]
    z_exact=(lc[oi]-mu_pd)/ (win.std(ddof=1) if win.std(ddof=1)>1e-9 else 1.0)
    z_pd=(lc[oi]-mu_pd)/ (sd_pd if sd_pd>1e-9 else 1.0)
    # numpy cumsum 单趟偏(有偏)方差: cs2/w - (cs/w)^2, ddof=0
    cs=np.cumsum(lc); cs2=np.cumsum(lc*lc)
    m=(cs[oi-w+1:oi+1].sum())/w
    # 直接对该窗口做有偏方差
    var_b=( (win*win).mean() )- (win.mean())**2
    sd_b=np.sqrt(max(var_b,0.0)); z_bias=(lc[oi]-mu_pd)/(sd_b if sd_b>1e-9 else 1.0)
    print(f"[{label}] off={X_off[k,COL]:.6f} on(exact)={Xt[k,COL]:.6f} D={D[k]:.2e} raw{rows[k]}")
    print(f"   z_exact={z_exact:.6f} z_pandas={z_pd:.6f} z_bias={z_bias:.6f} mu_pd={mu_pd:.10f} win_std={win.std(ddof=1):.3e} pd_std={sd_pd:.3e}")