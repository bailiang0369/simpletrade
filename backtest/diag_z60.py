#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from data_store import AssetContext, load_raw_bars
from features_online import compute_X, _cross_z_log
from validate_eth_quick import get_X, compute_extra_raw, CROSS_FEATURES

symbol="ETH"; other="BTC"
CROSS_PREFIX={"ETH":"BTC_","BTC":"ETH_"}
GLB0 = 38 + 3  # base(38)+extra(3)
cross_names=[CROSS_PREFIX[symbol]+c for c in CROSS_FEATURES]
COLZ = GLB0 + cross_names.index("BTC_z_60")   # 全局列号
print("COLZ =", COLZ, " total=58")

ctx=AssetContext(symbol,horizon=30)
blk=np.where(ctx.split_rows["test"])[0][-60000:]
rows=ctx.ds_to_raw[blk]
lo_ts=int(ctx.raw_ts[max(0,rows[0]-1500)])
self_df=load_raw_bars(symbol,since_ts=lo_ts).with_columns(pl.col("close").cast(pl.Float32))
other_df=load_raw_bars(other,since_ts=lo_ts)
X_on=compute_X(self_df,other_df,symbol)
local=np.searchsorted(self_df["ts"].to_numpy().astype(np.int64), ctx.ds_ts[blk])
Xt=X_on[local]
m=np.zeros(len(ctx.ds_ts),bool); m[blk]=True
X_off=get_X(ctx,compute_extra_raw(ctx),m)
D=abs(Xt[:,COLZ]-X_off[:,COLZ])
k=int(D.argmax())
print("max z60 diff=%.3f at block pos %d raw %d ds_ts %d self_pos %d"%(D[k],k,rows[k],ctx.ds_ts[blk[k]],local[k]))
print("on=%.6f off=%.6f"%(Xt[k,COLZ],X_off[k,COLZ]))

oc=other_df["close"].to_numpy().astype(np.float64)
o_ts=other_df["ts"].to_numpy().astype(np.int64)
oi=int(np.searchsorted(o_ts, ctx.ds_ts[blk[k]], side="right")-1)
lc=np.log(np.maximum(oc,1e-12))
print("aligned other_idx", oi, "BTC ts", o_ts[oi])
win=lc[oi-59:oi+1]
print("win len",len(win)," first/last logclose",win[0],win[-1])
print("manual z (60) =", (lc[oi]-win.mean())/win.std(ddof=1) if win.std(ddof=1)>1e-9 else "nan")
# 看该窗口内是否有极端点(与 win 均值差超 5 std)
d=abs(win-win.mean()); print("win max |z| inside:", (d/win.std(ddof=1)).max() if win.std(ddof=1)>1e-9 else "nan")
# 更大范围找 close 突刺: 该日附近 close 的 min/max log
seg=slice(max(0,oi-200), min(len(oc),oi+200)+1)
print("near range logclose min/max:", lc[seg].min(), lc[seg].max())
# 检查是否有 0/负 close 导致 ln nan
print("any close<=0 in whole frame:", (oc<=0).any(), " around:", (oc[max(0,oi-100):oi+101]<=0).any())
print("other frame len", len(oc), " own frame len", len(self_df))

# --- 直接对比 _cross_z_log 在 oi 处的输出 vs 手工 + 原始 pandas ---
import pandas as pd
z_cz = _cross_z_log(oc, 60)[oi]
spd = pd.Series(lc)
mu_pd = spd.rolling(60).mean().to_numpy()[oi]
sd_pd = spd.rolling(60).std().to_numpy()[oi]
print("_cross_z_log[oi]=", z_cz, " manual=", (lc[oi]-win.mean())/win.std(ddof=1) if win.std(ddof=1)>1e-9 else "nan")
print("pd rolling mean=", mu_pd, " manual mean=", win.mean())
print("pd rolling std =", sd_pd, " manual std(ddof=1)=", win.std(ddof=1))
# 检查 lc 在 [oi-80,oi] 内是否有 NaN 或 inf
print("lc nan in win:", np.isnan(lc[oi-80:oi+1]).sum(), " inf:", np.isinf(lc[oi-80:oi+1]).sum())
# oc 是否在 frame 某处为 0(导致 lc=-inf), 这会影响 rolling 但 causal 只影响之后
print("pos of close<=0 in frame:", np.where(oc<=0)[0][:10])