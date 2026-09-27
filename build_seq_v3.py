"""Build v3 data: W=64, S=8, 16 channels, MAX_TR=500K."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, gc, datetime as dtm, time
import config
from numpy.lib.stride_tricks import sliding_window_view

t0=time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right')-1, 0, len(btc)-1)
ts = eth['ts'].values.astype(np.int64); C = eth['close'].values.astype(np.float64)
O=eth['open'].values.astype(np.float64); Hv=eth['high'].values.astype(np.float64)
Lv=eth['low'].values.astype(np.float64); BV=eth['buy_vol'].values.astype(np.float64)
SV=eth['sell_vol'].values.astype(np.float64); FUND=eth['funding'].values.astype(np.float64)
BTC_C=btc['close'].values.astype(np.float64)[br]
del eth, btc; gc.collect(); N=len(C)

def z(x,w=2880):
    s=pd.Series(x.astype(np.float64)); mu=s.rolling(w,min_periods=w//4).mean().values
    sd=s.rolling(w,min_periods=w//4).std().values+1e-8
    return ((x-mu)/sd).astype(np.float32)
def rs(x,w): return pd.Series(x.astype(np.float64)).rolling(w,min_periods=w//4).std().values.astype(np.float32)

lr1=np.zeros(N,dtype=np.float64); lr1[1:]=np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8))
tre=int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()); trm=ts<tre
f_lo,f_hi=np.percentile(FUND[trm],0.5),np.percentile(FUND[trm],99.5); FC=np.clip(FUND,f_lo,f_hi)
TV=BV+SV; TV_s=np.where(TV>0,TV,1.0); CVD=(BV-SV)/TV_s
BR=np.log(np.where(BV>0,BV/np.maximum(SV,1e-8),1e-8))
B_lr1=np.zeros(N,dtype=np.float64); B_lr1[1:]=np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
TB=BV-np.roll(BV,1); TS_d=SV-np.roll(SV,1)
body=np.abs(C-O)/np.where(Hv>Lv,Hv-Lv,1.0)
# Add multi-horizon lrs
LR=np.zeros((4,N),dtype=np.float64)
for i,w in enumerate([5,15,30,60]): LR[i,w:]=np.log(np.maximum(C[w:],1e-8)/np.maximum(C[:-w],1e-8))

# 16 channels
channels = np.stack([
    z(lr1),                     # 0
    z(LR[0]), z(LR[1]), z(LR[2]), z(LR[3]),   # 1-4: lr 5/15/30/60
    z(rs(lr1, 15)), z(rs(lr1, 60)), z(rs(lr1, 240)),  # 5-7: vol
    z(CVD), z(BR),              # 8-9: order flow
    z(FC),                      # 10: funding
    z(B_lr1), z(rs(B_lr1, 60)), # 11-12: BTC
    z(body),                    # 13: shape
    z(TB), z(TS_d),             # 14-15: active buy/sell
], axis=0).astype(np.float32)
# Fill NaN from rolling warmup
channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
print(f"channels={channels.shape}", flush=True)

# Save as memmap
ch_path = '/tmp/ch16.bin'
np.memmap(ch_path, dtype=np.float32, mode='w+', shape=channels.shape)[:] = channels[:]
gc.collect()
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='r', shape=(16, N))

# Build anchors
H=15; W=64; S=8; MAX_TR=500_000
anchor_indices=np.arange(W,N-H,S,dtype=np.int64); anchor_ts=ts[anchor_indices]
def tmask_arr(ts_,s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)
tr_a=tmask_arr(anchor_ts,*config.SPLITS['train'])
es_a=tmask_arr(anchor_ts,*config.SPLITS['early_stop'])
te_a=tmask_arr(anchor_ts,*config.SPLITS['test'])
print(f"anchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}", flush=True)

rets=C[anchor_indices+H]/C[anchor_indices]-1
tr_sel=tr_a&(np.abs(rets)>0.0003)
if tr_sel.sum()>MAX_TR:
    rng=np.random.default_rng(42); keep=rng.choice(np.where(tr_sel)[0],MAX_TR,replace=False)
    tr_sel=np.zeros_like(tr_sel,dtype=bool); tr_sel[keep]=True
print(f"filtered tr={tr_sel.sum():,}", flush=True)

# Stride tricks
from numpy.lib.stride_tricks import sliding_window_view
swv = sliding_window_view(ch_rm, W, axis=1)   # view: (16, N-W, W)
wa = np.transpose(swv, (1,0,2))
win_idx = anchor_indices - W   # index into wa

def chunk_materialize(sel, name, batch=20000):
    picked = win_idx[sel]; n = len(picked); parts = []
    for i in range(0, n, batch):
        part = wa[picked[i:i+batch]].copy(); parts.append(part); del part; gc.collect()
    X = np.concatenate(parts, axis=0)
    print(f"  {name}: {X.shape} ({X.nbytes/1e9:.2f}GB)", flush=True)
    return X

t0=time.time()
X_tr = chunk_materialize(tr_sel, 'X_tr')
X_es = chunk_materialize(es_a, 'X_es')
X_te = chunk_materialize(te_a, 'X_te')

y_tr = (rets[tr_sel] > 0).astype(np.int64)
r_tr = rets[tr_sel].astype(np.float32)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64)
ts_te = anchor_ts[te_a]

np.savez_compressed('/workspace/models_saved/seq_data_v3.npz',
    X_tr=X_tr, y_tr=y_tr, r_tr=r_tr,
    X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f"\nv3 saved!  total {time.time()-t0:.0f}s", flush=True)
print(f"TR pos={y_tr.mean():.3f}  ES pos={y_es.mean():.3f}  TE pos={y_te.mean():.3f}", flush=True)
del ch_rm, wa; gc.collect()
