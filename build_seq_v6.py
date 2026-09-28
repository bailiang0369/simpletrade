"""v6: Raw OHLCV + vol channels (no per-window norm), W=64."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, gc, datetime as dtm, time
import config
from numpy.lib.stride_tricks import sliding_window_view

t0=time.time()
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right')-1, 0, len(btc)-1)
ts = eth['ts'].values.astype(np.int64)
C = eth['close'].values.astype(np.float64); O = eth['open'].values.astype(np.float64)
Hv = eth['high'].values.astype(np.float64); Lv = eth['low'].values.astype(np.float64)
BV = eth['buy_vol'].values.astype(np.float64); SV = eth['sell_vol'].values.astype(np.float64)
FUND = eth['funding'].values.astype(np.float64)
BTC_C = btc['close'].values.astype(np.float64)[br]; del eth, btc; gc.collect(); N=len(C)

# Stationary channels
lr1 = np.zeros(N, dtype=np.float64); lr1[1:]=np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8))
LR15 = np.zeros(N, dtype=np.float64); LR15[15:]=np.log(np.maximum(C[15:],1e-8)/np.maximum(C[:-15],1e-8))
LR60 = np.zeros(N, dtype=np.float64); LR60[60:]=np.log(np.maximum(C[60:],1e-8)/np.maximum(C[:-60],1e-8))
B_lr1 = np.zeros(N, dtype=np.float64); B_lr1[1:]=np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
TV = BV + SV; TV_s = np.where(TV > 0, TV, 1.0)
CVD = (BV-SV)/TV_s  # already [-1,1]
body = np.abs(C-O)/np.where(Hv>Lv, Hv-Lv, 1.0)  # already [0,1]
# Volume log
vol = np.log1p(TV)
# Roll std
def rstd(x, w):
    return np.nan_to_num(pd.Series(x).rolling(w, min_periods=max(5,w//3)).std().values, nan=0.0).astype(np.float32)
vol15 = rstd(lr1, 15); vol60 = rstd(lr1, 60); vol240 = rstd(lr1, 240)

# Global robust z-score (based on train portion only, to avoid leakage)
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre

def robust_z(x):
    lo, hi = np.percentile(x[trm], 0.5), np.percentile(x[trm], 99.5)
    xc = np.clip(x, lo, hi)
    mu = xc[trm].mean(); sd = xc[trm].std() + 1e-6
    return ((xc - mu) / sd).astype(np.float32)

def scale_unit(x):
    """For already bounded features: keep as-is but clip outliers"""
    lo, hi = np.percentile(x[trm], 0.5), np.percentile(x[trm], 99.5)
    return np.clip(x, lo, hi).astype(np.float32)

# 14 channels
ch_names = ['lr1','lr15','lr60','BTC_lr1','vol15','vol60','vol240','CVD','body','log_vol','funding','close_diff_open','close_diff_high','close_diff_low']
channels_list = [
    robust_z(lr1), robust_z(LR15), robust_z(LR60), robust_z(B_lr1),
    robust_z(vol15), robust_z(vol60), robust_z(vol240),
    scale_unit(CVD), scale_unit(body),
    robust_z(vol), robust_z(FUND),
    robust_z((C-O)/np.maximum(C,1e-8)),  # close-open
    robust_z((Hv-C)/np.maximum(Hv,1e-8)),  # high-close
    robust_z((C-Lv)/np.maximum(Lv,1e-8)),  # close-low
]
CIN = len(channels_list)
ch_arr = np.stack(channels_list, axis=0).astype(np.float32)
print(f"ch_arr shape={ch_arr.shape}  per-channel:", flush=True)
for i,nm in enumerate(ch_names):
    print(f"  {nm}: [{np.nanmin(ch_arr[i]):.2f}, {np.nanmax(ch_arr[i]):.2f}] mean={np.nanmean(ch_arr[i]):.3f}", flush=True)
ch_arr = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0)

# memmap
ch_path = '/tmp/ch_v6.bin'
ch_mem = np.memmap(ch_path, dtype=np.float32, mode='w+', shape=ch_arr.shape)
ch_mem[:] = ch_arr[:]; del ch_arr; gc.collect()
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='r', shape=(CIN, N))

# Anchors
H=15; W=64; S=8; MAX_TR=200_000
anchor_indices = np.arange(W, N-H, S, dtype=np.int64)
anchor_ts = ts[anchor_indices]
def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)
tr_a = tmask_arr(anchor_ts, *config.SPLITS['train'])
es_a = tmask_arr(anchor_ts, *config.SPLITS['early_stop'])
te_a = tmask_arr(anchor_ts, *config.SPLITS['test'])
print(f"\nanchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}", flush=True)

rets = C[anchor_indices+H]/C[anchor_indices] - 1
tr_sel = tr_a & (np.abs(rets) > 0.0003)
if tr_sel.sum() > MAX_TR:
    rng = np.random.default_rng(42); keep = rng.choice(np.where(tr_sel)[0], MAX_TR, replace=False)
    tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
print(f"filtered tr={tr_sel.sum():,}", flush=True)

# Stride tricks
swv = sliding_window_view(ch_rm, W, axis=1)  # (CIN, N-W, W)
wa = np.transpose(swv, (1,0,2))  # (N-W, CIN, W)
win_idx = anchor_indices - W

def chunk_materialize(sel, name, batch=15000):
    picked = win_idx[sel]; n = len(picked); parts = []
    for i in range(0, n, batch):
        part = wa[picked[i:i+batch]].copy()
        parts.append(part); del part; gc.collect()
    X = np.concatenate(parts, axis=0)
    print(f"  {name}: {X.shape} ({X.nbytes/1e9:.2f}GB)", flush=True)
    return X

t0=time.time()
X_tr = chunk_materialize(tr_sel, 'X_tr')
X_es = chunk_materialize(es_a, 'X_es')
X_te = chunk_materialize(te_a, 'X_te')

y_tr = (rets[tr_sel] > 0).astype(np.int64); r_tr = rets[tr_sel].astype(np.float32)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64); ts_te = anchor_ts[te_a]

np.savez_compressed('/workspace/models_saved/seq_data_v6.npz',
    X_tr=X_tr, y_tr=y_tr, r_tr=r_tr,
    X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f"\nv6 saved! total {time.time()-t0:.0f}s  CIN={CIN} W={W}", flush=True)
print(f"TR pos={y_tr.mean():.3f}  ES pos={y_es.mean():.3f}  TE pos={y_te.mean():.3f}", flush=True)
del ch_rm, wa; gc.collect()
