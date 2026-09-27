"""v4: close-normalized OHLCV + multi-horizon features (no z-score on raw series)."""
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
BTC_C = btc['close'].values.astype(np.float64)[br]
del eth, btc; gc.collect(); N = len(C)

# Log returns at multiple horizons
lr1 = np.zeros(N, dtype=np.float64); lr1[1:] = np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8))
LR = np.zeros((4,N), dtype=np.float64)
for i,w in enumerate([5,15,30,60]): LR[i,w:] = np.log(np.maximum(C[w:],1e-8)/np.maximum(C[:-w],1e-8))
B_lr1 = np.zeros(N, dtype=np.float64); B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))

# Rolling std of lr
def rstd(x, w):
    s = pd.Series(x).rolling(w, min_periods=max(5, w//3)).std().values.astype(np.float32)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
vol = np.stack([rstd(lr1, 15), rstd(lr1, 60), rstd(lr1, 240), rstd(lr1, 1440)], axis=0)

# ==========================================
# **KEY IDEA**: 12 channels, time-aligned to each anchor
# Each channel at time t=anchor is built using [anchor-W+1 ... anchor] raw values
# No global normalization (z-score would lose dynamics)
# Instead, close-normalize: OHLC -> ratio vs anchor-close; vol -> local pct
# ==========================================

def make_channel(arr, mode, anchor_ts, window, ts_all):
    """
    Make a time-series channel aligned to anchor timestamps.
    For each anchor, pull arr[anchor-W+1 ... anchor].
    mode=0: raw log-return difference from anchor close (all bars as %diff from anchor close)
    mode=1: log returns (already stationary)
    mode=2: close-normalized OHLC ratio
    """
    return None  # unused

# Simplest approach: just pull the raw arr values and normalize per-window
# Let's do: channels are STATIONARY by construction
# lr1 / lr5 / lr15 / lr30 / lr60: already stationary (log-return)
# vol15 / vol60 / vol240 / vol1440: already stationary (rolling std of lr)
# cvd (buy-sell)/total: already [-1,1]
# funding: clamp then normalize within each window
# BTC lr1 + BTC vol60
# body_pct = |close-open|/(high-low): [0,1]

TV = BV + SV; TV_s = np.where(TV > 0, TV, 1.0)
CVD = (BV - SV) / TV_s
body = np.abs(C - O) / np.where(Hv > Lv, Hv - Lv, 1.0)

f_lo, f_hi = np.nanpercentile(FUND, 0.5), np.nanpercentile(FUND, 99.5)
FC = np.clip(FUND, f_lo, f_hi)

# Per-channel normalize so each window has mean~0, std~1
def norm_per_window(X):
    """X shape (W,) → (W,) normalized"""
    m = X.mean() + 1e-8; s = X.std() + 1e-8
    return (X - m) / s

channels_list = [lr1, LR[0], LR[1], LR[2], LR[3],      # 5
                 vol[0], vol[1], vol[2], vol[3],       # 4
                 CVD, FC, B_lr1]                        # 3
CIN = len(channels_list)
ch_arr = np.stack(channels_list, axis=0).astype(np.float32)  # (CIN, N)
print(f"ch_arr shape={ch_arr.shape}", flush=True)
ch_arr = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0)

# memmap
ch_path = '/tmp/ch_v4.bin'
ch_mem = np.memmap(ch_path, dtype=np.float32, mode='w+', shape=ch_arr.shape)
ch_mem[:] = ch_arr[:]; del ch_arr; gc.collect()
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='r', shape=(CIN, N))

# Anchors
H=15; W=64; S=8; MAX_TR=400_000
anchor_indices = np.arange(W, N-H, S, dtype=np.int64)
anchor_ts = ts[anchor_indices]
def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)
tr_a = tmask_arr(anchor_ts, *config.SPLITS['train'])
es_a = tmask_arr(anchor_ts, *config.SPLITS['early_stop'])
te_a = tmask_arr(anchor_ts, *config.SPLITS['test'])
print(f"anchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}", flush=True)

rets = C[anchor_indices+H] / C[anchor_indices] - 1
tr_sel = tr_a & (np.abs(rets) > 0.0003)
if tr_sel.sum() > MAX_TR:
    rng = np.random.default_rng(42); keep = rng.choice(np.where(tr_sel)[0], MAX_TR, replace=False)
    tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
print(f"filtered tr={tr_sel.sum():,}", flush=True)

# Stride tricks
swv = sliding_window_view(ch_rm, W, axis=1)  # (CIN, N-W, W)
wa = np.transpose(swv, (1,0,2))  # (N-W, CIN, W)
win_idx = anchor_indices - W

# Per-window normalization (CRITICAL for TCN to learn trends!)
def chunk_materialize(sel, name, batch=20000):
    picked = win_idx[sel]; n = len(picked); parts = []
    for i in range(0, n, batch):
        part = wa[picked[i:i+batch]].copy()  # (b, CIN, W)
        # Per-channel per-window z-score
        m = part.mean(axis=2, keepdims=True)
        s = part.std(axis=2, keepdims=True) + 1e-6
        part = (part - m) / s
        parts.append(part); del part, m, s; gc.collect()
    X = np.concatenate(parts, axis=0)
    print(f"  {name}: {X.shape} ({X.nbytes/1e9:.2f}GB) mean={X.mean():.4f} std={X.std():.4f}", flush=True)
    return X

t0=time.time()
X_tr = chunk_materialize(tr_sel, 'X_tr')
X_es = chunk_materialize(es_a, 'X_es')
X_te = chunk_materialize(te_a, 'X_te')

y_tr = (rets[tr_sel] > 0).astype(np.int64); r_tr = rets[tr_sel].astype(np.float32)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64); ts_te = anchor_ts[te_a]

np.savez_compressed('/workspace/models_saved/seq_data_v4.npz',
    X_tr=X_tr, y_tr=y_tr, r_tr=r_tr,
    X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f"\nv4 saved! total {time.time()-t0:.0f}s  CIN={CIN}", flush=True)
print(f"TR pos={y_tr.mean():.3f}  ES pos={y_es.mean():.3f}  TE pos={y_te.mean():.3f}", flush=True)
del ch_rm, wa; gc.collect()
