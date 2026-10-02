"""v7: Dense stride=2, full train, NO |ret| filter, float16 storage, pre-alloc memmap-based chunking.

Memory-safe design:
  - All stationary channels as float32 (not float64)
  - Single memmap ch_rm for (CIN, N) channels, never fully loaded in RAM
  - Pre-allocate each split's X buffer, write chunks directly (no concatenate OOM)
  - Per-anchor windows extracted via ch_rm fancy indexing of individual time rows
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import resource, time, os, gc, datetime as dtm
import numpy as np, pandas as pd
import config

def mem_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

t0=time.time()
print(f'[0.0s] start: {mem_gb():.2f} GB')

# ---- Load raw parquet ----
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
br = np.clip(btc['ts'].values.searchsorted(eth['ts'].values, side='right')-1, 0, len(btc)-1)
ts = eth['ts'].values.astype(np.int64)
C = eth['close'].values.astype(np.float64); O = eth['open'].values.astype(np.float64)
Hv = eth['high'].values.astype(np.float64); Lv = eth['low'].values.astype(np.float64)
BV = eth['buy_vol'].values.astype(np.float64); SV = eth['sell_vol'].values.astype(np.float64)
FUND = eth['funding'].values.astype(np.float64)
BTC_C = btc['close'].values.astype(np.float64)[br]
del eth, btc; gc.collect()
N = len(C)
print(f'[{time.time()-t0:.1f}s] loaded N={N}: {mem_gb():.2f} GB')

# ---- Stationary channels (ALL float32 to save mem) ----
lr1 = np.zeros(N, dtype=np.float32); lr1[1:] = np.log(np.maximum(C[1:],1e-8)/np.maximum(C[:-1],1e-8)).astype(np.float32)
LR15 = np.zeros(N, dtype=np.float32); LR15[15:] = np.log(np.maximum(C[15:],1e-8)/np.maximum(C[:-15],1e-8)).astype(np.float32)
LR60 = np.zeros(N, dtype=np.float32); LR60[60:] = np.log(np.maximum(C[60:],1e-8)/np.maximum(C[:-60],1e-8)).astype(np.float32)
B_lr1 = np.zeros(N, dtype=np.float32); B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8)).astype(np.float32)
TV = BV + SV; TV_s = np.where(TV > 0, TV, 1.0)
CVD = ((BV-SV)/TV_s).astype(np.float32)
body = (np.abs(C-O)/np.where(Hv>Lv, Hv-Lv, 1.0)).astype(np.float32)
vol = np.log1p(TV).astype(np.float32)
def rstd(x, w):
    return np.nan_to_num(pd.Series(x.astype(np.float64)).rolling(w, min_periods=max(5,w//3)).std().values, nan=0.0).astype(np.float32)
vol15 = rstd(lr1.astype(np.float64), 15); vol60 = rstd(lr1.astype(np.float64), 60); vol240 = rstd(lr1.astype(np.float64), 240)
del BV, SV, TV, TV_s, BTC_C; gc.collect()
print(f'[{time.time()-t0:.1f}s] stationaries: {mem_gb():.2f} GB')

# ---- Normalization (train-only stats, no leakage) ----
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre

def robust_z(x):
    lo, hi = np.percentile(x[trm], 0.5), np.percentile(x[trm], 99.5)
    xc = np.clip(x, lo, hi)
    mu = xc[trm].mean(); sd = xc[trm].std() + 1e-6
    return ((xc - mu) / sd).astype(np.float32)

def scale_unit(x):
    lo, hi = np.percentile(x[trm], 0.5), np.percentile(x[trm], 99.5)
    return np.clip(x, lo, hi).astype(np.float32)

ch_list = [
    robust_z(lr1), robust_z(LR15), robust_z(LR60), robust_z(B_lr1),
    robust_z(vol15), robust_z(vol60), robust_z(vol240),
    scale_unit(CVD), scale_unit(body),
    robust_z(vol), robust_z(FUND.astype(np.float32)),
    robust_z(((C-O)/np.maximum(C,1e-8)).astype(np.float32)),
    robust_z(((Hv-C)/np.maximum(Hv,1e-8)).astype(np.float32)),
    robust_z(((C-Lv)/np.maximum(Lv,1e-8)).astype(np.float32)),
]
CIN = len(ch_list)
ch_arr = np.stack(ch_list, axis=0).astype(np.float32)
ch_arr = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0)
del ch_list, lr1, LR15, LR60, B_lr1, vol15, vol60, vol240, CVD, body, vol, FUND, trm; gc.collect()
print(f'[{time.time()-t0:.1f}s] ch_arr={ch_arr.shape} ({ch_arr.nbytes/1e9:.2f}GB): {mem_gb():.2f} GB')

# ---- Channels memmap ----
ch_path = '/tmp/ch_v7.bin'
ch_mem = np.memmap(ch_path, dtype=np.float32, mode='w+', shape=ch_arr.shape)
ch_mem[:] = ch_arr[:]
del ch_arr, ch_mem; gc.collect()
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='r', shape=(CIN, N))
print(f'[{time.time()-t0:.1f}s] memmap ready: {mem_gb():.2f} GB')

# ---- Anchors (dense stride=2, no |ret| filter) ----
H = 15; W = 64; S = 2; MAX_TR = 3_000_000
anchor_indices = np.arange(W, N-H, S, dtype=np.int64)
anchor_ts = ts[anchor_indices]
del ts; gc.collect()
def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)
tr_a = tmask_arr(anchor_ts, *config.SPLITS['train'])
es_a = tmask_arr(anchor_ts, *config.SPLITS['early_stop'])
te_a = tmask_arr(anchor_ts, *config.SPLITS['test'])
print(f'anchors: tr={tr_a.sum():,} es={es_a.sum():,} te={te_a.sum():,}')

rets = C[anchor_indices+H] / C[anchor_indices] - 1
del C; gc.collect()
tr_sel = tr_a.copy()  # NO |ret| filter
if tr_sel.sum() > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(np.where(tr_sel)[0], MAX_TR, replace=False)
    tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
print(f'tr after cap={tr_sel.sum():,}')
print(f'[{time.time()-t0:.1f}s] anchors ready: {mem_gb():.2f} GB')

# ---- Mem-safe chunk materialization: PRE-ALLOCATE + DIRECT WRITE ----
def build_X(sel, name, batch=8000):
    anchors = anchor_indices[sel]
    n = len(anchors)
    X = np.empty((n, CIN, W), dtype=np.float16)  # pre-allocate ONCE
    offset = 0
    for i in range(0, n, batch):
        a_batch = anchors[i:i+batch]
        # Extract W time rows from memmap → (CIN, bs, W) via list comprehension (loop over W=64)
        slices = [ch_rm[:, a_batch - (W-1-j)] for j in range(W)]
        part = np.stack(slices, axis=-1).astype(np.float16)  # (CIN, bs, W)
        part = part.transpose(1, 0, 2)                         # view → (bs, CIN, W)
        actual = len(a_batch)
        X[offset:offset+actual] = part                          # copy into pre-alloc buffer
        offset += actual
        del slices, part
        if (i // batch) % 200 == 0:
            print(f'  [{name}] chunk {i//batch}  mem={mem_gb():.2f} GB', flush=True)
    print(f'  {name}: X={X.shape} ({X.nbytes/1e9:.2f}GB)  mem={mem_gb():.2f} GB', flush=True)
    return X

# Labels — build first, before any sel masks get deleted
y_tr = (rets[tr_sel] > 0).astype(np.int64)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64)
ts_te = anchor_ts[te_a]

# Build each split separately → peak RAM ≈ max split size + base
t1 = time.time()
X_tr = build_X(tr_sel, 'X_tr')
del tr_sel, tr_a; gc.collect()
X_es = build_X(es_a, 'X_es')
del es_a; gc.collect()
X_te = build_X(te_a, 'X_te')
del te_a; gc.collect()
del rets, anchor_indices, anchor_ts; gc.collect()
print(f'[{time.time()-t0:.1f}s] all splits built ({time.time()-t1:.0f}s): {mem_gb():.2f} GB')
print(f'labels: TR pos={y_tr.mean():.3f} ES pos={y_es.mean():.3f} TE pos={y_te.mean():.3f}')

# ---- Save ----
out = '/workspace/models_saved/seq_data_v7.npz'
np.savez_compressed(out,
    X_tr=X_tr, y_tr=y_tr,
    X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f'[{time.time()-t0:.1f}s] SAVED → {out}  {mem_gb():.2f} GB')
print(f'  filesize: {os.path.getsize(out)/1e9:.2f} GB')
del ch_rm; gc.collect()
