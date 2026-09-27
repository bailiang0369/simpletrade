"""Build (67, 64) sequence from hand-crafted features, OOM-safe."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, gc, datetime as dtm, time
import polars as pl
import config
from numpy.lib.stride_tricks import sliding_window_view

t0 = time.time()
print("Loading raw ETH...", flush=True)
df = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
ts = df['ts'].to_numpy()
C_arr = df['close'].to_numpy().astype(np.float64)
N = len(ts)
print(f"  N={N}", flush=True)

print("Building features...", flush=True)
import features as fe
feats = fe.build_features(df)
del df; gc.collect()
feats_np = feats.to_numpy().astype(np.float32)
del feats; gc.collect()
C_feat = feats_np.shape[1]
print(f"  {C_feat} features, shape={feats_np.shape}", flush=True)

# Global robust z-score on train portion
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre
for j in range(C_feat):
    col = feats_np[:,j]
    valid = col[trm & ~np.isnan(col)]
    if len(valid) > 10:
        lo, hi = np.percentile(valid, 0.5), np.percentile(valid, 99.5)
        feats_np[:,j] = np.clip(col, lo, hi)
feats_np = np.nan_to_num(feats_np, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  normalized range=[{feats_np.min():.2f},{feats_np.max():.2f}]", flush=True)

# memmap ch_rm (C_feat, N)
ch_path = '/tmp/ch_manual.bin'
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='w+', shape=(C_feat, N))
ch_rm[:] = feats_np.T[:]; del feats_np; gc.collect()
ch_rm = np.memmap(ch_path, dtype=np.float32, mode='r', shape=(C_feat, N))

H=15; W=64; S=16; MAX_TR=50_000
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

rets = C_arr[anchor_indices+H]/C_arr[anchor_indices] - 1
tr_sel = tr_a & (np.abs(rets) > 0.0003)
if tr_sel.sum() > MAX_TR:
    rng = np.random.default_rng(42); keep = rng.choice(np.where(tr_sel)[0], MAX_TR, replace=False)
    tr_sel = np.zeros_like(tr_sel, dtype=bool); tr_sel[keep] = True
print(f"filtered tr={tr_sel.sum():,}", flush=True)

del C_arr; gc.collect()

swv = sliding_window_view(ch_rm, W, axis=1)  # (C_feat, N-W, W)
wa = np.transpose(swv, (1,0,2))
win_idx = anchor_indices - W

def chunk_materialize(sel, name, batch=8000):
    picked = win_idx[sel]; n = len(picked); parts = []
    for i in range(0, n, batch):
        part = wa[picked[i:i+batch]].copy()
        parts.append(part); del part; gc.collect()
    X = np.concatenate(parts, axis=0)
    print(f"  {name}: {X.shape} ({X.nbytes/1e9:.2f}GB)", flush=True)
    return X

print("Materializing...", flush=True)
t0 = time.time()
X_tr = chunk_materialize(tr_sel, 'X_tr')
X_es = chunk_materialize(es_a, 'X_es')
X_te = chunk_materialize(te_a, 'X_te')

y_tr = (rets[tr_sel] > 0).astype(np.int64); r_tr = rets[tr_sel].astype(np.float32)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64); ts_te = anchor_ts[te_a]

np.savez('/workspace/models_saved/seq_manual.npz',
    X_tr=X_tr, y_tr=y_tr, r_tr=r_tr, X_es=X_es, y_es=y_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
print(f"\nSAVED seq_manual.npz! C={C_feat} W={W} total={time.time()-t0:.0f}s", flush=True)
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape}", flush=True)
print(f"TR pos={y_tr.mean():.3f} ES pos={y_es.mean():.3f} TE pos={y_te.mean():.3f}", flush=True)
