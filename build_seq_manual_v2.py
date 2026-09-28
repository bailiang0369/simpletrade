"""Build seq_manual v2 — memory safe version (no sliding_window_view on full data)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, gc, datetime as dtm, time
import polars as pl
import config

t0 = time.time()
print("Loading raw ETH...", flush=True)
df = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
ts = df['ts'].to_numpy().astype(np.int64)
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
print(f"  {C_feat} features", flush=True)

# Robust z-score (train portion only)
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
trm = ts < tre
for j in range(C_feat):
    col = feats_np[:,j]
    valid = col[trm & ~np.isnan(col)]
    if len(valid) > 10:
        lo, hi = np.percentile(valid, 0.5), np.percentile(valid, 99.5)
        feats_np[:,j] = np.clip(col, lo, hi)
feats_np = np.nan_to_num(feats_np, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  range=[{feats_np.min():.2f},{feats_np.max():.2f}]", flush=True)

H=15; W=64; S=32  # S=32: fewer anchors (was 16), safer
MAX_TR=80_000

def tmask_arr(ts_, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_>=a)&(ts_<b)

# Determine anchor indices
anchor_indices = np.arange(W, N-H, S, dtype=np.int64)
anchor_ts = ts[anchor_indices]
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

# Build windows DIRECTLY without sliding_window_view on full array
# feats_np is (N, C_feat) — we index it directly
def make_windows(sel_indices, name, batch=1024):
    """sel_indices: indices into anchor_indices (not feats_np)."""
    picked = anchor_indices[sel_indices] - W  # start of each window
    n = len(picked)
    result = np.zeros((n, C_feat, W), dtype=np.float32)
    for i in range(0, n, batch):
        bi = picked[i:i+batch]
        for offset in range(W):
            result[i:i+batch, :, offset] = feats_np[bi + offset, :]
        if i % (batch*10) == 0:
            mem_mb = result.nbytes/1e6
            print(f"    {name}: {i}/{n} ({mem_mb:.0f}MB)", flush=True)
    print(f"  {name}: {result.shape} ({result.nbytes/1e9:.2f}GB)", flush=True)
    return result

print("Building windows...", flush=True)
t_build = time.time()
X_tr = make_windows(tr_sel, 'X_tr')
gc.collect()
X_es = make_windows(es_a, 'X_es')
gc.collect()
X_te = make_windows(te_a, 'X_te')
gc.collect()
print(f"  windows built in {time.time()-t_build:.0f}s", flush=True)

y_tr = (rets[tr_sel] > 0).astype(np.int64)
y_es = (rets[es_a] > 0).astype(np.int64)
y_te = (rets[te_a] > 0).astype(np.int64)
ts_tr = anchor_ts[tr_sel]
ts_es = anchor_ts[es_a]
ts_te = anchor_ts[te_a]

out = '/workspace/models_saved/seq_manual.npz'
np.savez(out,
    X_tr=X_tr, y_tr=y_tr, ts_tr=ts_tr,
    X_es=X_es, y_es=y_es, ts_es=ts_es,
    X_te=X_te, y_te=y_te, ts_te=ts_te)
del X_tr, X_es, X_te; gc.collect()
print(f"\n✅ SAVED {out}  total={time.time()-t0:.0f}s", flush=True)
