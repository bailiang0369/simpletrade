"""sklearn MLP on raw OHLCV windows — raw sequence → predict."""
import numpy as np, pandas as pd, time, gc, datetime as dtm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('[1] load raw OHLCV...', flush=True)
eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
close = eth['close'].values.astype(np.float32)
open_p = eth['open'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low = eth['low'].values.astype(np.float32)
bv = eth['buy_vol'].values.astype(np.float32)
sv = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts_vals = eth['ts'].values.astype(np.int64)

# BTC
btc = pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')
btc_idx = pd.Index(btc['ts'].values); eth_idx = pd.Index(ts_vals)
reorder = btc_idx.get_indexer(eth_idx)
reorder = np.where(reorder < 0, np.arange(len(reorder)), reorder)
btc_c = btc['close'].values.astype(np.float32)[reorder]

N = len(close)
H = 30; SEG = 64; STRIDE = 32

# Labels
labels = np.full(N, np.nan, np.float32)
labels[:-H] = close[H:]/close[:-H] - 1

print(f'  N={N}, building windows...', flush=True)

# Build normalized OHLCV windows — inline to save memory
rows = []; targets = []; tss = []
for start in range(0, N - SEG - H, STRIDE):
    end = start + SEG
    if np.isnan(labels[end]): continue
    c0 = close[start]
    if c0 <= 0: continue
    w = np.zeros(SEG * 7, np.float32)
    # OHLC relative to c0
    w[0::7] = open_p[start:end] / c0
    w[1::7] = high[start:end] / c0
    w[2::7] = low[start:end] / c0
    w[3::7] = close[start:end] / c0
    # log(buy_ratio)
    tot = bv[start:end] + sv[start:end]
    w[4::7] = np.log(np.maximum(bv[start:end] / np.maximum(tot, 1), 1e-6))
    # log(volume)
    w[5::7] = np.log(np.maximum(tot, 1e-6))
    # BTC ret relative
    b0 = btc_c[start]; b0 = max(b0, 1e-6)
    w[6::7] = btc_c[start:end] / b0
    
    rows.append(w)
    targets.append(1 if labels[end] > 0 else 0)
    tss.append(ts_vals[end])

X_all = np.array(rows, np.float32)
y_all = np.array(targets, np.int64)
ts_all = np.array(tss, np.int64)
del rows; gc.collect()
print(f'  windows: {X_all.shape}  pos_rate={y_all.mean():.3f}  ({time.time()-t0:.0f}s)', flush=True)

# Split
def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all>=a)&(ts_all<b)
tr_idx = np.where(ts_mask('2020-01-01','2024-06-30'))[0]
es_idx = np.where(ts_mask('2024-06-30','2024-09-30'))[0]
te_idx = np.where(ts_mask('2025-09-30','2026-08-29'))[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# Scale features
print('\n[2] scale...', flush=True)
X_tr = X_all[tr_idx]
# Compute scale stats on training only
tr_mean = X_tr.mean(axis=0); tr_std = np.maximum(X_tr.std(axis=0), 1e-6)
X_all = (X_all - tr_mean) / tr_std
del X_tr, tr_mean, tr_std; gc.collect()
print(f'  scaled')

# Train MLP (sklearn — slow but works)
print('\n[3] train MLP...', flush=True)
X_tr = X_all[tr_idx]; y_tr = y_all[tr_idx]
X_es = X_all[es_idx]; y_es = y_all[es_idx]
X_te = X_all[te_idx]; y_te = y_all[te_idx]
del X_all; gc.collect()

# Small subset for speed — sklearn MLP on 2M rows is way too slow
# Use every 5th row
sub = np.random.RandomState(42).choice(len(tr_idx), size=min(500000, len(tr_idx)), replace=False)
X_tr_sub = X_tr[sub]; y_tr_sub = y_tr[sub]
del X_tr, y_tr; gc.collect()
print(f'  training subset: {len(X_tr_sub):,} rows')

for hidden in [(64,32), (128,64), (256,128,64)]:
    print(f'\n  hidden={hidden}...', flush=True)
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', 
                        alpha=0.001, max_iter=50, early_stopping=True,
                        validation_fraction=0.1, random_state=42, verbose=False,
                        n_iter_no_change=10)
    mlp.fit(X_tr_sub, y_tr_sub)
    pv_es = mlp.predict_proba(X_es)[:,1]
    pv_te = mlp.predict_proba(X_te)[:,1]
    auc_es = roc_auc_score(y_es, pv_es)
    auc_te = roc_auc_score(y_te, pv_te)
    
    top1 = pv_te.argsort()[-max(1,int(len(pv_te)*0.01)):]
    acc1 = y_te[top1].mean()*100
    
    line = f'  hidden={hidden}  es={auc_es:.4f} te={auc_te:.4f} top1%={acc1:.1f}%  ({time.time()-t0:.0f}s)'
    print(line, flush=True)

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
