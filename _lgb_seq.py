"""LightGBM on 256-close sequence + 256-stoch_k14 sequence (flattened)
直接喂原始形态结构, 不用 ret. 对比 LightGBM on ret-based feats.
"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
log=lambda *a: print(' '.join(str(x) for x in a), )

# ============= 1. Build sequence features =============
print("[1] Build seq feats (256 close + 256 stoch_k14)...")
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts').reset_index(drop=True)
ds  = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts').reset_index(drop=True)
L14 = raw['low'].rolling(14,min_periods=1).min(); H14 = raw['high'].rolling(14,min_periods=1).max()
raw['sk14'] = np.where(H14-L14>1e-9, (raw['close']-L14)/(H14-L14)*100, 50.0).astype(np.float32)
raw['close'] = raw['close'].astype(np.float32)
raw_ts=raw['ts'].to_numpy().astype(np.int64); ds_ts=ds['ts'].to_numpy().astype(np.int64)
CLOSE=raw['close'].to_numpy(); SK14=raw['sk14'].to_numpy()
del raw; gc.collect()

idx = np.searchsorted(raw_ts, ds_ts)
idx = np.clip(idx, 512, len(CLOSE)-1)  # need window
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

W=256  # window

def build_seq_arr(indices):
    N=len(indices); out=np.empty((N, W*2), dtype=np.float32)
    for i in range(N):
        end=indices[i]; s=end-W+1
        c=CLOSE[s:end+1]; k=SK14[s:end+1]
        c0=c[0]; c=np.where(c0>1e-9, c/c0-1.0, 0.0)  # normalize by first close
        out[i,:W]=c; out[i,W:]=k/100.0
    return out

def split(lo, hi):
    m=(ds_ts>=lo)&(ds_ts<hi)
    return idx[m], ds.loc[m,'label'].to_numpy().astype(np.int32)

tr_idx, tr_y = split(0, TRAIN_END)
es_idx, es_y = split(TRAIN_END, ES_END)
te_idx, te_y = split(META_END, 10**18)
log(f"  tr={len(tr_idx):,} es={len(es_idx):,} te={len(te_idx):,}")

# Save to disk (memory)
print("  Build train seq arr...", )
Xtr = build_seq_arr(tr_idx); np.save(f'{NPY}/_seq_tr.npy', Xtr); del Xtr; gc.collect()
print("  Build early_stop seq arr...", )
Xes = build_seq_arr(es_idx); np.save(f'{NPY}/_seq_es.npy', Xes); del Xes; gc.collect()
print("  Build test seq arr...", )
Xte = build_seq_arr(te_idx); np.save(f'{NPY}/_seq_te.npy', Xte); del Xte; gc.collect()
print("  Saved.", )

# Also save labels for seq splits (对齐, 因为 idx 有 clip)
np.save(f'{NPY}/_seq_tr_y.npy', tr_y[:len(tr_idx)])
np.save(f'{NPY}/_seq_es_y.npy', es_y[:len(es_idx)])
np.save(f'{NPY}/_seq_te_y.npy', te_y[:len(te_idx)])
log(f"\n  ★ 序列特征 (close+stoch_k14 256根, 共 {W*2} 维, 全是形态结构, 零 ret)", )

# ============= 2. Train =============
log(f"\n[2] LightGBM on seq ({W*2} feats)...")
Xtr=np.load(f'{NPY}/_seq_tr.npy'); ytr=np.load(f'{NPY}/_seq_tr_y.npy')
Xes=np.load(f'{NPY}/_seq_es.npy'); yes=np.load(f'{NPY}/_seq_es_y.npy')
Xte=np.load(f'{NPY}/_seq_te.npy'); yte=np.load(f'{NPY}/_seq_te_y.npy')
log(f"  shapes: tr={Xtr.shape} te={Xte.shape}")

# negw on extreme |ret|... wait we don't have ret! Let's just not use negw this time (pure形态)
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':127,'min_child_samples':200,
        'feature_fraction':0.6,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}

pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte))
p_seq=np.mean(pte,axis=0); del Xtr,ytr,Xes,yes,pte; gc.collect()
auc_seq=roc_auc_score(yte,p_seq)
log(f"  ★ SEQ AUC = {auc_seq:.4f}  (ret-based LightGBM baseline = 0.5428) ({time.time()-t0:.0f}s)")

# ============= 3. Also: seq + base ret features =============
log(f"\n[3] SEQ + BASE ret feats (hybrid)...")
Xb_tr=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
Xb_es=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
Xb_te=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)

# 对齐 len (idx clip 导致部分行跳过)
mr_tr=min(Xb_tr.shape[0], Xtr.shape[0]); mr_es=min(Xb_es.shape[0], Xes.shape[0]); mr_te=min(Xb_te.shape[0], Xte.shape[0])
Xtr2=np.hstack([Xb_tr[:mr_tr], np.load(f'{NPY}/_seq_tr.npy')[:mr_tr]])
Xes2=np.hstack([Xb_es[:mr_es], np.load(f'{NPY}/_seq_es.npy')[:mr_es]])
Xte2=np.hstack([Xb_te[:mr_te], np.load(f'{NPY}/_seq_te.npy')[:mr_te]])
ytr2=ytr[:mr_tr]; yes2=yes[:mr_es]; yte2=yte[:mr_te]
del Xb_tr,Xb_es,Xb_es,Xb_te; gc.collect()

log(f"  shapes: tr={Xtr2.shape} ({56}+{W*2} feats)")
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr2,label=ytr2); es=lgb.Dataset(Xes2,label=yes2,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte2))
p_hyb=np.mean(pte,axis=0); del Xtr2,Xes2,Xte2,pte; gc.collect()
auc_hyb=roc_auc_score(yte2,p_hyb)
log(f"  ★ HYBRID (seq+base) AUC = {auc_hyb:.4f} ({time.time()-t0:.0f}s)")

# ============= 4. tpd sweep for seq model =============
log(f"\n[4] SEQ model tpd sweep (global q upper bound)...")
y=yte[:len(p_seq)] if len(p_seq)<len(yte) else yte
log(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'n':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985, 0.998, 0.0005):
    th=np.quantile(p_seq,q); lm=p_seq>th; sm=p_seq<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    tpd=n/332
    diff=acc-65
    mark='◀' if abs(tpd-14.4)<0.5 else ''
    if 8<=tpd<=22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {n:6d}  {diff:+7.1f}pp {mark}", )
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best=(acc,q,tpd)
if best: print(f"\n★ SEQ tpd≈14.4: ACC={best[0]:.1f}% q={best[1]:.4f} tpd={best[2]:.1f}", )

# ============= 5. Compare =============
log(f"\n{'='*60}")
log(f"  对比")
log(f"{'='*60}")
log(f"  ret-based LightGBM (56 feats):    AUC=0.5428")
log(f"  SEQ-only (256 close+256 sk14):    AUC={auc_seq:.4f}  ({(auc_seq-0.5428)*100:+.2f}pp)")
log(f"  SEQ+BASE hybrid:                   AUC={auc_hyb:.4f}  ({(auc_hyb-0.5428)*100:+.2f}pp)")
log(f"  SEQ tpd=14.4 ACC ceiling:          ≈{best[0]:.1f}%")
log(f"  差距 65%:                          {65-best[0]:.1f}pp")
log(f"\nTOTAL: {time.time()-T0:.0f}s", )
