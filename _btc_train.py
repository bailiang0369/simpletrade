import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
import polars as pl
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)

T0=time.time(); PAR='data/datasets'; NPY='data/splits_npy'
SEEDS=[42,49,56,63,70]
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200

def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Pre-compute ETH cross feats
log("[0] Build ETH cross feats...")
eth = pl.read_parquet(f'{PAR}/raw_ETH.parquet').sort('ts').to_pandas()
eth_ts = eth['ts'].to_numpy().astype(np.int64)
lc_e = np.log(eth['close'].to_numpy())
cf = {}
for k in (5,15,30,60,120,240,480,960):
    r=np.full(len(lc_e),np.nan); r[k:]=lc_e[k:]-lc_e[:-k]
    cf[f'ETH_lr_{k}']=r.astype(np.float32)
sr=pd.Series(lc_e)
for w in (30,60,120,240,480):
    mu=sr.rolling(w).mean().to_numpy(); sd=sr.rolling(w).std().to_numpy()
    cf[f'ETH_z_{w}']=np.where(sd>1e-9,(lc_e-mu)/sd,0.0).astype(np.float32)
lr1=np.full(len(lc_e),np.nan); lr1[1:]=lc_e[1:]-lc_e[:-1]
for w in (60,240):
    cf[f'ETH_rvol_{w}']=pd.Series(lr1).rolling(w).std().to_numpy().astype(np.float32)*100
ob_e=eth['buy_vol'].to_numpy(); os_e=eth['sell_vol'].to_numpy()
d=pd.Series(ob_e-os_e); tt=pd.Series(ob_e+os_e)
for w in (30,60):
    cf[f'ETH_cvd_{w}']=(d.rolling(w).sum()/(tt.rolling(w).sum()+1e-12)).to_numpy().astype(np.float32)
del eth, lc_e, sr, d, tt, ob_e, os_e, lr1; gc.collect()
log(f"  {len(cf)} cross feats")

# Pre-compute BTC raw ts, cross feat aligned to raw ts
btc_ts = pl.read_parquet(f'{PAR}/raw_BTC.parquet', columns=['ts']).sort('ts').to_numpy().flatten().astype(np.int64)
idx1 = np.searchsorted(eth_ts, btc_ts, side='right') - 1
idx1 = np.clip(idx1, 0, len(eth_ts)-1)
del eth_ts; gc.collect()
cross_btc = {k:v[idx1] for k,v in cf.items()}
del cf, idx1; gc.collect()
log("  Cross feats aligned to BTC raw ts")

def load_split(split, tlo, thi):
    X = np.load(f'{NPY}/BTC_h15_{split}_X.npy').astype(np.float32)
    y = np.load(f'{NPY}/BTC_h15_{split}_y.npy').astype(np.int32)
    df = pd.read_parquet(f'{PAR}/ds_BTC_h15.parquet', columns=['ts','ret_future']).sort_values('ts')
    ts = df['ts'].to_numpy().astype(np.int64); ret = df['ret_future'].to_numpy().astype(np.float32)
    del df; gc.collect()
    m = (ts>=tlo)&(ts<thi); ts_s = ts[m]; ret_s = ret[m]
    del ts, ret; gc.collect()
    idx2 = np.searchsorted(btc_ts, ts_s, side='right') - 1
    idx2 = np.clip(idx2, 0, len(btc_ts)-1)
    # keep btc_ts
    names = sorted(cross_btc.keys())
    cross = np.stack([cross_btc[n][idx2] for n in names], axis=1).astype(np.float32)
    return X, y, cross, ret_s

log("\n[1] Load splits...")
# Build btc_ts again for idx2
btc_ts = pl.read_parquet(f'{PAR}/raw_BTC.parquet', columns=['ts']).sort('ts').to_numpy().flatten().astype(np.int64)

Xtr_b, ytr_b, cross_tr, ret_tr = load_split('train', 0, TRAIN_END)
Xes_b, yes_b, cross_es, _ = load_split('early_stop', TRAIN_END, ES_END)
Xte_b, yte_b, cross_te, _ = load_split('test', META_END, 10**18)
# keep

log(f"  Base X: tr={Xtr_b.shape} es={Xes_b.shape} te={Xte_b.shape}")
log(f"  Cross:  {cross_tr.shape[1]} feats")

Xtr_c = np.hstack([Xtr_b, cross_tr]).astype(np.float32)
Xes_c = np.hstack([Xes_b, cross_es]).astype(np.float32)
Xte_c = np.hstack([Xte_b, cross_te]).astype(np.float32)
log(f"  Combined: tr={Xtr_c.shape}")

abs_r = np.abs(ret_tr); q2 = np.quantile(abs_r, 0.90)
sw = np.where(abs_r>=q2, 0.3, 1.0).astype(np.float32)

def train(Xtr,ytr,Xes,yes,Xte,yte,sw,label):
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]; t0=time.time()
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  {label:25s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate

log("\n[2] Train 4-way...")
results = {}
results['baseline']=train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,None,        'baseline (56)')
results['cross']   =train(Xtr_c,ytr_b,Xes_c,yes_b,Xte_c,yte_b,None,        '+ETH cross (73)')
results['+negw']   =train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,sw,         'baseline+negw')
results['+cross+negw']=train(Xtr_c,ytr_b,Xes_c,yes_b,Xte_c,yte_b,sw,      'cross+negw')

log(f"\n{'='*55}\nSUMMARY  (ETH h15 ref te=0.5432)\n{'='*55}")
best=max(results,key=lambda k:results[k][1]); base_te=results['baseline'][1]
for n,(e,a) in results.items():
    mark=' ★ BEST' if n==best else ''
    log(f"  {n:20s}: es={e:.4f} te={a:.4f}  Δ={a-base_te:+.4f}{mark}")
log(f"  BEST te={results[best][1]:.4f}  gap vs ETH={0.5432-results[best][1]:.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
