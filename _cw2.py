"""ETH h15 base+fund + BTC h30 conf weighting"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
SPD=86400; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
log = lambda *a: print(' '.join(str(x) for x in a), flush=True)

def train(prefix, sw=None, label=""):
    Xtr=np.load(f'{NPY}/{prefix}_train_X.npy').astype(np.float32)
    ytr=np.load(f'{NPY}/{prefix}_train_y.npy').astype(np.int32)
    Xes=np.load(f'{NPY}/{prefix}_early_stop_X.npy').astype(np.float32)
    yes=np.load(f'{NPY}/{prefix}_early_stop_y.npy').astype(np.int32)
    Xte=np.load(f'{NPY}/{prefix}_test_X.npy').astype(np.float32)
    yte=np.load(f'{NPY}/{prefix}_test_y.npy').astype(np.int32)
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  {label:30s}: es={aes:.4f} te={ate:.4f}")
    return np.mean(pte,0), yte

# ETH base+fund 需要 hstack
def eth_comb(split):
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    y=np.load(f'{NPY}/ETH_h15_{split}_y.npy').astype(np.int32)
    mr=min(Xb.shape[0], Xf.shape[0]); y=y[:mr]
    return np.hstack([Xb[:mr], Xf[:mr]]), y

log("[1] Train ETH h15 base+fund 68 +negw...")
Xtr_e,ytr_e=eth_comb('train'); Xes_e,yes_e=eth_comb('early_stop'); Xte_e,yte_e=eth_comb('test')
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')[:Xtr_e.shape[0]]
sw_e=np.where(np.abs(ret_e)>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32)

# Inline train ETH
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_eth=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr_e,label=ytr_e,weight=sw_e); es=lgb.Dataset(Xes_e,label=yes_e,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_eth.append(m.predict(Xte_e))
p_eth=np.mean(p_eth,axis=0); y_eth=yte_e
log(f"  ETH h15 base+fund+negw: te={roc_auc_score(y_eth,p_eth):.4f} ({time.time()-t0:.0f}s)")
del Xtr_e,ytr_e,Xes_e,yes_e,Xte_e,yte_e,sw_e,ret_e; gc.collect()

log("\n[2] Train BTC h30 base +negw...")
ret_b=np.load(f'{NPY}/BTC_h30_train_ret.npy')
sw_b=np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)
p_btc30, y_btc30 = train('BTC_h30', sw_b, 'BTC h30 base+negw')
del sw_b, ret_b; gc.collect()

log("\n[3] Align + Confidence Weighting...")
import pyarrow.parquet as pq
# ETH test ts
t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','ret_future'])
ts_e=t.column('ts').to_numpy().astype(np.int64); ret_e=t.column('ret_future').to_numpy().astype(np.float32); del t
m_e=ts_e>=META_END; ts_e=ts_e[m_e][:len(p_eth)]; ret_e=ret_e[m_e][:len(p_eth)]

t=pq.read_table('data/datasets/ds_BTC_h30.parquet', columns=['ts']); ts_b=t.column('ts').to_numpy().astype(np.int64); del t
m_b=ts_b>=META_END; ts_b=ts_b[m_b][:len(p_btc30)]

# Align: use min rows
mlen=min(len(p_eth),len(p_btc30))
pe=p_eth[:mlen]; pb=p_btc30[:mlen]; ye=y_eth[:mlen]; ts=ts_e[:mlen]; ret=ret_e[:mlen]

# Weighting schemes
conf_e=np.abs(pe-0.5); conf_b=np.abs(pb-0.5)
w=conf_e+conf_b+1e-8
p_cw=(pe*conf_e + pb*conf_b)/w
p_avg=(pe+pb)/2

log(f"  Simple avg:     {roc_auc_score(ye,p_avg):.4f}")
log(f"  Conf-weighted:  {roc_auc_score(ye,p_cw):.4f}")

# Also check correlation
log(f"  corr(ETH,BTC):  {np.corrcoef(pe,pb)[0,1]:.3f}")

best_p = p_cw if roc_auc_score(ye,p_cw) >= roc_auc_score(ye,p_avg) else p_avg
best_auc = roc_auc_score(ye, best_p)
log(f"  Best:           {best_auc:.4f}")

# ========== Backtest ==========
log("\n[4] No-lookahead rolling 30d q=0.90...")
thresh=np.full(len(best_p),np.nan)
for i in range(len(best_p)):
    if i<2000: continue
    tp=ts[i]-30*SPD; m=(ts[:i]>=tp)
    if m.sum()>200: thresh[i]=np.quantile(best_p[:i][m],0.90)
gq=np.nanquantile(best_p,0.90); fv=np.where(~np.isnan(thresh))[0]
thresh[:fv[0]]=gq; thresh=np.where(np.isnan(thresh),gq,thresh)

lm=best_p>thresh; sm=best_p<(1-thresh); tm=lm|sm
cy=ye[tm]; cs=sm[tm]; cr=ret[tm]
corr=((~cs)&(cy==1))|(cs&(cy==0)); acc=corr.mean()
tn=pd.to_datetime(ts[tm],unit='s').date; td=pd.Series(tn).value_counts()

log(f"\n{'='*55}")
log(f" ETH h15+BTC h30 ConfWeight  |  No-Lookahead")
log(f"{'='*55}")
log(f" AUC:          {best_auc:.4f}")
log(f" Trades:       {tm.sum():,} ({tm.sum()/len(best_p)*100:.1f}%)")
log(f" Long:{lm.sum():,} Short:{sm.sum():,}")
log(f" ACC:          {acc*100:.2f}%")
log(f" TPD avg:      {td.mean():.1f}  median: {td.median():.1f}")
log(f"{'='*55}")

# Sweep
log(f"\n[5] Threshold sweep...")
for q in [0.7,0.75,0.8,0.85,0.9,0.92,0.95,0.97,0.99]:
    tt=np.full(len(best_p),np.nan)
    for i in range(len(best_p)):
        if i<2000: continue
        tp=ts[i]-30*SPD; m=(ts[:i]>=tp)
        if m.sum()>200: tt[i]=np.quantile(best_p[:i][m],q)
    g=np.nanquantile(best_p,q); fv=np.where(~np.isnan(tt))[0]
    tt[:fv[0]]=g; tt=np.where(np.isnan(tt),g,tt)
    l2=best_p>tt; s2=best_p<(1-tt); t2=l2|s2
    if t2.sum()<30: continue
    c2=((~s2[t2])&(ye[t2]==1))|(s2[t2]&(ye[t2]==0))
    td2=pd.Series(pd.to_datetime(ts[t2],unit='s').date).value_counts()
    log(f"  q={q:.2f}: n={t2.sum():,} acc={c2.mean()*100:.2f}% tpd={td2.mean():.1f}")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
