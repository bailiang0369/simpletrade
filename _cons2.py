import numpy as np, pandas as pd, time, gc, sys, os, warnings
import lightgbm as lgb
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
SPD=86400; META_END=1759363200; NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def quick_train(prefix, sw=None):
    Xtr=np.load(f'{NPY}/{prefix}_train_X.npy').astype(np.float32)
    ytr=np.load(f'{NPY}/{prefix}_train_y.npy').astype(np.int32)
    Xes=np.load(f'{NPY}/{prefix}_early_stop_X.npy').astype(np.float32)
    yes=np.load(f'{NPY}/{prefix}_early_stop_y.npy').astype(np.int32)
    Xte=np.load(f'{NPY}/{prefix}_test_X.npy').astype(np.float32)
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte=[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte))
    return np.mean(pte,axis=0)

# Build ETH comb
for split in ['train','early_stop','test']:
    Xb=np.load(f'{NPY}/ETH_h15_{split}_X.npy').astype(np.float32)
    Xf=np.load(f'{NPY}/ETH_h15_fund_{split}_X.npy').astype(np.float32)
    mr=min(Xb.shape[0],Xf.shape[0])
    np.save(f'{NPY}/_ec_{split}_X.npy', np.hstack([Xb[:mr],Xf[:mr]]))
ret_e=np.load(f'{NPY}/ETH_h15_train_ret.npy')
np.save(f'{NPY}/_sw_e.npy', np.where(np.abs(ret_e[:np.load(f'{NPY}/_ec_train_X.npy').shape[0]])>=np.quantile(np.abs(ret_e),0.90),0.3,1.0).astype(np.float32))
ret_b=np.load(f'{NPY}/BTC_h15_train_ret.npy')
np.save(f'{NPY}/_sw_b.npy', np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32))
ret_b30=np.load(f'{NPY}/BTC_h30_train_ret.npy')
np.save(f'{NPY}/_sw_b30.npy', np.where(np.abs(ret_b30)>=np.quantile(np.abs(ret_b30),0.90),0.3,1.0).astype(np.float32))
log("Files ready")

log("[1] Train ETH..."); p_eth=quick_train('_ec', np.load(f'{NPY}/_sw_e.npy'))
log("[2] Train BTC h15..."); p_b15=quick_train('BTC_h15', np.load(f'{NPY}/_sw_b.npy'))
log("[3] Train BTC h30..."); p_b30=quick_train('BTC_h30', np.load(f'{NPY}/_sw_b30.npy'))

ml=min(len(p_eth),len(p_b15),len(p_b30))
p_eth=p_eth[:ml]; p_b15=p_b15[:ml]; p_b30=p_b30[:ml]
log(f"Aligned: {ml:,}  corr(E,B15)={np.corrcoef(p_eth,p_b15)[0,1]:.3f}")

t=pq.read_table('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future'])
ts=t.column('ts').to_numpy().astype(np.int64); y=t.column('label').to_numpy().astype(np.int32); ret=t.column('ret_future').to_numpy().astype(np.float32); del t
mm=ts>=META_END; ts=ts[mm][:ml]; y=y[mm][:ml]; ret=ret[mm][:ml]

def eval_c(pe,pb,qe,qb):
    N=len(pe); th=np.full(N,np.nan)
    for i in range(2000,N):
        tp=ts[i]-30*SPD; m2=(ts[:i]>=tp)
        if m2.sum()>200: th[i]=np.quantile(pe[:i][m2],qe)
    gq=np.nanquantile(pe,qe); fv=np.where(~np.isnan(th))[0]
    th[:fv[0]]=gq; th=np.where(np.isnan(th),gq,th)
    le=pe>th; se=pe<(1-th); sig=le|se
    if qb is None: bo=(le&(pb>0.5))|(se&(pb<0.5))
    else:
        thb=np.full(N,np.nan)
        for i in range(2000,N):
            tp=ts[i]-30*SPD; m2=(ts[:i]>=tp)
            if m2.sum()>200: thb[i]=np.quantile(pb[:i][m2],qb)
        gq=np.nanquantile(pb,qb); fv=np.where(~np.isnan(thb))[0]
        thb[:fv[0]]=gq; thb=np.where(np.isnan(thb),gq,thb)
        lb=pb>thb; sb=pb<(1-thb)
        bo=(le&lb)|(se&sb)
    tm=sig&bo
    if tm.sum()<10: return 0,0,0
    ss=se[tm]; acc=(((~ss)&(y[tm]==1))|(ss&(y[tm]==0))).mean()*100
    tn=pd.to_datetime(ts[tm],unit='s').date; td=pd.Series(tn).value_counts()
    return tm.sum(), acc, td.mean()

log(f"\n{'='*60}"); log(f" CONSENSUS (ETH→BTC filter, no-lookahead rolling 30d)"); log(f"{'='*60}")
log(f"\n[A] ETH alone BASELINE:")
for q in [0.85,0.90,0.92,0.95,0.97,0.99]:
    n,a,t=eval_c(p_eth,p_eth,q,None); log(f"  q={q:.2f}: n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\n[B] ETH=0.90 + BTC h15 DIRECTION filter:")
n,a,t=eval_c(p_eth,p_b15,0.90,None); log(f"  → n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\n[C] ETH=0.90 + BTC h15 ALSO filtered:")
for qb in [0.6,0.7,0.75,0.8,0.85]:
    n,a,t=eval_c(p_eth,p_b15,0.90,qb); log(f"  BTC={qb:.2f}: n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\n[D] ETH=0.90 + BTC h30 DIRECTION:")
n,a,t=eval_c(p_eth,p_b30,0.90,None); log(f"  → n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\n[E] Higher ETH + BTC h15 dir:")
for qe in [0.92,0.94,0.95,0.96,0.97,0.98]:
    n,a,t=eval_c(p_eth,p_b15,qe,None); log(f"  ETH={qe:.2f}: n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\n[F] Both high-conf ETH=0.95 BTC=0.80:")
n,a,t=eval_c(p_eth,p_b15,0.95,0.80); log(f"  → n={n:,} ACC={a:.1f}% tpd={t:.1f}")

log(f"\nDONE")
