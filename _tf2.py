import numpy as np, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def run(prefix, sw_file=None, label=""):
    def ld(p): return np.load(f'{NPY}/BTC_h15{prefix}_{p}.npy')
    Xtr=ld('train_X').astype(np.float32); ytr=ld('train_y').astype(np.int32)
    Xes=ld('early_stop_X').astype(np.float32); yes=ld('early_stop_y').astype(np.int32)
    Xte=ld('test_X').astype(np.float32); yte=ld('test_y').astype(np.int32)
    sw = np.load(sw_file) if sw_file else None
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
    log(f"  {label:30s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return ate, np.mean(pte,0)

log("[1] Training 4 models...")
_,pte_base=run('','','base LGB 56')
_,pte_bnw =run('',f'{NPY}/BTC_h15_sw_base.npy','base+negw')
_,pte_fund=run('_fund','','fundonly 12')
_,pte_fnw =run('_fund',f'{NPY}/BTC_h15_sw_fund.npy','fundonly+negw')

yref=np.load(f'{NPY}/BTC_h15_test_y.npy').astype(np.int32)
log("\n[2] Ensembles...")
log(f"  base+fund:       {roc_auc_score(yref,(pte_base+pte_fund)/2):.4f}")
log(f"  base_nw+fund_nw: {roc_auc_score(yref,(pte_bnw+pte_fnw)/2):.4f}")
log(f"  all4 avg:        {roc_auc_score(yref,(pte_base+pte_bnw+pte_fund+pte_fnw)/4):.4f}")
log("\n[3] Analysis...")
log(f"  corr(base,fund)={np.corrcoef(pte_base,pte_fund)[0,1]:.3f}")
log(f"  corr(base_nw,fund_nw)={np.corrcoef(pte_bnw,pte_fnw)[0,1]:.3f}")
log(f"  funding-only te={roc_auc_score(yref,pte_fund):.4f}")
best=max([roc_auc_score(yref,(pte_base+pte_fund)/2),roc_auc_score(yref,(pte_bnw+pte_fnw)/2)])
log(f"\n  BEST ensemble={best:.4f}  ETH_ref=0.5432  gap={0.5432-best:.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
