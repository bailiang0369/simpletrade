"""ETH h15: base vs base+fund vs base+fund+bs (+negw), 5-seed"""
import numpy as np, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def load_comb(splits_to_load):
    result={}
    for split in ['train','early_stop','test']:
        Xs=[]; y=None
        for xp, yp in splits_to_load:
            X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
            if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
            Xs.append(X)
        min_rows=min(X.shape[0] for X in Xs)
        X=np.hstack([X[:min_rows] for X in Xs])
        if y is not None: y=y[:min_rows]
        result[split]=(X,y)
        log(f"    {split}: {X.shape}")
        del Xs; gc.collect()
    return result

def train(Xtr,ytr,Xes,yes,Xte,yte,sw=None,label=""):
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
    log(f"  {label:40s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate,np.mean(pte,0)

log("[1] Build datasets...")
log("  base 56:")
b=load_comb([('','')])
log("  base+fund 68:")
bf=load_comb([('',''),('_fund',None)])
log("  base+fund+bs 93:")
bfb=load_comb([('',''),('_fund',None),('_bs',None)])

# negw (用 base ret, trim 到 min rows)
ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
n_base=bf['train'][0].shape[0]
ret_trim=ret_full[:n_base]
sw=np.where(np.abs(ret_trim)>=np.quantile(np.abs(ret_trim),0.90),0.3,1.0).astype(np.float32)
del ret_full; gc.collect()

log("\n[2] Train (all +negw)...")
Xtr_b,ytr_b=b['train']; Xes_b,yes_b=b['early_stop']; Xte_b,yte_b=b['test']
Xtr_bf,ytr_bf=bf['train']; Xes_bf,yes_bf=bf['early_stop']; Xte_bf,yte_bf=bf['test']
Xtr_bfb,ytr_bfb=bfb['train']; Xes_bfb,yes_bfb=bfb['early_stop']; Xte_bfb,yte_bfb=bfb['test']
del b,bf,bfb; gc.collect()

_,_,p_base = train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,sw,'ETH base 56 +negw')
_,_,p_bf   = train(Xtr_bf,ytr_bf,Xes_bf,yes_bf,Xte_bf,yte_bf,sw,'ETH base+fund 68 +negw')
_,_,p_bfb  = train(Xtr_bfb,ytr_bfb,Xes_bfb,yes_bfb,Xte_bfb,yte_bfb,sw,'ETH base+fund+bs 93 +negw')

# Also test without negw to confirm baseline
log("\n[2.5] Train without negw (reference)...")
_,_,p_base_now = train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,None,'ETH base 56 no-negw')

log("\n[3] Ensembles...")
yref = yte_b
log(f"  base te:          {roc_auc_score(yref, p_base):.4f}")
log(f"  base no-negw te:  {roc_auc_score(yref, p_base_now):.4f}")
log(f"  base+fund te:     {roc_auc_score(yref, p_bf):.4f}  vs base: {roc_auc_score(yref,p_bf)-roc_auc_score(yref,p_base):+.4f}")
log(f"  base+fund+bs te:  {roc_auc_score(yref, p_bfb):.4f}  vs base: {roc_auc_score(yref,p_bfb)-roc_auc_score(yref,p_base):+.4f}")
log(f"  base+fund ens:    {roc_auc_score(yref, (p_base+p_bf)/2):.4f}")
log(f"  all3 avg:         {roc_auc_score(yref, (p_base+p_bf+p_bfb)/3):.4f}")

best_single = max(roc_auc_score(yref,p) for p in [p_base,p_bf,p_bfb])
best_ens = max([roc_auc_score(yref,(p_base+p_bf)/2), roc_auc_score(yref,(p_base+p_bf+p_bfb)/3)])
log(f"\n[4] ETH FINAL")
log(f"  ETH base ref (prev): 0.5432")
log(f"  ETH base+negw:       {roc_auc_score(yref,p_base):.4f}")
log(f"  ETH best single:     {best_single:.4f}")
log(f"  ETH best ensemble:   {best_ens:.4f}")
log(f"  NEW vs prev ref:     {best_single-0.5432:+.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
