"""ETH: 分3步跑避免 OOM"""
import numpy as np, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def load_comb(prefixes):
    result={}
    for split in ['train','early_stop','test']:
        Xs=[]; y=None
        for xp, yp in prefixes:
            X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
            if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
            Xs.append(X)
        mr=min(X.shape[0] for X in Xs)
        X=np.hstack([X[:mr] for X in Xs])
        if y is not None: y=y[:mr]
        result[split]=(X,y); del Xs; gc.collect()
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
    log(f"  {label:35s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate,np.mean(pte,0)

# === Step 1: base 56 (已经知道了, 但跑一次) ===
log("[1] base 56...")
b=load_comb([('','')])
Xtr,ytr=b['train']; Xes,yes=b['early_stop']; Xte,yte=b['test']
del b; gc.collect()
ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw10=np.where(np.abs(ret_full)>=np.quantile(np.abs(ret_full),0.90),0.3,1.0).astype(np.float32)
del ret_full; gc.collect()
_,_,p_base=train(Xtr,ytr,Xes,yes,Xte,yte,sw10,'base 56 +negw')

# === Step 2: base+fund 68 ===
log("\n[2] base+fund 68...")
bf=load_comb([('',''),('_fund',None)])
Xtr2,ytr2=bf['train']; Xes2,yes2=bf['early_stop']; Xte2,yte2=bf['test']
del bf; gc.collect()
sw2 = sw10[:Xtr2.shape[0]]
_,_,p_bf=train(Xtr2,ytr2,Xes2,yes2,Xte2,yte2,sw2,'base+fund 68 +negw')
del Xtr2,ytr2,Xes2,yes2,Xte2,yte2; gc.collect()

# === Step 3: base+bs 81 ===
log("\n[3] base+bs 81...")
bbs=load_comb([('',''),('_bs',None)])
Xtr3,ytr3=bbs['train']; Xes3,yes3=bbs['early_stop']; Xte3,yte3=bbs['test']
del bbs; gc.collect()
sw3 = sw10[:Xtr3.shape[0]]
_,_,p_bbs=train(Xtr3,ytr3,Xes3,yes3,Xte3,yte3,sw3,'base+bs 81 +negw')
del Xtr3,ytr3,Xes3,yes3,Xte3,yte3; gc.collect()

# === Step 4: base+fund+bs 93 (分块) ===
log("\n[4] base+fund+bs 93...")
bfb=load_comb([('',''),('_fund',None),('_bs',None)])
Xtr4,ytr4=bfb['train']; Xes4,yes4=bfb['early_stop']; Xte4,yte4=bfb['test']
del bfb; gc.collect()
sw4 = sw10[:Xtr4.shape[0]]
_,_,p_bfb=train(Xtr4,ytr4,Xes4,yes4,Xte4,yte4,sw4,'base+fund+bs 93 +negw')

# === Results ===
log("\n" + "="*60)
log("ETH h15 FINAL SUMMARY (5-seed, negw)")
log("="*60)
log(f"  base 56:          te={roc_auc_score(yte,p_base):.4f}")
log(f"  base+fund 68:     te={roc_auc_score(yte,p_bf):.4f}  (vs base: {roc_auc_score(yte,p_bf)-roc_auc_score(yte,p_base):+.4f})")
log(f"  base+bs 81:       te={roc_auc_score(yte,p_bbs):.4f}  (vs base: {roc_auc_score(yte,p_bbs)-roc_auc_score(yte,p_base):+.4f})")
log(f"  base+fund+bs 93:  te={roc_auc_score(yte,p_bfb):.4f}  (vs base: {roc_auc_score(yte,p_bfb)-roc_auc_score(yte,p_base):+.4f})")
log(f"\nEnsembles:")
log(f"  base+fund avg:    {roc_auc_score(yte, (p_base+p_bf)/2):.4f}")
log(f"  base+fund+bs avg: {roc_auc_score(yte, (p_base+p_bf+p_bbs)/3):.4f}")
log(f"  all4 avg:         {roc_auc_score(yte, (p_base+p_bf+p_bbs+p_bfb)/4):.4f}")

single_best = max(roc_auc_score(yte,p) for p in [p_base,p_bf,p_bbs,p_bfb])
ens_best = max([roc_auc_score(yte,(p_base+p_bf)/2), roc_auc_score(yte,(p_base+p_bf+p_bbs)/3), 
                roc_auc_score(yte,(p_base+p_bf+p_bbs+p_bfb)/4), roc_auc_score(yte,(p_bf+p_bfb)/2)])
log(f"\n  BEST SINGLE:  {single_best:.4f}  (+{single_best-0.5428:+.4f} vs base ref)")
log(f"  BEST ENS:     {ens_best:.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
