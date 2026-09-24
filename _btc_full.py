"""BTC h15: base+fund+bs full features + negw"""
import numpy as np, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def load_combined(splits_to_load):
    """Load + hstack multiple prefixes per split, trim to min rows"""
    result = {}
    for split in ['train','early_stop','test']:
        Xs = []
        y = None
        for prefix, y_prefix in splits_to_load:
            X = np.load(f'{NPY}/BTC_h15{prefix}_{split}_X.npy').astype(np.float32)
            if y_prefix is not None:
                y = np.load(f'{NPY}/BTC_h15{y_prefix}_{split}_y.npy').astype(np.int32)
            Xs.append(X)
        # hstack with min rows
        min_rows = min(X.shape[0] for X in Xs)
        X_trimmed = np.hstack([X[:min_rows] for X in Xs])
        if y is not None:
            y = y[:min_rows]
        result[split] = (X_trimmed, y)
        log(f"  {split}: combined={X_trimmed.shape}")
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

log("[1] Build combined feature sets...")
# splits list: (X_prefix, y_prefix) — y_prefix is None for additional X blocks (use first set's y)
# Order: base (has y) + fund (no y, just X) + bs (no y, just X)

log("  [base 56] train+es+test:")
b = load_combined([('', '')])
Xtr_b, ytr_b = b['train']; Xes_b, yes_b = b['early_stop']; Xte_b, yte_b = b['test']
del b; gc.collect()

log("\n  [base+fund 68] train+es+test:")
bf = load_combined([('', ''), ('_fund', None)])
Xtr_bf, ytr_bf = bf['train']; Xes_bf, yes_bf = bf['early_stop']; Xte_bf, yte_bf = bf['test']
del bf; gc.collect()

log("\n  [base+fund+bs 93] train+es+test:")
bfb = load_combined([('', ''), ('_fund', None), ('_bs', None)])
Xtr_bfb, ytr_bfb = bfb['train']; Xes_bfb, yes_bfb = bfb['early_stop']; Xte_bfb, yte_bfb = bfb['test']
del bfb; gc.collect()

# negw weights (use base ret)
log("\n[2] Train all variants...")
ret_b = np.load(f'{NPY}/BTC_h15_train_ret.npy')
sw10 = np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)

_,_,p_base = train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,sw10,'base 56 +negw (ref)')
_,_,p_bf   = train(Xtr_bf,ytr_bf,Xes_bf,yes_bf,Xte_bf,yte_bf,sw10,'base+fund 68 +negw')
_,_,p_bfb  = train(Xtr_bfb,ytr_bfb,Xes_bfb,yes_bfb,Xte_bfb,yte_bfb,sw10,'base+fund+bs 93 +negw')

# Also test: just base+bs (no funding)
log("\n  [base+bs 81] train+es+test:")
bbs = load_combined([('', ''), ('_bs', None)])
Xtr_bbs, ytr_bbs = bbs['train']; Xes_bbs, yes_bbs = bbs['early_stop']; Xte_bbs, yte_bbs = bbs['test']
del bbs; gc.collect()
_,_,p_bbs = train(Xtr_bbs,ytr_bbs,Xes_bbs,yes_bbs,Xte_bbs,yte_bbs,sw10,'base+bs 81 +negw')

# Ensemble best
log("\n[3] Ensembles...")
yref = yte_b  # all aligned to base split
log(f"  base+fund_avg:       {roc_auc_score(yref, (p_base+p_bf)/2):.4f}")
log(f"  base+fund+bs_avg:    {roc_auc_score(yref, (p_base+p_bf+p_bfb)/3):.4f}")
log(f"  bf+bfb_avg:          {roc_auc_score(yref, (p_bf+p_bfb)/2):.4f}")
log(f"  all 4 avg:           {roc_auc_score(yref, (p_base+p_bf+p_bfb+p_bbs)/4):.4f}")

best_te = max(roc_auc_score(yref, p) for p in [p_base,p_bf,p_bfb,p_bbs])
best_ens = max([roc_auc_score(yref,(p_base+p_bf)/2), roc_auc_score(yref,(p_base+p_bf+p_bfb)/3), 
                roc_auc_score(yref,(p_bf+p_bfb)/2), roc_auc_score(yref,(p_base+p_bf+p_bfb+p_bbs)/4)])
log(f"\n[4] FINAL SUMMARY")
log(f"  ETH h15 ref:      0.5432")
log(f"  BTC best single:  {best_te:.4f}  gap={0.5432-best_te:.4f}")
log(f"  BTC best ens:     {best_ens:.4f}  gap={0.5432-best_ens:.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
