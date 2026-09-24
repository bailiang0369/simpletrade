"""ETH h15: CatBoost (比 LGB 不同的树模型, ensemble 更有效)
只用 base+fund 68 避免 OOM
"""
import numpy as np, time, gc, sys, warnings
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# Load base+fund 68
def load_c(prefixes, split):
    Xs=[]; y=None
    for xp, yp in prefixes:
        X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
        if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
        Xs.append(X)
    mr=min(X.shape[0] for X in Xs)
    return np.hstack([X[:mr] for X in Xs]), y[:mr] if y is not None else None

Xtr,ytr=load_c([('',''),('_fund',None)],'train')
Xes,yes=load_c([('',''),('_fund',None)],'early_stop')
Xte,yte=load_c([('',''),('_fund',None)],'test')
log(f"[1] ETH h15 base+fund 68 loaded: tr={Xtr.shape} te={Xte.shape}")

# negw
ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_full[:Xtr.shape[0]])>=np.quantile(np.abs(ret_full[:Xtr.shape[0]]),0.90),0.3,1.0).astype(np.float32)
del ret_full; gc.collect()

# ========== LGB reference ==========
log("\n[2] LGB reference (5-seed, for correlation)...")
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_lgb=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_lgb.append(m.predict(Xte))
p_lgb_mean = np.mean(p_lgb,axis=0)
log(f"  LGB base+fund+negw: te={roc_auc_score(yte, p_lgb_mean):.4f}")

# ========== CatBoost ==========
log("\n[3] CatBoost (5-seed)...")
for variant, kwargs in [
    ('CB default', dict(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3)),
    ('CB small',   dict(iterations=3000, learning_rate=0.03, depth=4, l2_leaf_reg=5)),
    ('CB deep',    dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=1)),
]:
    pte=[]
    t0=time.time()
    for sd in SEEDS:
        m=cb.CatBoostClassifier(**kwargs, random_seed=sd, verbose=0,
                                early_stopping_rounds=100)
        m.fit(Xtr, ytr, eval_set=(Xes, yes), use_best_model=True, sample_weight=sw)
        pte.append(m.predict_proba(Xte)[:,1])
    p_mean=np.mean(pte,axis=0)
    cb_auc = roc_auc_score(yte, p_mean)
    corr_lgb = np.corrcoef(p_mean, p_lgb_mean)[0,1]
    log(f"  {variant:12s}: te={cb_auc:.4f}  corr_LGB={corr_lgb:.3f}  ({time.time()-t0:.0f}s)")
    
    # Ensemble with LGB
    ens = (p_mean + p_lgb_mean) / 2
    log(f"    +LGB ensemble: {roc_auc_score(yte, ens):.4f}  (gain={roc_auc_score(yte,ens)-roc_auc_score(yte,p_lgb_mean):+.4f})")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
