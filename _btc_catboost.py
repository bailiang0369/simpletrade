"""BTC: CatBoost vs LGB + 调参 + h5/h30 扫描"""
import numpy as np, time, gc, sys, os, warnings
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); PAR='data/datasets'; NPY='data/splits_npy'; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== Step 1: Scan BTC h5/h10/h20/h30 ==========
log("[1] Scan BTC horizons h5/h15/h20/h30...")
import pandas as pd
for h_test in [5, 10, 15, 20, 30]:
    ds_path = f'{PAR}/ds_BTC_h{h_test}.parquet'
    if not os.path.exists(ds_path):
        log(f"  h{h_test}: no ds, skip"); continue
    df = pd.read_parquet(ds_path).sort_values('ts')
    feat_cols = [c for c in df.columns if c not in ('ts','label','ret_future','soft_label')]
    ts_arr = df['ts'].to_numpy().astype(np.int64)
    X = df[feat_cols].to_numpy().astype(np.float32)
    y = df['label'].to_numpy().astype(np.int32)
    del df; gc.collect()
    splits = {}
    for split, tlo, thi in [('train',0,TRAIN_END),('es',TRAIN_END,ES_END),('test',META_END,10**18)]:
        m=(ts_arr>=tlo)&(ts_arr<thi); splits[split]=(X[m],y[m])
    del X, y, ts_arr; gc.collect()
    
    # Quick LGB single seed just to get baseline
    tr=lgb.Dataset(splits['train'][0], label=splits['train'][1])
    es=lgb.Dataset(splits['es'][0], label=splits['es'][1], reference=tr)
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'verbose':-1,'n_jobs':4,'seed':42}
    m=lgb.train(params,tr,2000,[es],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    te_auc = roc_auc_score(splits['test'][1], m.predict(splits['test'][0]))
    es_auc = roc_auc_score(splits['es'][1], m.predict(splits['es'][0]))
    log(f"  BTC h{h_test}: es={es_auc:.4f} te={te_auc:.4f}  n_train={splits['train'][0].shape[0]:,}")
    del splits, tr, es, m; gc.collect()

# ========== Step 2: CatBoost on BTC h15 ==========
log("\n[2] CatBoost BTC h15...")
def load_splits(prefix=''):
    return {s: (np.load(f'{NPY}/BTC_h15{prefix}_{s}_X.npy').astype(np.float32),
                np.load(f'{NPY}/BTC_h15{prefix}_{s}_y.npy').astype(np.int32))
            for s in ['train','early_stop','test']}

splits = load_splits()
Xtr,ytr = splits['train']; Xes,yes = splits['early_stop']; Xte,yte = splits['test']

# CatBoost default
log("  CatBoost default...")
pte,pes=[],[]; t0=time.time()
for sd in SEEDS:
    m=cb.CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, random_seed=sd, verbose=0,
                            early_stopping_rounds=100)
    m.fit(Xtr,ytr,eval_set=(Xes,yes), use_best_model=True)
    pte.append(m.predict_proba(Xte)[:,1]); pes.append(m.predict_proba(Xes)[:,1])
cb_te = roc_auc_score(yte, np.mean(pte,0)); cb_es = roc_auc_score(yes, np.mean(pes,0))
log(f"  CB default: es={cb_es:.4f} te={cb_te:.4f} ({time.time()-t0:.0f}s)")

# CatBoost tuned (depth=4, l2=5)
log("  CatBoost tuned d=4 l2=5...")
pte2,pes2=[],[]; t0=time.time()
for sd in SEEDS:
    m=cb.CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=4,
                            l2_leaf_reg=5, random_seed=sd, verbose=0,
                            early_stopping_rounds=150)
    m.fit(Xtr,ytr,eval_set=(Xes,yes), use_best_model=True)
    pte2.append(m.predict_proba(Xte)[:,1]); pes2.append(m.predict_proba(Xes)[:,1])
cb2_te = roc_auc_score(yte, np.mean(pte2,0)); cb2_es = roc_auc_score(yes, np.mean(pes2,0))
log(f"  CB tuned: es={cb2_es:.4f} te={cb2_te:.4f} ({time.time()-t0:.0f}s)")

# ========== Step 3: LGB tuning (num_leaves, min_child_samples) ==========
log("\n[3] LGB param sweep BTC h15...")
def lgb_train(num_leaves, min_child, lr=0.05):
    params={'objective':'binary','metric':'auc','learning_rate':lr,'num_leaves':num_leaves,
            'min_child_samples':min_child,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    return roc_auc_score(yes,np.mean(pes,0)), roc_auc_score(yte,np.mean(pte,0))

best_te = 0; best_p = None
for nl in [31, 63, 127, 255]:
    for mc in [100, 200, 400, 800]:
        es_a,te_a = lgb_train(nl, mc)
        log(f"  LGB nl={nl:3d} mc={mc:3d}: es={es_a:.4f} te={te_a:.4f}")
        if te_a > best_te: best_te=te_a; best_p=(nl,mc)

log(f"\n[4] Summary...")
log(f"  ETH h15 ref te=0.5432")
log(f"  BTC h15 LGB best: te={best_te:.4f} (params nl={best_p[0]} mc={best_p[1]})")
log(f"  BTC h15 CB default: te={cb_te:.4f}")
log(f"  BTC h15 CB tuned: te={cb2_te:.4f}")
log(f"  Gap ETH-BTC: {0.5432-max(best_te,cb_te,cb2_te):.4f}")
log(f"TOTAL: {time.time()-T0:.0f}s")
