import numpy as np, gc, time
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
SPLITS = "/workspace/data/splits_npy"
SEEDS = [42,49,56,63,70,77,84,91,98,105]
T0 = time.time()
def now(): return f"{time.time()-T0:.0f}s"

def load(sym, h, split):
    X = np.load(f"{SPLITS}/{sym}_h{h}_{split}_X.npy").astype(np.float32)
    y = np.load(f"{SPLITS}/{sym}_h{h}_{split}_y.npy").astype(np.int32)
    return X, y

def rank_ens(A):
    R = np.zeros_like(A)
    for i in range(A.shape[0]):
        R[i] = np.argsort(np.argsort(A[i])).astype(np.float64) / (A.shape[1] - 1)
    return R.mean(0)

print("加载 ETH h15 splits...", flush=True)
Xtr, ytr = load("ETH",15,"train")
Xes, yes = load("ETH",15,"early_stop")
Xmv, ymv = load("ETH",15,"meta_val")
Xte, yte = load("ETH",15,"test")
print(f"  train={Xtr.shape} es={Xes.shape} mv={Xmv.shape} te={Xte.shape}", flush=True)

# LightGBM
print(f"\n[1] LightGBM {now()}", flush=True)
import lightgbm as lgb
lgb_Pmv, lgb_Pte = [], []
for si, s in enumerate(SEEDS):
    p={"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,"min_data_in_leaf":200,
       "learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"seed":s}
    tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(p,tr,3000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    lgb_Pmv.append(m.predict(Xmv).astype(np.float64))
    lgb_Pte.append(m.predict(Xte).astype(np.float64))
    del m,tr,es; gc.collect()
lgb_mv, lgb_te = rank_ens(np.array(lgb_Pmv)), rank_ens(np.array(lgb_Pte))
print(f"  LGB mv={roc_auc_score(ymv,lgb_mv):.4f} te={roc_auc_score(yte,lgb_te):.4f} {now()}", flush=True)
del lgb_Pmv,lgb_Pte; gc.collect()

# CatBoost
print(f"\n[2] CatBoost {now()}", flush=True)
from catboost import CatBoostClassifier
cb_Pmv, cb_Pte = [], []
for si, s in enumerate(SEEDS):
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            random_seed=s, loss_function='Logloss', eval_metric='AUC', verbose=False, thread_count=4)
    cb.fit(Xtr, ytr, eval_set=[(Xes, yes)], early_stopping_rounds=100, verbose=False)
    cb_Pmv.append(cb.predict_proba(Xmv)[:,1].astype(np.float64))
    cb_Pte.append(cb.predict_proba(Xte)[:,1].astype(np.float64))
    del cb; gc.collect()
cb_mv, cb_te = rank_ens(np.array(cb_Pmv)), rank_ens(np.array(cb_Pte))
print(f"  CB mv={roc_auc_score(ymv,cb_mv):.4f} te={roc_auc_score(yte,cb_te):.4f} {now()}", flush=True)
del cb_Pmv,cb_Pte; gc.collect()

# XGBoost  
print(f"\n[3] XGBoost {now()}", flush=True)
import xgboost as xgb
xgb_Pmv, xgb_Pte = [], []
for si, s in enumerate(SEEDS):
    p={"objective":"binary:logistic","eval_metric":"auc","max_depth":6,"learning_rate":0.05,
       "subsample":0.8,"colsample_bytree":0.8,"reg_lambda":3,"seed":s,"tree_method":"hist"}
    dtrain = xgb.DMatrix(Xtr,label=ytr); des = xgb.DMatrix(Xes,label=yes)
    m = xgb.train(p,dtrain,3000,[(des,'es')],early_stopping_rounds=100,verbose_eval=False)
    xgb_Pmv.append(m.predict(xgb.DMatrix(Xmv),iteration_range=(0,m.best_iteration)).astype(np.float64))
    xgb_Pte.append(m.predict(xgb.DMatrix(Xte),iteration_range=(0,m.best_iteration)).astype(np.float64))
    del m,dtrain,des; gc.collect()
xgb_mv, xgb_te = rank_ens(np.array(xgb_Pmv)), rank_ens(np.array(xgb_Pte))
print(f"  XGB mv={roc_auc_score(ymv,xgb_mv):.4f} te={roc_auc_score(yte,xgb_te):.4f} {now()}", flush=True)
del xgb_Pmv,xgb_Pte; gc.collect()

del Xtr,ytr,Xes,yes; gc.collect()

# Ensemble
print(f"\n[4] Ensemble 融合 {now()}", flush=True)
from itertools import product
best_w, best_auc = None, 0
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3 = round(1-w1-w2,2)
        if w3 < -1e-6: continue
        P = w1*lgb_mv + w2*cb_mv + w3*xgb_mv
        a = roc_auc_score(ymv,P)
        if a > best_auc: best_auc,best_w = a,(round(w1,1),round(w2,1),round(w3,1))
print(f"  best weights LGB={best_w[0]} CB={best_w[1]} XGB={best_w[2]} mv_auc={best_auc:.4f}")

ens_mv = best_w[0]*lgb_mv + best_w[1]*cb_mv + best_w[2]*xgb_mv
ens_te = best_w[0]*lgb_te + best_w[1]*cb_te + best_w[2]*xgb_te
eq_te = (lgb_te + cb_te + xgb_te) / 3
print(f"\n=== 最终结果 ===")
print(f"  LGB single    te AUC={roc_auc_score(yte,lgb_te):.4f}")
print(f"  CB  single    te AUC={roc_auc_score(yte,cb_te):.4f}")
print(f"  XGB single    te AUC={roc_auc_score(yte,xgb_te):.4f}")
print(f"  ENSEMBLE(best_w) te AUC={roc_auc_score(yte,ens_te):.4f}")
print(f"  EQUAL avg     te AUC={roc_auc_score(yte,eq_te):.4f}")
print(f"\n✅ DONE {now()}", flush=True)
