import numpy as np, gc, time, os
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
SPLITS = "/workspace/data/splits_npy"; OUT="/workspace/data/ens56_probs"
os.makedirs(OUT, exist_ok=True)
SEEDS=[42,49,56,63,70,77,84,91,98,105]
BASE_COLS = 56  # ETH base feats (前 56 列, 最佳 baseline)
T0=time.time()
def now(): return f"{time.time()-T0:.0f}s"

def load56(s):
    X=np.load(f"{SPLITS}/ETH_h15_{s}_X.npy").astype(np.float32)[:,:BASE_COLS]
    y=np.load(f"{SPLITS}/ETH_h15_{s}_y.npy").astype(np.int32)
    return X,y
def rank_ens(A):
    R=np.zeros_like(A)
    for i in range(A.shape[0]):
        R[i]=np.argsort(np.argsort(A[i])).astype(np.float64)/(A.shape[1]-1)
    return R.mean(0)

Xtr,ytr=load56("train"); Xes,yes=load56("early_stop")
Xmv,ymv=load56("meta_val"); Xte,yte=load56("test")
print(f"train={Xtr.shape} (前{BASE_COLS} ETH base feats) {now()}",flush=True)

# LGB
if os.path.exists(f"{OUT}/lgb_te.npy"):
    print("加载已存 LGB"); lgb_mv=np.load(f"{OUT}/lgb_mv.npy"); lgb_te=np.load(f"{OUT}/lgb_te.npy")
else:
    import lightgbm as lgb
    print(f"\n[1] LightGBM 56feats {now()}",flush=True)
    lPmv,lPte=[],[]
    for s in SEEDS:
        p={"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,"min_data_in_leaf":200,
           "learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"seed":s}
        tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(p,tr,3000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        lPmv.append(m.predict(Xmv).astype(np.float64))
        lPte.append(m.predict(Xte).astype(np.float64))
        del m,tr,es; gc.collect()
    lgb_mv=rank_ens(np.array(lPmv)); lgb_te=rank_ens(np.array(lPte))
    np.save(f"{OUT}/lgb_mv.npy",lgb_mv); np.save(f"{OUT}/lgb_te.npy",lgb_te)
    print(f"  LGB mv={roc_auc_score(ymv,lgb_mv):.4f} te={roc_auc_score(yte,lgb_te):.4f} {now()}",flush=True)

# CB
if os.path.exists(f"{OUT}/cb_te.npy"):
    print("加载已存 CB"); cb_mv=np.load(f"{OUT}/cb_mv.npy"); cb_te=np.load(f"{OUT}/cb_te.npy")
else:
    from catboost import CatBoostClassifier
    print(f"\n[2] CatBoost 56feats {now()}",flush=True)
    cPmv,cPte=[],[]
    for s in SEEDS:
        cb=CatBoostClassifier(iterations=3000,learning_rate=0.05,depth=6,l2_leaf_reg=3,
                              random_seed=s,loss_function='Logloss',eval_metric='AUC',verbose=False,thread_count=4)
        cb.fit(Xtr,ytr,eval_set=[(Xes,yes)],early_stopping_rounds=100,verbose=False)
        cPmv.append(cb.predict_proba(Xmv)[:,1].astype(np.float64))
        cPte.append(cb.predict_proba(Xte)[:,1].astype(np.float64))
        del cb; gc.collect()
    cb_mv=rank_ens(np.array(cPmv)); cb_te=rank_ens(np.array(cPte))
    np.save(f"{OUT}/cb_mv.npy",cb_mv); np.save(f"{OUT}/cb_te.npy",cb_te)
    print(f"  CB mv={roc_auc_score(ymv,cb_mv):.4f} te={roc_auc_score(yte,cb_te):.4f} {now()}",flush=True)
del Xtr,ytr,Xes,yes; gc.collect()

# 融合
print(f"\n[3] Ensemble 融合 {now()}",flush=True)
best_w,best_a=None,0
for w1 in np.arange(0,1.01,0.05):
    P=w1*lgb_mv+(1-w1)*cb_mv
    a=roc_auc_score(ymv,P)
    if a>best_a: best_a,best_w=a,w1
print(f"  best LGB={round(best_w,2)} CB={round(1-best_w,2)} mv={best_a:.4f}")
ens_te=best_w*lgb_te+(1-best_w)*cb_te
eq_te=(lgb_te+cb_te)/2
print(f"\n=== 56 ETH base feats te AUC ===")
print(f"  LGB single    te={roc_auc_score(yte,lgb_te):.4f}")
print(f"  CB  single    te={roc_auc_score(yte,cb_te):.4f}")
print(f"  ENSEMBLE      te={roc_auc_score(yte,ens_te):.4f}")
print(f"  EQUAL avg     te={roc_auc_score(yte,eq_te):.4f}")
print(f"\n✅ DONE {now()}",flush=True)
