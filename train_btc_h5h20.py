"""训练 BTC h5 + h20 (LGB+CB, 带 funding), 与已有 h10/h15/h30/h60 对比"""
import numpy as np, gc, time, os
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
NPY = "/workspace/data/splits_npy"; OUT = "/workspace/data/ens_probs"
os.makedirs(OUT, exist_ok=True)
SEEDS=[42,49,56,63,70,77,84,91,98,105]
T0=time.time()
def now(): return f"{time.time()-T0:.0f}s"

def load(sym,h,s):
    return np.load(f"{NPY}/{sym}_h{h}_{s}_X.npy").astype(np.float32), \
           np.load(f"{NPY}/{sym}_h{h}_{s}_y.npy").astype(np.int32)
def rank_ens(A):
    R=np.zeros_like(A)
    for i in range(A.shape[0]): R[i]=np.argsort(np.argsort(A[i])).astype(np.float64)/(A.shape[1]-1)
    return R.mean(0)

def run_lgb(sym,h):
    tag = f"{sym}_h{h}"
    if os.path.exists(f"{OUT}/lgb_{tag}_te.npy"): print(f"  LGB {tag} 已存"); return
    Xtr,ytr=load(sym,h,"train"); Xes,yes=load(sym,h,"early_stop")
    print(f"[LGB {tag}] {Xtr.shape} {now()}",flush=True)
    import lightgbm as lgb
    Pmv,Pte=[],[]
    for s in SEEDS:
        p={"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,"min_data_in_leaf":200,
           "learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,"seed":s}
        tr=lgb.Dataset(Xtr,label=ytr); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(p,tr,3000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        Xmv,_=load(sym,h,"meta_val"); Xte,_=load(sym,h,"test")
        Pmv.append(m.predict(Xmv).astype(np.float64)); Pte.append(m.predict(Xte).astype(np.float64))
        del m,tr,es,Xmv,Xte; gc.collect()
    del Xtr,ytr,Xes,yes; gc.collect()
    np.save(f"{OUT}/lgb_{tag}_mv.npy",rank_ens(np.array(Pmv)))
    np.save(f"{OUT}/lgb_{tag}_te.npy",rank_ens(np.array(Pte)))
    ymv=np.load(f"{NPY}/{sym}_h{h}_meta_val_y.npy"); yte=np.load(f"{NPY}/{sym}_h{h}_test_y.npy")
    print(f"  LGB {tag} mv={roc_auc_score(ymv,np.load(f'{OUT}/lgb_{tag}_mv.npy')):.4f} "
          f"te={roc_auc_score(yte,np.load(f'{OUT}/lgb_{tag}_te.npy')):.4f} {now()}",flush=True)

def run_cb(sym,h):
    tag = f"{sym}_h{h}"
    if os.path.exists(f"{OUT}/cb_{tag}_te.npy"): print(f"  CB {tag} 已存"); return
    Xtr,ytr=load(sym,h,"train"); Xes,yes=load(sym,h,"early_stop")
    print(f"[CB {tag}] {Xtr.shape} {now()}",flush=True)
    from catboost import CatBoostClassifier
    Pmv,Pte=[],[]
    for s in SEEDS:
        cb=CatBoostClassifier(iterations=3000,learning_rate=0.05,depth=6,l2_leaf_reg=3,
                              random_seed=s,loss_function='Logloss',eval_metric='AUC',verbose=False,thread_count=4)
        cb.fit(Xtr,ytr,eval_set=[(Xes,yes)],early_stopping_rounds=100,verbose=False)
        Xmv,_=load(sym,h,"meta_val"); Xte,_=load(sym,h,"test")
        Pmv.append(cb.predict_proba(Xmv)[:,1].astype(np.float64))
        Pte.append(cb.predict_proba(Xte)[:,1].astype(np.float64))
        del cb,Xmv,Xte; gc.collect()
    del Xtr,ytr,Xes,yes; gc.collect()
    np.save(f"{OUT}/cb_{tag}_mv.npy",rank_ens(np.array(Pmv)))
    np.save(f"{OUT}/cb_{tag}_te.npy",rank_ens(np.array(Pte)))
    ymv=np.load(f"{NPY}/{sym}_h{h}_meta_val_y.npy"); yte=np.load(f"{NPY}/{sym}_h{h}_test_y.npy")
    print(f"  CB {tag} mv={roc_auc_score(ymv,np.load(f'{OUT}/cb_{tag}_mv.npy')):.4f} "
          f"te={roc_auc_score(yte,np.load(f'{OUT}/cb_{tag}_te.npy')):.4f} {now()}",flush=True)

# BTC h5 + h20 (新建, 带 funding)
for h in [5, 20]:
    run_lgb("BTC", h)
    run_cb("BTC", h)

print(f"\n✅ DONE {now()}")
