import numpy as np, lightgbm as lgb, gc, os
from sklearn.metrics import roc_auc_score
SEEDS=[42,49,56,63,70,77,84,91,98,105]
SPLITS = "/workspace/data/splits_npy"

def load(sym, h, split):
    return np.load(f"{SPLITS}/{sym}_h{h}_{split}_X.npy", mmap_mode='r').astype(np.float32), \
           np.load(f"{SPLITS}/{sym}_h{h}_{split}_y.npy", mmap_mode='r').astype(np.int32)

def run(name, sym, h, params={}):
    Xtr, ytr = load(sym, h, "train")
    Xes, yes = load(sym, h, "early_stop")
    print(f"  train {Xtr.shape} es {Xes.shape}", flush=True)
    base_p={"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,
            "min_data_in_leaf":200,"learning_rate":0.05,"feature_fraction":0.8,
            "bagging_fraction":0.8}
    # copy_data=False: LightGBM 不复制数据, 省内存
    tr = lgb.Dataset(Xtr, label=ytr, params={"copy_data": False})
    es = lgb.Dataset(Xes, label=yes, reference=tr, params={"copy_data": False})
    Pmv, Pte = [], []
    for si, s in enumerate(SEEDS):
        p = {**base_p, **params, "seed": s}
        print(f"  seed {si+1}/{len(SEEDS)}...", flush=True)
        m = lgb.train(p, tr, 3000, [es], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split, Pout in [("meta_val", Pmv), ("test", Pte)]:
            Xv, _ = load(sym, h, split)
            Pout.append(m.predict(Xv, num_iteration=m.best_iteration).astype(np.float64))
            del Xv; gc.collect()
        del m; gc.collect()
    def rank_ens(a):
        P = np.stack(a, 0); R = np.zeros_like(P)
        for i in range(P.shape[0]):
            R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
        return R.mean(0)
    Pmv, Pte = rank_ens(Pmv), rank_ens(Pte)
    for split, P in [("mv", Pmv), ("te", Pte)]:
        real = "meta_val" if split == "mv" else "test"
        _, lb = load(sym, h, real)
        auc = roc_auc_score(lb, P)
        print(f"[{sym} h{h} {name}] {split} AUC={auc:.4f}", flush=True)
    del Xtr, ytr, Xes, yes, tr, es; gc.collect()

print("=== ETH h15 baseline NEW (75 feats incl BTC cross-asset + funding) ===", flush=True)
run("baseline_newfeats", "ETH", 15)
run("nL255_newfeats", "ETH", 15, {"num_leaves": 255})
print("\n=== BTC h15 baseline ===", flush=True)
run("baseline", "BTC", 15)
print("\n✅ ALL DONE", flush=True)
