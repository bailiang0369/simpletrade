import numpy as np, lightgbm as lgb, gc
from sklearn.metrics import roc_auc_score
SEEDS=[42,49,56,63,70,77,84,91,98,105]
SPLITS = "/workspace/data/splits_npy"

def run(name, sym, h, params={}):
    Xtr = np.load(f"{SPLITS}/{sym}_h{h}_train_X.npy").astype(np.float32)
    ytr = np.load(f"{SPLITS}/{sym}_h{h}_train_y.npy").astype(np.int32)
    Xes = np.load(f"{SPLITS}/{sym}_h{h}_early_stop_X.npy").astype(np.float32)
    yes = np.load(f"{SPLITS}/{sym}_h{h}_early_stop_y.npy").astype(np.int32)
    print(f"  train {Xtr.shape} ({Xtr.nbytes/1e6:.0f}MB) es {Xes.shape}", flush=True)
    base_p={"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,
            "min_data_in_leaf":200,"learning_rate":0.05,"feature_fraction":0.8,
            "bagging_fraction":0.8}
    tr = lgb.Dataset(Xtr, label=ytr); del Xtr, ytr; gc.collect()
    es = lgb.Dataset(Xes, label=yes, reference=tr); del Xes, yes; gc.collect()
    Pmv, Pte = [], []
    for si, s in enumerate(SEEDS):
        p = {**base_p, **params, "seed": s}
        print(f"  seed {si+1}/{len(SEEDS)}...", flush=True)
        m = lgb.train(p, tr, 3000, [es], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split, Pout in [("meta_val", Pmv), ("test", Pte)]:
            Xv = np.load(f"{SPLITS}/{sym}_h{h}_{split}_X.npy").astype(np.float32)
            Pout.append(m.predict(Xv, num_iteration=m.best_iteration).astype(np.float64))
            del Xv; gc.collect()
        del m; gc.collect()
    def rank_ens(a):
        P = np.stack(a, 0); R = np.zeros_like(P)
        for i in range(P.shape[0]): R[i] = np.argsort(np.argsort(P[i])).astype(np.float64)/(P.shape[1]-1)
        return R.mean(0)
    Pmv, Pte = rank_ens(Pmv), rank_ens(Pte)
    for split, P in [("mv", Pmv), ("te", Pte)]:
        real = "meta_val" if split == "mv" else "test"
        lb = np.load(f"{SPLITS}/{sym}_h{h}_{real}_y.npy").astype(int)
        auc = roc_auc_score(lb, P)
        print(f"[{sym} h{h} {name}] {split} AUC={auc:.4f}", flush=True)

# ETH h15 新特征
print("=== ETH h15 (75 feats: 56 base + 19 BTC cross-asset + funding) ===", flush=True)
run("baseline_newfeats", "ETH", 15)
run("nL255_newfeats", "ETH", 15, {"num_leaves": 255})
print("\n=== BTC h15 baseline ===", flush=True)
run("baseline", "BTC", 15)
print("\n✅ ALL DONE", flush=True)
