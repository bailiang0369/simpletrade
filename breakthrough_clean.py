"""ETH h15 单模型突破 — 简洁版。每个 params 独立构建 Dataset。"""
import sys; sys.path.insert(0,'/workspace')
import os, gc, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
CTX = AssetContext("ETH", horizon=15)
T0 = time.time()
def now(): return f"{(time.time()-T0)/60:.1f}m"

def rank_ens(arrays):
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)

def nolookahead(q, conf, pred, y, ts, win=30, cold=30):
    day_of = ts // 86400
    days = np.unique(day_of)
    sel = np.zeros(len(y), dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di-win):di]
        if len(prior) < cold: continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0: continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True
    acc = float((pred[sel]==y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(days), 1)
    return acc*100, tpd, int(sel.sum())

def train_ens(name, params, Xtr, ytr, Xes, yes, wtr=None):
    root = f"{config.MODEL_DIR}/bt_{name}"
    os.makedirs(root, exist_ok=True)
    mm = CTX.split_rows["meta_val"]; mt = CTX.split_rows["test"]
    Pmv, Pte = [], []
    for s in SEEDS:
        p = {**params, "seed": s}
        tr = lgb.Dataset(Xtr, label=ytr, weight=wtr) if wtr is not None else lgb.Dataset(Xtr, label=ytr)
        es = lgb.Dataset(Xes, label=yes, reference=tr)
        m = lgb.train(p, tr, num_boost_round=3000, valid_sets=[es],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        Xmv = CTX.Xall[mm]; Xte = CTX.Xall[mt]
        pmv = m.predict(Xmv).astype(np.float64); pte = m.predict(Xte).astype(np.float64)
        np.save(f"{root}/s{s}_mv.npy", pmv.astype(np.float32))
        np.save(f"{root}/s{s}_te.npy", pte.astype(np.float32))
        Pmv.append(pmv); Pte.append(pte)
        gc.collect()
    return rank_ens(Pmv), rank_ens(Pte)

def show_curve(name, Pmv, Pte):
    for split, P in [("mv", Pmv), ("te", Pte)]:
        mask = CTX.split_rows["meta_val"] if split=="mv" else CTX.split_rows["test"]
        y = CTX.label[mask]
        te = np.asarray(CTX.times("meta_val" if split=="mv" else "test")).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2
        pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"  [{name}] {split} AUC={auc:.4f}")
        for q in [98.5, 99.0, 99.2, 99.4, 99.5]:
            a, t, n = nolookahead(q, conf, pred, y, te)
            star = " ★★★" if a >= 65 and t >= 12 else (" ★★" if a >= 64 and t >= 12 else "")
            print(f"      q={q} → {a:.2f}% @ {t:.1f}t ({n}){star}")

# ===== 准备数据 =====
tr_mask = CTX.split_rows["train"]; es_mask = CTX.split_rows["early_stop"]
Xtr = CTX.Xall[tr_mask]; Xes = CTX.Xall[es_mask]
ytr_b = CTX.label[tr_mask].astype(np.float64); yes_b = CTX.label[es_mask].astype(np.float64)
ret_tr = CTX.retf("train")
wtr_ret = np.clip(np.abs(ret_tr) * 50, 0.5, 5.0)

print("=" * 70)
print(f"ETH h15 单模型突破")
print(f"baseline test q=99: 62.01% @ 13.7t (AUC=0.5424)")
print("=" * 70)

# 方向 1: num_leaves
print(f"\n[{now()}] 1: num_leaves 对比 (mD=200)")
for nL in [63, 127, 255]:
    p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": nL,
         "min_data_in_leaf": 200, "learning_rate": 0.05,
         "feature_fraction": 0.8, "bagging_fraction": 0.8}
    Pmv, Pte = train_ens(f"nL{nL}", p, Xtr, ytr_b, Xes, yes_b, wtr_ret)
    show_curve(f"nL{nL}", Pmv, Pte)
    gc.collect()

# 方向 2: nL=127 下 mD
print(f"\n[{now()}] 2: mD 对比 (nL=127)")
for mD in [50, 200, 500]:
    p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 127,
         "min_data_in_leaf": mD, "learning_rate": 0.05,
         "feature_fraction": 0.8, "bagging_fraction": 0.8}
    Pmv, Pte = train_ens(f"nL127_mD{mD}", p, Xtr, ytr_b, Xes, yes_b, wtr_ret)
    show_curve(f"nL127_mD{mD}", Pmv, Pte)
    gc.collect()

# 方向 3: 去噪
print(f"\n[{now()}] 3: 去噪 ε (nL=127 mD=200)")
for eps in [0.0, 0.0005, 0.001]:
    if eps == 0:
        Xd, yd, wd = Xtr, ytr_b, wtr_ret
    else:
        m = np.abs(ret_tr) > eps
        Xd = Xtr[m]; yd = ytr_b[m]; wd = wtr_ret[m]
    print(f"  ε={eps}: {len(Xd)}/{len(Xtr)}")
    p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 127,
         "min_data_in_leaf": 200, "learning_rate": 0.05,
         "feature_fraction": 0.8, "bagging_fraction": 0.8}
    Pmv, Pte = train_ens(f"nL127_eps{eps}", p, Xd, yd, Xes, yes_b, wd)
    show_curve(f"nL127_eps{eps}", Pmv, Pte)
    gc.collect()

# 方向 4: regression (predict ret, AUC early_stop)
print(f"\n[{now()}] 4: regression (predict ret, AUC early_stop)")
Pmv_r, Pte_r = [], []
for s in SEEDS:
    tr = lgb.Dataset(Xtr, label=ret_tr, weight=wtr_ret)
    es = lgb.Dataset(Xes, label=yes_b, reference=tr)
    def auc_eval(yhat, _): return ("AUC", roc_auc_score(yes_b, yhat), True)
    m = lgb.train({"objective": "regression", "metric": "l2", "verbose": -1, "seed": s,
                   "num_leaves": 127, "min_data_in_leaf": 200, "learning_rate": 0.05},
                  tr, num_boost_round=3000, valid_sets=[es], feval=auc_eval,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    mm = CTX.split_rows["meta_val"]; mt = CTX.split_rows["test"]
    Pmv_r.append(m.predict(CTX.Xall[mm]).astype(np.float64))
    Pte_r.append(m.predict(CTX.Xall[mt]).astype(np.float64))
    gc.collect()
Pmv_r = rank_ens(Pmv_r); Pte_r = rank_ens(Pte_r)
show_curve("reg", Pmv_r, Pte_r)
del Pmv_r, Pte_r; gc.collect()

# 方向 5: rank:pairwise
print(f"\n[{now()}] 5: rank:pairwise")
Pmv_k, Pte_k = [], []
for s in SEEDS:
    tr = lgb.Dataset(Xtr, label=ytr_b.astype(int))
    es = lgb.Dataset(Xes, label=yes_b.astype(int), reference=tr)
    m = lgb.train({"objective": "rank:pairwise", "metric": "auc", "verbose": -1, "seed": s,
                   "num_leaves": 127, "min_data_in_leaf": 200, "learning_rate": 0.05},
                  tr, num_boost_round=3000, valid_sets=[es],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    mm = CTX.split_rows["meta_val"]; mt = CTX.split_rows["test"]
    Pmv_k.append(m.predict(CTX.Xall[mm]).astype(np.float64))
    Pte_k.append(m.predict(CTX.Xall[mt]).astype(np.float64))
    gc.collect()
Pmv_k = rank_ens(Pmv_k); Pte_k = rank_ens(Pte_k)
# rank 模型输出是分数, 不分概率
for split, P in [("mv", Pmv_k), ("te", Pte_k)]:
    mask = CTX.split_rows["meta_val"] if split=="mv" else CTX.split_rows["test"]
    y = CTX.label[mask]
    te = np.asarray(CTX.times("meta_val" if split=="mv" else "test")).astype("datetime64[s]").astype(np.int64)
    conf = np.abs(P); pred = (P >= 0).astype(np.int8)
    auc = roc_auc_score(y, P)
    print(f"  [rank] {split} AUC={auc:.4f}")
    for q in [99.0, 99.5]:
        a, t, n = nolookahead(q, conf, pred, y, te)
        print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")
del Pmv_k, Pte_k; gc.collect()

# 方向 6: CatBoost
print(f"\n[{now()}] 6: CatBoost")
try:
    from catboost import CatBoostClassifier
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "--quiet"])
    from catboost import CatBoostClassifier

import warnings; warnings.filterwarnings("ignore")
Pmv_cb, Pte_cb = [], []
for s in SEEDS:
    cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, loss_function="Logloss", eval_metric="AUC",
                            random_seed=s, verbose=False)
    cb.fit(Xtr, ytr_b.astype(int), eval_set=(Xes, yes_b.astype(int)), use_best_model=True)
    mm = CTX.split_rows["meta_val"]; mt = CTX.split_rows["test"]
    Pmv_cb.append(cb.predict_proba(CTX.Xall[mm])[:, 1].astype(np.float64))
    Pte_cb.append(cb.predict_proba(CTX.Xall[mt])[:, 1].astype(np.float64))
    gc.collect()
Pmv_cb = rank_ens(Pmv_cb); Pte_cb = rank_ens(Pte_cb)
show_curve("CB_d6", Pmv_cb, Pte_cb)
del Pmv_cb, Pte_cb; gc.collect()

print(f"\n[{now()}] ✅ 全部完成")
