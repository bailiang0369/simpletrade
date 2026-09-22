"""复现单模型突破: 只跑关键方向验证 (nL255 + 去噪 ε=0.0005), 确认 65% 可复现。"""
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
        print(f"  [{name}] {split} AUC={auc:.4f}", flush=True)
        for q in [98.5, 99.0, 99.2, 99.4, 99.5, 99.6]:
            a, t, n = nolookahead(q, conf, pred, y, te)
            star = " ★★★" if a >= 65 and t >= 12 else (" ★★" if a >= 64 and t >= 12 else "")
            print(f"      q={q} → {a:.2f}% @ {t:.1f}t ({n}){star}", flush=True)

# ===== 数据 =====
tr_mask = CTX.split_rows["train"]; es_mask = CTX.split_rows["early_stop"]
Xtr = CTX.Xall[tr_mask]; Xes = CTX.Xall[es_mask]
ytr_b = CTX.label[tr_mask].astype(np.float64); yes_b = CTX.label[es_mask].astype(np.float64)
ret_tr = CTX.retf("train")
wtr_ret = np.clip(np.abs(ret_tr) * 50, 0.5, 5.0)

print("=" * 70, flush=True)
print("复现单模型突破 (环境重置后)", flush=True)
print("=" * 70, flush=True)

# A: baseline nL=63 (验证可复现)
print(f"\n[{now()}] A: baseline nL=63", flush=True)
p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 63,
     "min_data_in_leaf": 200, "learning_rate": 0.05,
     "feature_fraction": 0.8, "bagging_fraction": 0.8}
Pmv, Pte = train_ens("repro_nL63", p, Xtr, ytr_b, Xes, yes_b, wtr_ret)
show_curve("nL63", Pmv, Pte); gc.collect()

# B: nL=255
print(f"\n[{now()}] B: nL=255", flush=True)
p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 255,
     "min_data_in_leaf": 200, "learning_rate": 0.05,
     "feature_fraction": 0.8, "bagging_fraction": 0.8}
Pmv, Pte = train_ens("repro_nL255", p, Xtr, ytr_b, Xes, yes_b, wtr_ret)
show_curve("nL255", Pmv, Pte); gc.collect()

# C: 去噪 ε=0.0005 (关键!)
print(f"\n[{now()}] C: 去噪 ε=0.0005 (nL=127)", flush=True)
m = np.abs(ret_tr) > 0.0005
Xd, yd, wd = Xtr[m], ytr_b[m], wtr_ret[m]
print(f"  ε=0.0005: {len(Xd)}/{len(Xtr)}", flush=True)
p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 127,
     "min_data_in_leaf": 200, "learning_rate": 0.05,
     "feature_fraction": 0.8, "bagging_fraction": 0.8}
Pmv, Pte = train_ens("repro_eps5", p, Xd, yd, Xes, yes_b, wd)
show_curve("eps0.0005", Pmv, Pte); gc.collect()

# D: 去噪 ε=0.0005 + nL=255 (叠加)
print(f"\n[{now()}] D: 去噪 ε=0.0005 + nL=255", flush=True)
p = {"objective": "binary", "metric": "auc", "verbose": -1, "num_leaves": 255,
     "min_data_in_leaf": 200, "learning_rate": 0.05,
     "feature_fraction": 0.8, "bagging_fraction": 0.8}
Pmv, Pte = train_ens("repro_eps5_nL255", p, Xd, yd, Xes, yes_b, wd)
show_curve("eps0.0005_nL255", Pmv, Pte); gc.collect()

print(f"\n[{now()}] ✅ 复现完成", flush=True)
