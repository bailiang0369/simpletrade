"""快速基线对比: LightGBM vs MLP on ETH h15 数据集。
降采样训练数据避免 OOM。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np, pandas as pd
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

torch.manual_seed(42); np.random.seed(42)
print(f"torch={torch.__version__}", flush=True)

import config

# ============ 1. 加载数据集 ============
t0 = time.time()
print("Loading ETH h15 dataset...", flush=True)
df = pl.read_parquet(f"{config.DS_DIR}/ds_ETH_h15.parquet").sort("ts")
print(f"  shape: {df.shape}", flush=True)

TRAIN_END = 1722556800
ES_END = 1729641600
META_END = 1751241600

ts = df["ts"].to_numpy()
mask_tr = ts <= TRAIN_END
mask_es = (ts > TRAIN_END) & (ts <= ES_END)
mask_mv = (ts > ES_END) & (ts <= META_END)
mask_te = ts > META_END

print(f"  train={mask_tr.sum():,} es={mask_es.sum():,} mv={mask_mv.sum():,} te={mask_te.sum():,}", flush=True)

label = df["label"].to_numpy().astype(np.float32)
ret_future = df["ret_future"].to_numpy().astype(np.float32)
feat_cols = [c for c in df.columns if c not in ["ts", "label", "soft_label", "ret_future"]]
print(f"  n_features={len(feat_cols)}", flush=True)

# 降采样训练数据到 800K
tr_idx = np.where(mask_tr)[0]
np.random.seed(42)
tr_idx_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)
mask_tr_small = np.zeros(len(mask_tr), dtype=bool)
mask_tr_small[tr_idx_sub] = True

# 验证/测试集保持全量
X = df.select(feat_cols).to_numpy().astype(np.float32)
# 填充 NaN/Inf (只有35个, 不影响)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_tr, y_tr, r_tr = X[mask_tr_small], label[mask_tr_small], ret_future[mask_tr_small]
X_es, y_es = X[mask_es], label[mask_es]
X_mv, y_mv = X[mask_mv], label[mask_mv]
X_te, y_te, ts_te = X[mask_te], label[mask_te], ts[mask_te]

print(f"  After subsample: tr={len(X_tr):,} es={len(X_es):,} mv={len(X_mv):,} te={len(X_te):,}", flush=True)
gc.collect()

# ============ 2. LightGBM 基线 ============
print(f"\n{'='*60}\nLightGBM 5-seed rank ensemble\n{'='*60}", flush=True)

rw = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10)
hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, rw * 0.3, rw).astype(np.float32)

pvs_te = []; pvs_es = []; pvs_mv = []
for s in [42, 49, 56, 63, 70]:
    lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                      num_leaves=127, min_child_samples=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                      verbose=-1, seed=s, n_jobs=-1)
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=10000,
                    valid_sets=[es_ds], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_te.append(bst.predict(X_te))
    pvs_es.append(bst.predict(X_es))
    pvs_mv.append(bst.predict(X_mv))
    print(f"  seed={s} trees={bst.best_iteration} te_auc={roc_auc_score(y_te, pvs_te[-1]):.4f}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

pv_tree_te = rank_agg(pvs_te)
pv_tree_es = rank_agg(pvs_es)
pv_tree_mv = rank_agg(pvs_mv)
auc_tree_te = roc_auc_score(y_te, pv_tree_te)
auc_tree_es = roc_auc_score(y_es, pv_tree_es)
auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv)
print(f"\n  ★ Tree ES={auc_tree_es:.4f} MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)

# ============ 3. MLP 基线 ============
print(f"\n{'='*60}\nMLP 5-seed rank ensemble (baseline)\n{'='*60}", flush=True)

mu = X_tr.mean(axis=0)
sd = X_tr.std(axis=0) + 1e-8
X_tr_n = ((X_tr - mu) / sd).astype(np.float32)
X_es_n = ((X_es - mu) / sd).astype(np.float32)
X_mv_n = ((X_mv - mu) / sd).astype(np.float32)
X_te_n = ((X_te - mu) / sd).astype(np.float32)
del X_tr, X, label, ret_future; gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev = ft; layers = []
        for h in hs:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_mlp(m, Xtr, ytr, Xes, yes, ep=30, bs=512, pat=8, smooth=0.15, lr=5e-4, wd=0.06):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; best_ep = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr)); tl = 0; nb = 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        pv_es = evaluate(m, Xes); auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k: v.detach().clone() for k, v in m.state_dict().items()}; ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te_n); pv_es = evaluate(m, Xes); pv_mv = evaluate(m, X_mv_n)
    return pv_te, pv_es, pv_mv, roc_auc_score(y_te, pv_te)

FT = X_tr_n.shape[1]
pvs_te = []; pvs_es = []; pvs_mv = []
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT, [512, 256], 0.6)
    np_ = sum(p.numel() for p in m.parameters())
    pv_t, pv_e, pv_m, auc_t = train_mlp(m, X_tr_n, y_tr, X_es_n, y_es, ep=25, bs=512, pat=7)
    pvs_te.append(pv_t); pvs_es.append(pv_e); pvs_mv.append(pv_m)
    print(f"  seed={s} params={np_:,} te_auc={auc_t:.4f}", flush=True)

pv_mlp_te = rank_agg(pvs_te)
pv_mlp_es = rank_agg(pvs_es)
pv_mlp_mv = rank_agg(pvs_mv)
auc_mlp_te = roc_auc_score(y_te, pv_mlp_te)
auc_mlp_es = roc_auc_score(y_es, pv_mlp_es)
auc_mlp_mv = roc_auc_score(y_mv, pv_mlp_mv)
print(f"\n  ★ MLP ES={auc_mlp_es:.4f} MV={auc_mlp_mv:.4f} TE={auc_mlp_te:.4f}", flush=True)

# ============ 4. 对比 ============
print(f"\n{'='*60}\n基线对比\n{'='*60}", flush=True)
print(f"  Tree: ES={auc_tree_es:.4f} MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)
print(f"  MLP:  ES={auc_mlp_es:.4f} MV={auc_mlp_mv:.4f} TE={auc_mlp_te:.4f}", flush=True)
print(f"  Gap:  TE={auc_tree_te - auc_mlp_te:.4f}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
print(f"\n  Top-% Acc Comparison (test, {DAYS:.0f} days):", flush=True)
for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
    k = max(1, int(len(y_te) * pct / 100))
    acc_t = y_te[np.argsort(-pv_tree_te)[:k]].mean() * 100
    acc_m = y_te[np.argsort(-pv_mlp_te)[:k]].mean() * 100
    tpd = k / DAYS
    print(f"    top-{pct:.1f}%: Tree={acc_t:.1f}% MLP={acc_m:.1f}% diff={acc_t-acc_m:+.1f}% tpd≈{tpd:.1f}", flush=True)

from scipy.stats import spearmanr
corr, _ = spearmanr(pv_tree_te.astype(np.float64), pv_mlp_te.astype(np.float64))
print(f"\n  Rank correlation (TE): {corr:.4f}", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
