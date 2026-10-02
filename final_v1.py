"""FINAL V1: ETH Tree-ENS + ETH NN-ENS + BTC NN-ENS → Stack → Rolling Quantile Eval.

核心思路:
  1. ETH Tree: LGBM 5-seed + CatBoost 5-seed (全量训练 2.4M)
  2. ETH NN:  ResMLP [256,128] + top20 interactions (全量训练 2.4M)
  3. BTC NN:  BTC 特征对齐 ETH 时间戳 → 预测 ETH H15 label
  4. Stack:   三者 rank 融合, meta_val 扫最优权重
  5. Eval:    rolling 30-day quantile threshold, 严格无前视
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, datetime as dtm
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import config
import features as fe

t0_all = time.time()
torch.manual_seed(42); np.random.seed(42)

# ================================================================
# 1. DATA LOAD
# ================================================================
print("=" * 60); print("1. LOADING DATA"); print("=" * 60, flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

# ETH
feats_eth = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

# BTC
feats_btc = fe.build_features(btc)
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()

# Align BTC features to ETH timestamps
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_ts) - 1)
# BTC close log return 1min
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[idx[1:]], 1e-8) / np.maximum(BTC_C[idx[:-1]], 1e-8))
del BTC_ts, BTC_C; gc.collect()

feats_eth_np = feats_eth.to_numpy().astype(np.float32)
feats_btc_np = feats_btc.to_numpy().astype(np.float32)
del feats_eth, feats_btc; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)

# ETH feature matrix (ETH feats + BTC lr1 as cross-asset feat)
X_eth_all = np.concatenate([feats_eth_np[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
# BTC feature matrix (BTC feats aligned to ETH timestamps)
X_btc_all = feats_btc_np[idx[:-H]]
del feats_eth_np, feats_btc_np, B_lr1, C_all; gc.collect()

ts_all_used = ts_all[:-H]
del ts_all; gc.collect()

# Time splits
tre = int(dtm.datetime.strptime(config.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
mv_mask = (ts_all_used >= es_end) & (ts_all_used < meta_end)
te_mask = ts_all_used >= meta_end

y_all = label; r_all = ret_future
del label, ret_future; gc.collect()

def split_and_norm(X_all, y_all, r_all, name):
    """Split train/es/mv/te, z-score norm per col."""
    X_tr = X_all[tr_mask].copy()
    X_es = X_all[es_mask].copy()
    X_mv = X_all[mv_mask].copy()
    X_te = X_all[te_mask].copy()
    y_tr = y_all[tr_mask]; y_es = y_all[es_mask]
    y_mv = y_all[mv_mask]; y_te = y_all[te_mask]
    r_tr = r_all[tr_mask]
    
    FT = X_tr.shape[1]
    for j in range(FT):
        col = X_tr[:, j]
        col[np.isnan(col)] = 0
        lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
        m, s = col.mean(), col.std() + 1e-6
        X_tr[:, j] = np.clip((col - m) / s, -5, 5)
        X_es[:, j] = np.clip((np.nan_to_num(X_es[:, j], nan=0) - m) / s, -5, 5)
        X_mv[:, j] = np.clip((np.nan_to_num(X_mv[:, j], nan=0) - m) / s, -5, 5)
        X_te[:, j] = np.clip((np.nan_to_num(X_te[:, j], nan=0) - m) / s, -5, 5)
    
    print(f"  [{name}] FT={FT} tr={len(X_tr)} es={len(X_es)} mv={len(X_mv)} te={len(X_te)}", flush=True)
    return X_tr, X_es, X_mv, X_te, y_tr, y_es, y_mv, y_te, r_tr

print("  Splitting ETH features...", flush=True)
X_eth_tr, X_eth_es, X_eth_mv, X_eth_te, y_tr, y_es, y_mv, y_te, r_tr = \
    split_and_norm(X_eth_all, y_all, r_all, "ETH")
del X_eth_all; gc.collect()

print("  Splitting BTC features...", flush=True)
X_btc_tr, X_btc_es, X_btc_mv, X_btc_te, _, _, _, _, _ = \
    split_and_norm(X_btc_all, y_all, r_all, "BTC")
del X_btc_all; gc.collect()

ts_te = ts_all_used[te_mask]
del ts_all_used, y_all, r_all; gc.collect()

# ================================================================
# 2. ETH TREE-ENS (LGBM 5 + CatBoost 5)
# ================================================================
print(f"\n{'='*60}"); print("2. ETH TREE-ENS"); print("="*60, flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

pos = y_tr.mean()
pw = np.where(y_tr > 0.5, (1 - pos) / pos, pos / (1 - pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)
del r_tr, ret_abs, lo_q, hi_q, ext_mask, pw, rw; gc.collect()

# LGBM
pvs_lgb_t = []; pvs_lgb_m = []
lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_eth_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_eth_es, label=y_es, reference=tr_ds)
    t0s = time.time()
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(1000)])
    pvs_lgb_t.append(bst.predict(X_eth_te))
    pvs_lgb_m.append(bst.predict(X_eth_mv))
    print(f"  LGBM s={s}: best_iter={bst.best_iteration_} [{time.time()-t0s:.0f}s]", flush=True)
    del bst; gc.collect()

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_m)
print(f"  LGBM: MV={roc_auc_score(y_mv, pv_lgb_mv):.4f} TE={roc_auc_score(y_te, pv_lgb_te):.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m; gc.collect()

# CatBoost
pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70]:
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3.0, random_seed=s, verbose=0,
                            early_stopping_rounds=200, eval_metric='AUC',
                            bagging_temperature=1.0, random_strength=0.5)
    t0s = time.time()
    cb.fit(X_eth_tr, y_tr, eval_set=[(X_eth_es, y_es)], sample_weight=sw, use_best_model=True)
    pvs_cat_t.append(cb.predict_proba(X_eth_te)[:, 1])
    pvs_cat_m.append(cb.predict_proba(X_eth_mv)[:, 1])
    print(f"  CatBoost s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
auc_tree_te = roc_auc_score(y_te, pv_tree_te); auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv)
print(f"  ★ TREE-ENS: MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)

del X_eth_tr, X_eth_es, sw; gc.collect()

# ================================================================
# 3. ETH NN (ResMLP + top20 interactions, full 2.4M train)
# ================================================================
print(f"\n{'='*60}"); print("3. ETH NN-ENS (ResMLP + top20 int)"); print("="*60, flush=True)

# Need reload ETH train for NN (we deleted it above!)
# Actually let's just reload from X_eth_all... oh wait we deleted that too.
# Let's compute ETH interactions from what we have.

# Actually let me just reload the train split since we need it for NN
# (we only deleted the normalized X_eth_tr; but original X_eth_all is gone too)
# Let me recompute

print("  Reloading ETH train split for NN...", flush=True)
# We need feats_eth and B_lr1 again
del X_btc_tr, X_btc_es; gc.collect()   # free memory first

eth2 = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats_eth2 = fe.build_features(eth2)
ts2 = eth2['ts'].to_numpy().astype(np.int64)
C2 = eth2['close'].to_numpy().astype(np.float64)
del eth2; gc.collect()

feats_eth2_np = feats_eth2.to_numpy().astype(np.float32)
del feats_eth2; gc.collect()

# BTC lr1 again
btc2 = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts2 = btc2['ts'].to_numpy().astype(np.int64)
BTC_C2 = btc2['close'].to_numpy().astype(np.float64)
del btc2; gc.collect()
idx2 = np.searchsorted(BTC_ts2, ts2, side="right") - 1
idx2 = np.clip(idx2, 0, len(BTC_ts2) - 1)
B_lr1_2 = np.zeros(len(ts2), dtype=np.float64)
B_lr1_2[1:] = np.log(np.maximum(BTC_C2[idx2[1:]], 1e-8) / np.maximum(BTC_C2[idx2[:-1]], 1e-8))
del BTC_ts2, BTC_C2; gc.collect()

X_eth_all2 = np.concatenate([feats_eth2_np[:-H], B_lr1_2[:-H, np.newaxis].astype(np.float32)], axis=1)
del feats_eth2_np, B_lr1_2, ts2, C2; gc.collect()

# Now normalize ETH train again
X_eth_tr2 = X_eth_all2[tr_mask].copy()
FT = X_eth_tr2.shape[1]
for j in range(FT):
    col = X_eth_tr2[:, j]
    col[np.isnan(col)] = 0
    m, s = col.mean(), col.std() + 1e-6
    X_eth_tr2[:, j] = np.clip((col - m) / s, -5, 5)
del X_eth_all2; gc.collect()

# Feature interactions: top20 by abs corr with label
print("  Computing top20 feature interactions...", flush=True)
corrs = np.array([abs(np.corrcoef(X_eth_tr2[:, j], y_tr)[0, 1]) for j in range(FT)])
top20 = np.argsort(-corrs)[:20]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i + 1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_eth_tr_nn = add_interactions(X_eth_tr2, top20)
X_eth_es_nn = add_interactions(X_eth_es, top20)
X_eth_mv_nn = add_interactions(X_eth_mv, top20)
X_eth_te_nn = add_interactions(X_eth_te, top20)
del X_eth_tr2, X_eth_es, X_eth_mv, X_eth_te; gc.collect()
FT_NN = X_eth_tr_nn.shape[1]
print(f"  ETH NN FT={FT_NN} (67+1+20+190=278)", flush=True)

# NN model
class ResMLP(nn.Module):
    def __init__(self, ft, hs, drop=0.3):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hs[i], hs[i + 1]), nn.BatchNorm1d(hs[i + 1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Linear(hs[-1], 1)

    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x) if x.shape == b(x).shape else b(x)
        return self.head(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])))).numpy())
    return np.concatenate(pv)

def train_resmlp(m, Xtr, ytr, Xes, yes, Xte, Xmv,
                 ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; best_ep = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            if smooth > 0:
                yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        auc_es = roc_auc_score(yes, evaluate(m, Xes))
        if auc_es > best + 1e-5:
            best = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat:
                break
    if bst:
        m.load_state_dict(bst)
    return evaluate(m, Xte), evaluate(m, Xmv)

pvs_eth_nn_t = []; pvs_eth_nn_m = []
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = ResMLP(FT_NN, [256, 128], 0.3)
    t0s = time.time()
    pv_t, pv_m = train_resmlp(m, X_eth_tr_nn, y_tr, X_eth_es_nn, y_es,
                              X_eth_te_nn, X_eth_mv_nn,
                              ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  ETH NN s={s}: MV={auc_m:.4f} TE={auc_t:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_eth_nn_t.append(pv_t); pvs_eth_nn_m.append(pv_m)
    del m; gc.collect()

pv_eth_nn_te = rank_agg(pvs_eth_nn_t)
pv_eth_nn_mv = rank_agg(pvs_eth_nn_m)
auc_eth_nn_te = roc_auc_score(y_te, pv_eth_nn_te)
auc_eth_nn_mv = roc_auc_score(y_mv, pv_eth_nn_mv)
print(f"  ★ ETH NN-ENS: MV={auc_eth_nn_mv:.4f} TE={auc_eth_nn_te:.4f}", flush=True)
del X_eth_tr_nn, X_eth_es_nn, X_eth_mv_nn, X_eth_te_nn; gc.collect()
del pvs_eth_nn_t, pvs_eth_nn_m; gc.collect()

# ================================================================
# 4. BTC NN (BTC features → predict ETH H15 label)
# ================================================================
print(f"\n{'='*60}"); print("4. BTC NN-ENS (BTC feats → ETH label)"); print("="*60, flush=True)

# We need to recompute BTC splits since we deleted them
btc3 = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
feats_btc3 = fe.build_features(btc3)
del btc3; gc.collect()

X_btc_all3 = feats_btc3.to_numpy().astype(np.float32)
del feats_btc3; gc.collect()

X_btc_tr3 = X_btc_all3[tr_mask].copy()
X_btc_es3 = X_btc_all3[es_mask].copy()
X_btc_mv3 = X_btc_all3[mv_mask].copy()
X_btc_te3 = X_btc_all3[te_mask].copy()
del X_btc_all3; gc.collect()

FT_BTC = X_btc_tr3.shape[1]
# z-score norm
for j in range(FT_BTC):
    col = X_btc_tr3[:, j]
    col[np.isnan(col)] = 0
    m, s = col.mean(), col.std() + 1e-6
    X_btc_tr3[:, j] = np.clip((col - m) / s, -5, 5)
    X_btc_es3[:, j] = np.clip((np.nan_to_num(X_btc_es3[:, j], nan=0) - m) / s, -5, 5)
    X_btc_mv3[:, j] = np.clip((np.nan_to_num(X_btc_mv3[:, j], nan=0) - m) / s, -5, 5)
    X_btc_te3[:, j] = np.clip((np.nan_to_num(X_btc_te3[:, j], nan=0) - m) / s, -5, 5)

# BTC interactions: top15 by abs corr with ETH label (BTC features ~67, but interactions might be noisy)
print("  Computing BTC top15 interactions...", flush=True)
corrs_btc = np.array([abs(np.corrcoef(X_btc_tr3[:, j], y_tr)[0, 1]) for j in range(FT_BTC)])
top15_btc = np.argsort(-corrs_btc)[:15]

X_btc_tr_nn = add_interactions(X_btc_tr3, top15_btc)
X_btc_es_nn = add_interactions(X_btc_es3, top15_btc)
X_btc_mv_nn = add_interactions(X_btc_mv3, top15_btc)
X_btc_te_nn = add_interactions(X_btc_te3, top15_btc)
del X_btc_tr3, X_btc_es3, X_btc_mv3, X_btc_te3; gc.collect()
FT_BTC_NN = X_btc_tr_nn.shape[1]
print(f"  BTC NN FT={FT_BTC_NN}", flush=True)

pvs_btc_nn_t = []; pvs_btc_nn_m = []
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = ResMLP(FT_BTC_NN, [256, 128], 0.3)
    t0s = time.time()
    pv_t, pv_m = train_resmlp(m, X_btc_tr_nn, y_tr, X_btc_es_nn, y_es,
                              X_btc_te_nn, X_btc_mv_nn,
                              ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  BTC NN s={s}: MV={auc_m:.4f} TE={auc_t:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_btc_nn_t.append(pv_t); pvs_btc_nn_m.append(pv_m)
    del m; gc.collect()

pv_btc_nn_te = rank_agg(pvs_btc_nn_t)
pv_btc_nn_mv = rank_agg(pvs_btc_nn_m)
auc_btc_nn_te = roc_auc_score(y_te, pv_btc_nn_te)
auc_btc_nn_mv = roc_auc_score(y_mv, pv_btc_nn_mv)
print(f"  ★ BTC NN-ENS: MV={auc_btc_nn_mv:.4f} TE={auc_btc_nn_te:.4f}", flush=True)
del X_btc_tr_nn, X_btc_es_nn, X_btc_mv_nn, X_btc_te_nn; gc.collect()
del pvs_btc_nn_t, pvs_btc_nn_m; gc.collect()

# ================================================================
# 5. STACK: 3-way blend on ranks
# ================================================================
print(f"\n{'='*60}"); print("5. STACK 3-WAY BLEND"); print("="*60, flush=True)

def to_rank(pv):
    return np.argsort(np.argsort(pv)).astype(np.float64) / (len(pv) - 1)

r_tree_mv = to_rank(pv_tree_mv)
r_eth_nn_mv = to_rank(pv_eth_nn_mv)
r_btc_nn_mv = to_rank(pv_btc_nn_mv)

r_tree_te = to_rank(pv_tree_te)
r_eth_nn_te = to_rank(pv_eth_nn_te)
r_btc_nn_te = to_rank(pv_btc_nn_te)

# Check tail correlations
def tail_corr(a, b, q_low=99):
    m = np.argsort(-a)[:int(len(a) * (100 - q_low) / 100)]
    return np.corrcoef(a[m], b[m])[0, 1]

print("  Tail correlations (top 1%):", flush=True)
print(f"    Tree vs ETH_NN:     overall={np.corrcoef(r_tree_mv, r_eth_nn_mv)[0,1]:.4f}  tail1%={tail_corr(r_tree_mv, r_eth_nn_mv):.4f}", flush=True)
print(f"    Tree vs BTC_NN:     overall={np.corrcoef(r_tree_mv, r_btc_nn_mv)[0,1]:.4f}  tail1%={tail_corr(r_tree_mv, r_btc_nn_mv):.4f}", flush=True)
print(f"    ETH_NN vs BTC_NN:   overall={np.corrcoef(r_eth_nn_mv, r_btc_nn_mv)[0,1]:.4f}  tail1%={tail_corr(r_eth_nn_mv, r_btc_nn_mv):.4f}", flush=True)

# Grid search 3-way blend weights
best_auc = 0.0; best_w = (0.4, 0.4, 0.2)  # default: Tree 0.4, ETH_NN 0.4, BTC_NN 0.2
for wt in np.arange(0.0, 1.01, 0.1):
    for we in np.arange(0.0, 1.01 - wt, 0.1):
        wb = 1.0 - wt - we
        if wb < -1e-6:
            continue
        pv = wt * r_tree_mv + we * r_eth_nn_mv + max(wb, 0) * r_btc_nn_mv
        auc = roc_auc_score(y_mv, pv)
        if auc > best_auc:
            best_auc = auc; best_w = (wt, we, max(wb, 0))

# Fine-grained around best
wt_c, we_c, wb_c = best_w
for wt in np.arange(max(0, wt_c - 0.15), min(1, wt_c + 0.151), 0.03):
    for we in np.arange(max(0, we_c - 0.15), min(1, we_c + 0.151 - wt), 0.03):
        wb = 1.0 - wt - we
        if wb < -1e-6:
            continue
        pv = wt * r_tree_mv + we * r_eth_nn_mv + max(wb, 0) * r_btc_nn_mv
        auc = roc_auc_score(y_mv, pv)
        if auc > best_auc:
            best_auc = auc; best_w = (wt, we, max(wb, 0))

wt, we, wb = best_w
print(f"  Optimal blend: Tree={wt:.2f} ETH_NN={we:.2f} BTC_NN={wb:.2f}  MV AUC={best_auc:.4f}", flush=True)

pv_stack_te = wt * r_tree_te + we * r_eth_nn_te + wb * r_btc_nn_te
auc_stack_te = roc_auc_score(y_te, pv_stack_te)
print(f"  ★ STACK TE AUC={auc_stack_te:.4f}", flush=True)

# ================================================================
# 6. ROLLING QUANTILE NO-LOOKAHEAD EVAL
# ================================================================
print(f"\n{'='*60}"); print("6. ROLLING QUANTILE EVAL"); print("="*60, flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def rolling_eval(name, pv, y, ts, days, q_list=[97, 98, 98.5, 99, 99.2, 99.5, 99.8]):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400
    day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    window = 30

    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    header = f"    {'q':>6} {'acc':>8} {'tpd':>6} {'n':>7}"
    print(header, flush=True)
    for q in q_list:
        trades = []
        for d in range(window, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - window]; hist_hi = day_lo
            hist_mask = (ts_arr >= hist_lo) & (ts_arr < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100:
                continue
            today_mask = (ts_arr >= day_lo) & (ts_arr < day_hi)
            thr = np.percentile(hist_pv, q)
            pick = pv_arr[today_mask] >= thr
            if pick.sum() > 0:
                trades.extend(y_arr[pick].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            marker = " ★" if (q >= 99 and tpd >= 14 and acc >= 65) else ""
            print(f"    {q:>6.1f} {acc:>7.1f}% {tpd:>6.1f} {len(trades):>7}{marker}", flush=True)

for name, pv in [('LGBM', pv_lgb_te), ('CatBoost', pv_cat_te),
                 ('TREE-ENS', pv_tree_te), ('ETH_NN-ENS', pv_eth_nn_te),
                 ('BTC_NN-ENS', pv_btc_nn_te), ('★ STACK 3WAY', pv_stack_te)]:
    rolling_eval(name, pv, y_te, ts_te, DAYS)

# ================================================================
# 7. SAVE RESULTS
# ================================================================
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/final_v1.npz',
         pv_lgb_te=pv_lgb_te, pv_cat_te=pv_cat_te,
         pv_tree_te=pv_tree_te, pv_tree_mv=pv_tree_mv,
         pv_eth_nn_te=pv_eth_nn_te, pv_eth_nn_mv=pv_eth_nn_mv,
         pv_btc_nn_te=pv_btc_nn_te, pv_btc_nn_mv=pv_btc_nn_mv,
         pv_stack_te=pv_stack_te,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te,
         weights=np.array([wt, we, wb]))
print(f"\nSaved final_v1.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"FINAL V1 DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  Tree     TE AUC={auc_tree_te:.4f}", flush=True)
print(f"  ETH NN   TE AUC={auc_eth_nn_te:.4f}", flush=True)
print(f"  BTC NN   TE AUC={auc_btc_nn_te:.4f}", flush=True)
print(f"  Stack    TE AUC={auc_stack_te:.4f} (w_T={wt:.2f} w_E={we:.2f} w_B={wb:.2f})", flush=True)
print(f"{'='*60}", flush=True)
