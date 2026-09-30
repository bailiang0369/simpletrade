"""Part 3 (v2): Per-hour LGB + 更多 BTC 特征 + 进阶堆叠"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np, datetime as dtm
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import config, features as fe

t0 = time.time()

# ============================================================
# 0. 重新构建 (加更多 BTC 特征)
# ============================================================
print("[0] 重新构建数据 (加更多 BTC 特征)...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)

# BTC 特征
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C = btc['close'].to_numpy().astype(np.float64)
BTC_O = btc['open'].to_numpy().astype(np.float64)
BTC_H = btc['high'].to_numpy().astype(np.float64)
BTC_L = btc['low'].to_numpy().astype(np.float64)
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C)-1)
BTC_Ca = BTC_C[idx]; BTC_Oa = BTC_O[idx]; BTC_Ha = BTC_H[idx]; BTC_La = BTC_L[idx]

# BTC log returns 多尺度
B_lr1 = np.zeros(len(ts_all), np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_Ca[1:],1e-8)/np.maximum(BTC_Ca[:-1],1e-8))
B_lr5 = np.zeros(len(ts_all), np.float64)
B_lr5[5:] = np.log(np.maximum(BTC_Ca[5:],1e-8)/np.maximum(BTC_Ca[:-5],1e-8))
B_lr30 = np.zeros(len(ts_all), np.float64)
B_lr30[30:] = np.log(np.maximum(BTC_Ca[30:],1e-8)/np.maximum(BTC_Ca[:-30],1e-8))
B_lr120 = np.zeros(len(ts_all), np.float64)
B_lr120[120:] = np.log(np.maximum(BTC_Ca[120:],1e-8)/np.maximum(BTC_Ca[:-120],1e-8))

# BTC rvol
B_rvol30 = np.zeros(len(ts_all), np.float64)
for i in range(30, len(ts_all)):
    B_rvol30[i] = B_lr1[max(0,i-30):i].std() + 1e-8

# BTC z-score (位置)
B_pos60 = np.zeros(len(ts_all), np.float64)
for i in range(60, len(ts_all)):
    w_hi = BTC_Ha[max(0,i-60):i].max()
    w_lo = BTC_La[max(0,i-60):i].min()
    B_pos60[i] = (BTC_Ca[i] - w_lo) / (w_hi - w_lo + 1e-8)

# ETH-BTC 相关性 (滚动)
B_corr30 = np.zeros(len(ts_all), np.float64)
E_lr1 = np.zeros(len(ts_all), np.float64)
E_lr1[1:] = np.log(np.maximum(C_all[1:],1e-8)/np.maximum(C_all[:-1],1e-8))
for i in range(30, len(ts_all)):
    w_e = E_lr1[max(0,i-30):i]
    w_b = B_lr1[max(0,i-30):i]
    if w_e.std() > 1e-8 and w_b.std() > 1e-8:
        B_corr30[i] = np.corrcoef(w_e, w_b)[0,1]
del BTC_ts, BTC_C, BTC_O, BTC_H, BTC_L, BTC_Ca, BTC_Oa, BTC_Ha, BTC_La, E_lr1; gc.collect()

feats_np = feats.to_numpy().astype(np.float32)
del feats, eth, btc; gc.collect()

extra_btc = np.stack([B_lr1, B_lr5, B_lr30, B_lr120, B_rvol30, B_pos60, B_corr30], axis=1).astype(np.float32)
del B_lr1, B_lr5, B_lr30, B_lr120, B_rvol30, B_pos60, B_corr30; gc.collect()

X_all = np.concatenate([feats_np, extra_btc], axis=1)
del feats_np, extra_btc; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = X_all[:-H]; ts_all_u = ts_all[:-H]
del C_all, ts_all; gc.collect()

def ts_mask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all_u >= a) & (ts_all_u < b)

tr_idx = np.where(ts_mask(*config.SPLITS['train']))[0]
es_idx = np.where(ts_mask(*config.SPLITS['early_stop']))[0]
mv_idx = np.where(ts_mask(*config.SPLITS['meta_val']))[0]
te_idx = np.where(ts_mask(*config.SPLITS['test']))[0]

np.random.seed(42)
if len(tr_idx) > 1_500_000:
    tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)

X_tr = X_all[tr_idx].astype(np.float32)
y_tr = label[tr_idx]; r_tr = ret_future[tr_idx]
X_es = X_all[es_idx].astype(np.float32); y_es = label[es_idx]
X_mv = X_all[mv_idx].astype(np.float32); y_mv = label[mv_idx]
X_te = X_all[te_idx].astype(np.float32); y_te = label[te_idx]; ts_te = ts_all_u[te_idx]

hr_tr = (ts_all_u[tr_idx] % 86400) // 3600
hr_es = (ts_all_u[es_idx] % 86400) // 3600
hr_mv = (ts_all_u[mv_idx] % 86400) // 3600
hr_te = (ts_all_u[te_idx] % 86400) // 3600
del X_all, label, ret_future, ts_all_u; gc.collect()

# Robust z-score
print("  z-score...", flush=True)
for j in range(X_tr.shape[1]):
    col = X_tr[:,j]
    lo, hi = np.percentile(col[~np.isnan(col)], 0.5), np.percentile(col[~np.isnan(col)], 99.5)
    m, s = np.nanmean(col), np.nanstd(col) + 1e-6
    X_tr[:,j] = np.nan_to_num((np.clip(col, lo, hi) - m) / s, nan=0.0)
    X_es[:,j] = np.nan_to_num((np.clip(X_es[:,j], lo, hi) - m) / s, nan=0.0)
    X_mv[:,j] = np.nan_to_num((np.clip(X_mv[:,j], lo, hi) - m) / s, nan=0.0)
    X_te[:,j] = np.nan_to_num((np.clip(X_te[:,j], lo, hi) - m) / s, nan=0.0)

print(f"  X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)
gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def to_rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64) / len(pv)

# ============================================================
# 1. Tree (新特征) 5-seed + 负权重
# ============================================================
print("\n[1] Tree 新特征 5-seed + 负权重", flush=True)
abs_ret = np.abs(r_tr)
q90 = np.quantile(abs_ret, 0.90)
q10 = np.quantile(abs_ret, 0.10)
w_tree = np.where((abs_ret >= q90) | (abs_ret <= q10), 0.3, 1.0).astype(np.float32)
print(f"  负权重比例={((abs_ret >= q90) | (abs_ret <= q10)).mean()*100:.1f}%", flush=True)

lgb_params = dict(
    objective='binary', metric='auc', learning_rate=0.05,
    num_leaves=127, max_depth=-1, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
    lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,
    verbose=-1, n_jobs=-1,
)

pvs_tree_es = []; pvs_tree_mv = []; pvs_tree_te = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tree)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_tree_es.append(bst.predict(X_es))
    pvs_tree_mv.append(bst.predict(X_mv))
    pvs_tree_te.append(bst.predict(X_te))
    print(f"  seed{s}: TE AUC={roc_auc_score(y_te, pvs_tree_te[-1]):.4f} iters={bst.best_iteration}", flush=True)
    del bst, tr_ds, es_ds; gc.collect()

pv_tree_mv = rank_agg(pvs_tree_mv)
pv_tree_te = rank_agg(pvs_tree_te)
print(f"\n  ★ TREE NEW: MV AUC={roc_auc_score(y_mv, pv_tree_mv):.4f} TE AUC={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)
del pvs_tree_es, pvs_tree_mv, pvs_tree_te; gc.collect()

# ============================================================
# 2. Per-hour LGB (24个)
# ============================================================
print("\n[2] Per-hour LGB 24个", flush=True)

pv_ph_mv = pv_tree_mv.copy()
pv_ph_te = pv_tree_te.copy()

for h in range(24):
    tr_h = hr_tr == h; es_h = hr_es == h; mv_h = hr_mv == h; te_h = hr_te == h
    if tr_h.sum() < 3000: continue
    
    params_h = dict(objective='binary', metric='auc', learning_rate=0.05,
                     num_leaves=31, min_child_samples=50, feature_fraction=0.9,
                     bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.1,
                     verbose=-1, n_jobs=3, seed=42)
    tr_ds_h = lgb.Dataset(X_tr[tr_h], label=y_tr[tr_h], weight=w_tree[tr_h])
    es_ds_h = lgb.Dataset(X_es[es_h], label=y_es[es_h])
    bst_h = lgb.train(params_h, tr_ds_h, num_boost_round=2000,
                      valid_sets=[es_ds_h], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pv_ph_mv[mv_h] = bst_h.predict(X_mv[mv_h])
    pv_ph_te[te_h] = bst_h.predict(X_te[te_h])
    auc_h = roc_auc_score(y_te[te_h], pv_ph_te[te_h]) if te_h.sum() > 100 else 0.5
    print(f"  h{h:02d}: tr={tr_h.sum()} te={te_h.sum()} AUC={auc_h:.4f}", flush=True)
    del bst_h, tr_ds_h, es_ds_h; gc.collect()

pv_ph_rank_mv = to_rank(pv_ph_mv)
pv_ph_rank_te = to_rank(pv_ph_te)
print(f"\n  ★ Per-hour LGB: MV AUC={roc_auc_score(y_mv, pv_ph_rank_mv):.4f} TE AUC={roc_auc_score(y_te, pv_ph_rank_te):.4f}", flush=True)

# Global + PH blend
best_gw = 1.0; best_gauc = 0
for w in np.arange(0.0, 1.05, 0.1):
    pv = w*to_rank(pv_tree_mv) + (1-w)*pv_ph_rank_mv
    a = roc_auc_score(y_mv, pv)
    if a > best_gauc: best_gauc = a; best_gw = w
print(f"  Global+PH blend: w_global={best_gw:.1f}, MV AUC={best_gauc:.4f}", flush=True)

pv_gph_te = best_gw*to_rank(pv_tree_te) + (1-best_gw)*pv_ph_rank_te

# ============================================================
# 3. NN (新特征) 训练
# ============================================================
print("\n[3] NN 新特征 MLP 6-seed", flush=True)

# 子采样
np.random.seed(42)
sel = np.random.choice(len(X_tr), 800_000, replace=False)
X_tr_nn = X_tr[sel].copy(); y_tr_nn = y_tr[sel].copy()
gc.collect()

# 构造 NN 特征
corrs = np.array([np.corrcoef(X_tr_nn[:,j][~np.isnan(X_tr_nn[:,j])], y_tr_nn[~np.isnan(X_tr_nn[:,j])])[0,1]
                   if np.std(X_tr_nn[:,j][~np.isnan(X_tr_nn[:,j])]) > 1e-8 else 0.0
                   for j in range(X_tr_nn.shape[1])])
corrs = np.nan_to_num(corrs, nan=0.0)
topk = np.argsort(-np.abs(corrs))[:10]
print(f"  Top 10 corr: {[f'{corrs[j]:.3f}' for j in topk[:5]]}", flush=True)

def build_nn_feats(X, topk):
    from itertools import combinations
    parts = [X]
    parts.append(X[:, topk] ** 2)
    for i,j in combinations(topk, 2):
        parts.append((X[:,i] * X[:,j])[:, np.newaxis])
    return np.concatenate(parts, axis=1).astype(np.float32)

X_tr_nn = build_nn_feats(X_tr_nn, topk)
X_es_nn = build_nn_feats(X_es, topk)
X_mv_nn = build_nn_feats(X_mv, topk)
X_te_nn = build_nn_feats(X_te, topk)
print(f"  NN FT={X_tr_nn.shape[1]}", flush=True)
gc.collect()

import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def evaluate_nn(m, X, bs=8192):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_mlp(m, Xtr, ytr, Xes, yes, pat=6):
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    best_auc = 0.0; bst_state = None; ni = 0
    for e in range(30):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            yb = yb*0.95 + 0.5*0.05
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es = evaluate_nn(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best_auc + 1e-5:
            best_auc = auc_es
            bst_state = {k:v.detach().clone() for k,v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat: break
    if bst_state: m.load_state_dict(bst_state)
    return evaluate_nn(m, Xes), evaluate_nn(m, X_mv_nn), evaluate_nn(m, X_te_nn)

pvs_nn_mv = []; pvs_nn_te = []
for s in [42, 49, 56, 63, 70, 101]:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(X_tr_nn.shape[1], [256, 128], drop=0.3)
    _, pv_mv, pv_te = train_mlp(m, X_tr_nn, y_tr_nn, X_es_nn, y_es)
    pvs_nn_mv.append(pv_mv); pvs_nn_te.append(pv_te)
    print(f"  NN seed{s}: TE AUC={roc_auc_score(y_te, pv_te):.4f}", flush=True)
    del m; gc.collect()

pv_nn_mv = rank_agg(pvs_nn_mv)
pv_nn_te = rank_agg(pvs_nn_te)
print(f"\n  ★ NN 6-seed NEW: MV AUC={roc_auc_score(y_mv, pv_nn_mv):.4f} TE AUC={roc_auc_score(y_te, pv_nn_te):.4f}", flush=True)
del pvs_nn_mv, pvs_nn_te, X_tr_nn; gc.collect()

# ============================================================
# 4. 三者堆叠 (Tree + PH + NN)
# ============================================================
print("\n[4] 三者堆叠 (Tree + PH + NN)", flush=True)

rt_mv = to_rank(pv_tree_mv); rt_te = to_rank(pv_tree_te)
rph_mv = pv_ph_rank_mv; rph_te = pv_ph_rank_te
rn_mv = to_rank(pv_nn_mv); rn_te = to_rank(pv_nn_te)

# Tree+NN
best_tnw = 0.5; best_tnauc = 0
for w in np.arange(0.0, 1.05, 0.05):
    pv = w*rt_mv + (1-w)*rn_mv
    a = roc_auc_score(y_mv, pv)
    if a > best_tnauc: best_tnauc = a; best_tnw = w
print(f"  Tree+NN: w_Tree={best_tnw:.2f}, MV AUC={best_tnauc:.4f}", flush=True)

# PH+NN
best_phnw = 0.5; best_phnauc = 0
for w in np.arange(0.0, 1.05, 0.05):
    pv = w*rph_mv + (1-w)*rn_mv
    a = roc_auc_score(y_mv, pv)
    if a > best_phnauc: best_phnauc = a; best_phnw = w
print(f"  PH+NN: w_PH={best_phnw:.2f}, MV AUC={best_phnauc:.4f}", flush=True)

# Tree+PH+NN (粗扫)
best_3w = (1/3, 1/3, 1/3); best_3auc = 0
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05-w1, 0.1):
        w3 = 1 - w1 - w2
        pv = w1*rt_mv + w2*rph_mv + w3*rn_mv
        a = roc_auc_score(y_mv, pv)
        if a > best_3auc: best_3auc = a; best_3w = (w1, w2, w3)
print(f"  Tree+PH+NN: T={best_3w[0]:.1f} PH={best_3w[1]:.1f} N={best_3w[2]:.1f}, MV AUC={best_3auc:.4f}", flush=True)

# 细扫 Tree+PH+NN around best
b1, b2, b3 = best_3w
for dw1 in np.arange(-0.1, 0.11, 0.02):
    for dw2 in np.arange(-0.1, 0.11, 0.02):
        w1 = max(0, min(1, b1+dw1)); w2 = max(0, min(1-w1, b2+dw2)); w3 = 1-w1-w2
        pv = w1*rt_mv + w2*rph_mv + w3*rn_mv
        a = roc_auc_score(y_mv, pv)
        if a > best_3auc: best_3auc = a; best_3w = (w1, w2, w3)
print(f"  细扫: T={best_3w[0]:.2f} PH={best_3w[1]:.2f} N={best_3w[2]:.2f}, MV AUC={best_3auc:.4f}", flush=True)

# ============================================================
# 5. 全方法 TOP-K 评估
# ============================================================
print("\n[5] 全方法 TOP-K 评估 (Test)", flush=True)
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

methods = {
    'Tree_5seed': rt_te,
    'PH_LGB': rph_te,
    'Global+PH': pv_gph_te,
    'NN_6seed': rn_te,
    f'Tree+NN({best_tnw:.2f}T/{1-best_tnw:.2f}N)': best_tnw*rt_te + (1-best_tnw)*rn_te,
    f'PH+NN({best_phnw:.2f}PH/{1-best_phnw:.2f}N)': best_phnw*rph_te + (1-best_phnw)*rn_te,
    f'Tree+PH+NN({best_3w[0]:.2f}/{best_3w[1]:.2f}/{best_3w[2]:.2f})': best_3w[0]*rt_te + best_3w[1]*rph_te + best_3w[2]*rn_te,
}

for name, pv in methods.items():
    auc = roc_auc_score(y_te, pv)
    line = f"  [{name:>32s}] AUC={auc:.4f}"
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean() * 100; tpd = k / DAYS
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line, flush=True)

# ============================================================
# 6. Per-hour 内 top-k 选择 (非常规阈值)
# ============================================================
print("\n[6] Per-hour 内 top-k 选择", flush=True)
# 不做全局 top-1%, 而是每个 hour 内取 top-pct
pv_ph_topk = np.full(len(pv_tree_te), -1e18, dtype=np.float32)
for h in range(24):
    hmask = hr_te == h
    n = hmask.sum()
    if n < 50: continue
    scores = pv_tree_te[hmask]
    # 每个 hour 内取 top 2%
    k = max(1, int(n*0.02))
    ranked = np.argsort(-scores)
    top_idx = np.where(hmask)[0][ranked[:k]]
    pv_ph_topk[top_idx] = scores[ranked[:k]]

valid = pv_ph_topk > -1e9
print(f"  Per-hour top-2%: n={valid.sum()}", flush=True)
for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    k = max(1, int(valid.sum()*pct/100))
    idx = np.argsort(-pv_ph_topk[valid])[:k]
    acc = y_te[valid][idx].mean() * 100; tpd = k/DAYS
    print(f"    top{pct:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)

# ============================================================
# 7. 保存
# ============================================================
print("\n[7] 保存", flush=True)
np.savez('/workspace/models_saved/v2_preds.npz',
    pv_tree_mv=pv_tree_mv, pv_tree_te=pv_tree_te,
    pv_ph_mv=pv_ph_mv, pv_ph_te=pv_ph_te,
    pv_nn_mv=pv_nn_mv, pv_nn_te=pv_nn_te,
    y_mv=y_mv, y_te=y_te, ts_te=ts_te, hr_te=hr_te)
print(f"  Done ⏱ {time.time()-t0:.0f}s", flush=True)
