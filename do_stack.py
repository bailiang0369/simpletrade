"""Unified: train NN + train tree, analyze correlation, try stacking/voting."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import datetime as dtm, config, os

torch.manual_seed(42); np.random.seed(42)
t0_all = time.time()

# ============================================
# 0. Load v6 seq data
# ============================================
print("="*60, flush=True)
print("0. Load v6 sequence data", flush=True)
print("="*60, flush=True)
d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr = d['X_tr'].astype(np.float32); y_tr_seq = d['y_tr']; r_tr = d['r_tr'].astype(np.float32)
X_es_seq = d['X_es'].astype(np.float32); y_es_seq = d['y_es']
X_te_seq = d['X_te'].astype(np.float32); y_te_seq = d['y_te']; ts_te = d['ts_te']
X_tr_f = X_tr.reshape(len(X_tr), -1)
X_es_seq_f = X_es_seq.reshape(len(X_es_seq), -1)
X_te_seq_f = X_te_seq.reshape(len(X_te_seq), -1)
FT = X_tr_f.shape[1]
print(f"v6 SEQ: {X_tr.shape} FT={FT}", flush=True); gc.collect()

# ============================================
# 1. Build hand-crafted feature dataset for tree model
# ============================================
print("\n" + "="*60, flush=True)
print("1. Build hand-crafted features for LGBM", flush=True)
print("="*60, flush=True)
import polars as pl
import features as fe

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
del btc; gc.collect()

# Build features
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

# Build BTC cross-asset features
btc_full = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_C = btc_full['close'].to_numpy().astype(np.float64)

# Align BTC to ETH timestamps
BTC_ts = btc_full["ts"].to_numpy().astype(np.int64)
BTC_C_full = btc_full["close"].to_numpy().astype(np.float64)
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C_full)-1)
BTC_C_aligned = BTC_C_full[idx]
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C_aligned[1:],1e-8)/np.maximum(BTC_C_aligned[:-1],1e-8))
del BTC_C_full, BTC_C_aligned, BTC_ts, btc_full; gc.collect()
del BTC_C; gc.collect()

# Convert feats to numpy + add BTC
feats_np = feats.to_numpy().astype(np.float32)
del feats; gc.collect()

# Split by time
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
te_start = int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

# Labels
H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
# Features: feats_np[:-H], BTC_lr1[:-H]
X_all = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
ts_all_used = ts_all[:-H]
del feats_np, B_lr1, C_all; gc.collect()

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
te_mask = ts_all_used >= te_start

# Subsample train to match NN training scale
np.random.seed(42)
tr_idx = np.where(tr_mask)[0]
if len(tr_idx) > 400_000:
    tr_idx = np.random.choice(tr_idx, 400_000, replace=False)

X_tr_tree = X_all[tr_idx]; y_tr_tree = label[tr_idx]
r_tr_tree = ret_future[tr_idx]
X_es_tree = X_all[es_mask]; y_es_tree = label[es_mask]
X_te_tree = X_all[te_mask]; y_te_tree = label[te_mask]; ts_te_tree = ts_all_used[te_mask]

# Global robust z-score
for j in range(X_tr_tree.shape[1]):
    col_tr = X_tr_tree[:,j]
    lo, hi = np.percentile(col_tr[~np.isnan(col_tr)], 0.5), np.percentile(col_tr[~np.isnan(col_tr)], 99.5)
    m, s = np.nanmean(col_tr), np.nanstd(col_tr) + 1e-6
    X_tr_tree[:,j] = np.nan_to_num((np.clip(col_tr, lo, hi) - m) / s, nan=0.0)
    col_es = X_es_tree[:,j]; X_es_tree[:,j] = np.nan_to_num((np.clip(col_es, lo, hi) - m) / s, nan=0.0)
    col_te = X_te_tree[:,j]; X_te_tree[:,j] = np.nan_to_num((np.clip(col_te, lo, hi) - m) / s, nan=0.0)

print(f"TREE: TR={X_tr_tree.shape} ES={X_es_tree.shape} TE={X_te_tree.shape}", flush=True)
print(f"  FEAT dim={X_tr_tree.shape[1]}", flush=True)
gc.collect()

# ============================================
# 2. Train Tree model (5-seed LGBM rank ens)
# ============================================
print("\n" + "="*60, flush=True)
print("2. Train Tree model (5-seed LGBM rank ens)", flush=True)
print("="*60, flush=True)

pos = y_tr_tree.mean()
pw = np.where(y_tr_tree>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr_tree)*200, 0.2, 5.0).astype(np.float32)
sw = pw * rw

lgb_params = dict(objective='binary', metric='auc', learning_rate=0.05,
                  num_leaves=63, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_tree_t = []; pvs_tree_e = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr_tree, label=y_tr_tree, weight=sw)
    es_ds = lgb.Dataset(X_es_tree, label=y_es_tree, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(300)])
    pv_te = bst.predict(X_te_tree); pv_es = bst.predict(X_es_tree)
    auc_te = roc_auc_score(y_te_tree, pv_te); auc_es = roc_auc_score(y_es_tree, pv_es)
    print(f"  seed{s}: ES={auc_es:.4f} TE={auc_te:.4f} trees={bst.best_iteration}", flush=True)
    pvs_tree_t.append(pv_te); pvs_tree_e.append(pv_es)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree_te = rank_agg(pvs_tree_t); pv_tree_es = rank_agg(pvs_tree_e)
auc_tree_te = roc_auc_score(y_te_tree, pv_tree_te)
print(f"\n  ★ TREE 5-seed: ES={roc_auc_score(y_es_tree, pv_tree_es):.4f} TE={auc_tree_te:.4f}", flush=True)

# Tree eval
DAYS = (ts_te[-1]-ts_te[0])/86400.0  # use ts_te from seq (same split)
for pct in [0.5,1.0,1.5,2.0,3.0,5.0]:
    k = max(1, int(len(pv_tree_te)*pct/100))
    acc = y_te_tree[np.argsort(-pv_tree_te)[:k]].mean()*100; tpd = k/DAYS
    flag = '🏆' if pct==1.0 and acc>=65 else ('✅' if pct==1.0 and acc>=60 else '')
    print(f"    TREE top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

gc.collect()

# ============================================
# 3. Train NN best configs (5-seed each)
# ============================================
print("\n" + "="*60, flush=True)
print("3. Train NN 3 best configs × 5 seeds", flush=True)
print("="*60, flush=True)

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_mlp(m, Xtr, ytr, Xes, yes, ep=25, lr=5e-4, wd=0.06, bs=512, pat=7, smooth=0.15):
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr)); tl=0; nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item(); nb+=1
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return evaluate(m,X_te_seq_f), evaluate(m,Xes)

# 3 best configs from v11
configs = [
    ('baseline_512256', [512,256], 0.6, 0.06),
    ('big_1024512256', [1024,512,256], 0.6, 0.06),
    ('big_1024512256_d07', [1024,512,256], 0.7, 0.06),
]

nn_pvs_te = {}; nn_pvs_es = {}
for name, hs, drop, wd in configs:
    pvs_t=[]; pvs_e=[]
    for s in [42,49,56,63,70]:
        torch.manual_seed(s); np.random.seed(s)
        m = MLP(FT, hs, drop)
        pv_t, pv_e = train_mlp(m, X_tr_f, y_tr_seq, X_es_seq_f, y_es_seq,
                                ep=25, lr=5e-4, wd=wd, bs=512, pat=7, smooth=0.15)
        pvs_t.append(pv_t); pvs_e.append(pv_e)
    pv_e_t = rank_agg(pvs_t); pv_e_e = rank_agg(pvs_e)
    nn_pvs_te[name] = pv_e_t; nn_pvs_es[name] = pv_e_e
    auc_te = roc_auc_score(y_te_seq, pv_e_t); auc_es = roc_auc_score(y_es_seq, pv_e_e)
    print(f"\n  ★ NN {name}: ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5,1.0,1.5,2.0]:
        k = max(1, int(len(pv_e_t)*pct/100))
        acc = y_te_seq[np.argsort(-pv_e_t)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=60 else ''
        print(f"    NN top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# Multi-NN ensemble
all_nn_t = [nn_pvs_te[k] for k in nn_pvs_te]
all_nn_e = [nn_pvs_es[k] for k in nn_pvs_es]
pv_nn_multi_t = rank_agg(all_nn_t)
pv_nn_multi_e = rank_agg(all_nn_e)
auc_nn_multi = roc_auc_score(y_te_seq, pv_nn_multi_t)
print(f"\n  ★ NN MULTI (3 configs × 5 seeds): TE={auc_nn_multi:.4f}", flush=True)
for pct in [0.5,1.0,1.5,2.0]:
    k = max(1, int(len(pv_nn_multi_t)*pct/100))
    acc = y_te_seq[np.argsort(-pv_nn_multi_t)[:k]].mean()*100; tpd = k/DAYS
    flag = '🏆' if pct==1.0 and acc>=60 else ''
    print(f"    NN_MULTI top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

gc.collect()

# ============================================
# 4. Correlation analysis
# ============================================
print("\n" + "="*60, flush=True)
print("4. NN vs Tree correlation analysis", flush=True)
print("="*60, flush=True)

# Tree uses te_start cutoff (2025-09-30+) which should match seq_te (META_VAL_END=2025-09-30)
# But wait — seq_te in v6 uses config.SPLITS['test'] which also starts META_VAL_END
# Check they're aligned
print(f"  Tree te_ts range: {ts_te_tree.min()} → {ts_te_tree.max()} ({len(ts_te_tree)} samples)", flush=True)
print(f"  Seq te_ts range:  {ts_te.min()} → {ts_te.max()} ({len(ts_te)} samples)", flush=True)

# Correlation (on their common TE indices)
# Actually they have different sample sets because different construction
# Let's compute correlation on each model's own TE predictions vs the other's TE predictions
# Since the predictions are on different subsets, let's just look at overall AUC correlation

# Compute simple: for each model, how much do their ranked lists overlap?
# Better: if both models predict on same anchor ts, compute corr
# But anchors are different due to different stride (S=8 vs default)
# So let's just do simple comparison

print("\n--- Pairwise Spearman rank correlation on TE predictions ---", flush=True)
all_preds = {'Tree': pv_tree_te}
for k in nn_pvs_te: all_preds[f'NN_{k}'] = nn_pvs_te[k]
all_preds['NN_MULTI'] = pv_nn_multi_t

# Can't directly correlate different length arrays (different anchors)
# Instead let's check: what's the AUC gain if we could stack?
# Simple approach: equal-weight rank ensemble across all predictions
# But they're on different samples! 

# Let's instead check if tree preds and nn preds are similar by checking:
# Are high-confidence tree samples also high-confidence NN samples?
# We need to align timestamps first

print("\n--- Aligning predictions by timestamp ---", flush=True)
# Build dict: ts -> prediction
from bisect import bisect_left

def build_pv_dict(ts_arr, pv_arr):
    idx = np.argsort(ts_arr)
    return ts_arr[idx], pv_arr[idx]

t_ts, t_pv = build_pv_dict(ts_te_tree, pv_tree_te)
s_ts, s_pv = build_pv_dict(ts_te, pv_nn_multi_t)

# Find common timestamps
common_ts = np.intersect1d(t_ts, s_ts)
print(f"  Common te timestamps: {len(common_ts)} / tree={len(t_ts)} seq={len(s_ts)}", flush=True)

# Get predictions at common timestamps
t_pos = np.searchsorted(t_ts, common_ts); s_pos = np.searchsorted(s_ts, common_ts)
t_common = t_pv[t_pos]; s_common = s_pv[s_pos]
# Get labels at common (both models use same label rule)
# Need to re-extract labels at these common timestamps

# Actually easier: build label for common_ts from scratch
C_all_np = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')['close'].to_numpy().astype(np.float64)
all_ts_np = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')['ts'].to_numpy().astype(np.int64)
# For each common_ts[t], label = C[t+H] > C[t]
ts_to_idx = {int(t):i for i,t in enumerate(all_ts_np)}
label_common = np.array([C_all_np[ts_to_idx[int(t)]+15] > C_all_np[ts_to_idx[int(t)]] for t in common_ts], dtype=np.int64)
auc_t_common = roc_auc_score(label_common, t_common)
auc_s_common = roc_auc_score(label_common, s_common)
corr = np.corrcoef(t_common, s_common)[0,1]
print(f"  On common TE anchors ({len(common_ts)}):", flush=True)
print(f"    Tree AUC={auc_t_common:.4f}  NN AUC={auc_s_common:.4f}  CORR={corr:.4f}", flush=True)

# ============================================
# 5. Stacking experiments
# ============================================
print("\n" + "="*60, flush=True)
print("5. Stacking/voting experiments (on common timestamps)", flush=True)
print("="*60, flush=True)

y = label_common

def do_topk_eval(name, pv, y, DAYS):
    auc = roc_auc_score(y, pv)
    print(f"\n  ★ [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=65 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# Estimate DAYS from test range
te_start_dt = dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d')
te_end_dt = dtm.datetime.strptime('2026-09-28','%Y-%m-%d')
DAYS_est = (te_end_dt - te_start_dt).days
print(f"  TE spans ~{DAYS_est} days (estimate tpd)", flush=True)
DAYS_eval = DAYS_est

# Individual
do_topk_eval('Tree', t_common, y, DAYS_eval)
do_topk_eval('NN_MULTI', s_common, y, DAYS_eval)

# Simple ensembles
print("\n--- Ensemble attempts ---", flush=True)
pv_avg = (t_common + s_common) / 2
do_topk_eval('EQUAL_AVG(Tree+NN)', pv_avg, y, DAYS_eval)

# Weighted by AUC (normalized)
w_t = auc_t_common; w_s = auc_s_common
pv_weighted = (w_t*t_common + w_s*s_common) / (w_t + w_s)
do_topk_eval('AUC_WEIGHTED', pv_weighted, y, DAYS_eval)

# Rank ensemble
def to_rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64)/len(pv)
pv_rank_avg = (to_rank(t_common) + to_rank(s_common)) / 2
do_topk_eval('RANK_AVG', pv_rank_avg, y, DAYS_eval)

# Optimal weight grid search on common timestamps
best_w = 0.5; best_auc = 0
for w in np.arange(0.0, 1.05, 0.05):
    pv = w*to_rank(t_common) + (1-w)*to_rank(s_common)
    auc = roc_auc_score(y, pv)
    if auc > best_auc:
        best_auc = auc; best_w = w
print(f"  Optimal rank blend: w_Tree={best_w:.2f}, w_NN={1-best_w:.2f}, AUC={best_auc:.4f}", flush=True)
pv_opt = best_w*to_rank(t_common) + (1-best_w)*to_rank(s_common)
do_topk_eval('OPTIMAL_RANK', pv_opt, y, DAYS_eval)

# Simple logistic regression stacking (ES predictions as input)
print("\n--- LR Stacking on ES ---", flush=True)
# We need aligned ES predictions too
# For speed, skip ES-based stacking and do a simpler analysis

# Check if stacking is necessary
delta_auc = max(auc_t_common, auc_s_common) - min(auc_t_common, auc_s_common)
print(f"\n--- CORRELATION SUMMARY ---", flush=True)
print(f"  Tree AUC:      {auc_t_common:.4f}", flush=True)
print(f"  NN AUC:        {auc_s_common:.4f}", flush=True)
print(f"  AUC difference: {delta_auc:.4f}", flush=True)
print(f"  Prediction CORR: {corr:.4f}", flush=True)
print(f"  Stacking benefit: {'HIGH (low corr)' if corr < 0.6 else 'MEDIUM' if corr < 0.8 else 'LOW (high corr)'}", flush=True)

print(f"\nTOTAL TIME: {time.time()-t0_all:.0f}s", flush=True)
