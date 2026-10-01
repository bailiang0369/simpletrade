"""V3 优化: 多架构 NN 集成 + CatBoost + 更强 Stack.
核心思路: 异构模型 → 更低 tail corr → 更好 Stack.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import datetime as dtm, config

torch.manual_seed(42); np.random.seed(42)
t0_all = time.time()

# ============================================
# Data pipeline (same as V1/V2)
# ============================================
print("Loading data...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)

btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts = btc["ts"].to_numpy().astype(np.int64)
BTC_C = btc["close"].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C)-1)
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[idx[1:]],1e-8)/np.maximum(BTC_C[idx[:-1]],1e-8))
del BTC_ts, BTC_C; gc.collect()

feats_np = feats.to_numpy().astype(np.float32)
del feats, eth; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
ts_all_used = ts_all[:-H]
del feats_np, B_lr1, C_all, ts_all; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
mv_mask = (ts_all_used >= es_end) & (ts_all_used < meta_end)
te_mask = ts_all_used >= meta_end

np.random.seed(42)
tr_idx = np.where(tr_mask)[0]
if len(tr_idx) > 800_000:
    tr_idx = np.random.choice(tr_idx, 800_000, replace=False)

X_tr = X_all[tr_idx].copy()
X_es = X_all[es_mask].copy()
X_mv = X_all[mv_mask].copy()
X_te = X_all[te_mask].copy()
y_tr = label[tr_idx]; r_tr = ret_future[tr_idx]
y_es = label[es_mask]; y_mv = label[mv_mask]; y_te = label[te_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

FT = X_tr.shape[1]
for j in range(FT):
    col = X_tr[:,j]; col[np.isnan(col)] = 0
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    m, s = col.mean(), col.std() + 1e-6
    X_tr[:,j] = np.clip((col - m) / s, -5, 5)
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()
print(f"  TR={X_tr.shape} TE={X_te.shape} FT={FT}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

# ============================================
# V3-Tree-LGBM: 10-seed (from V2 results)
# ============================================
print("\n" + "="*60, flush=True)
print("V3-Tree-LGBM: 10-seed", flush=True)
print("="*60, flush=True)

pos = y_tr.mean()
pw = np.where(y_tr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)

lgb_params = dict(objective='binary', metric='auc', learning_rate=0.05,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_lgb_t = []; pvs_lgb_m = []
for s in [42, 49, 56, 63, 70, 115, 122, 129, 136, 143]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(5000)])
    pvs_lgb_t.append(bst.predict(X_te)); pvs_lgb_m.append(bst.predict(X_mv))

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_m)
auc_lgb_te = roc_auc_score(y_te, pv_lgb_te); auc_lgb_mv = roc_auc_score(y_mv, pv_lgb_mv)
print(f"  ★ LGBM: MV={auc_lgb_mv:.4f} TE={auc_lgb_te:.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m; gc.collect()

# ============================================
# V3-Tree-CatBoost: 5-seed (异构!)
# ============================================
print("\n" + "="*60, flush=True)
print("V3-Tree-CatBoost: 5-seed", flush=True)
print("="*60, flush=True)

pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70]:
    cb = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, random_seed=s, verbose=0,
        early_stopping_rounds=200, eval_metric='AUC',
        bagging_temperature=1.0, random_strength=0.5,
    )
    t0s = time.time()
    cb.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], sample_weight=sw, use_best_model=True)
    pv_te = cb.predict_proba(X_te)[:,1]; pv_mv = cb.predict_proba(X_mv)[:,1]
    auc_te = roc_auc_score(y_te, pv_te); auc_mv = roc_auc_score(y_mv, pv_mv)
    print(f"  s={s}: MV={auc_mv:.4f} TE={auc_te:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_cat_t.append(pv_te); pvs_cat_m.append(pv_mv)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
auc_cat_te = roc_auc_score(y_te, pv_cat_te); auc_cat_mv = roc_auc_score(y_mv, pv_cat_mv)
print(f"  ★ CatBoost: MV={auc_cat_mv:.4f} TE={auc_cat_te:.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

# ============================================
# V3-Tree-Ens: LGBM + CatBoost rank ens
# ============================================
print("\n" + "="*60, flush=True)
print("V3-Tree-Ens: LGBM+CatBoost", flush=True)
print("="*60, flush=True)

pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
auc_tree_te = roc_auc_score(y_te, pv_tree_te); auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv)
print(f"  ★ TREE-ENS: MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)

# ============================================
# V3-NN: 3 异构 MLP 架构
# ============================================
print("\n" + "="*60, flush=True)
print("V3-NN: 3 异构架构 × 3 seeds each", flush=True)
print("="*60, flush=True)

corrs = np.array([abs(np.corrcoef(X_tr[:,j], y_tr)[0,1]) for j in range(FT)])
top15 = np.argsort(-corrs)[:15]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn = add_interactions(X_tr, top15)
X_es_nn = add_interactions(X_es, top15)
X_mv_nn = add_interactions(X_mv, top15)
X_te_nn = add_interactions(X_te, top15)
del X_tr, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn.shape[1]
print(f"  NN FT: {FT_NN}", flush=True)

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

def train_mlp(m, Xtr, ytr, Xes, yes, ep=30, lr=5e-4, wd=1e-3, bs=1024, pat=6, smooth=0.05):
    opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr))
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es=evaluate(m,Xes); auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return evaluate(m,X_te_nn), evaluate(m,Xes), evaluate(m,X_mv_nn)

# 3 异构架构
nn_configs = [
    ('MLP_sm  [256,128] d0.3 wd1e-3', [256, 128], 0.3, 1e-3, 5e-4),
    ('MLP_md  [512,256] d0.4 wd5e-3', [512, 256], 0.4, 5e-3, 5e-4),
    ('MLP_deep [256,256,128] d0.25 wd1e-4', [256, 256, 128], 0.25, 1e-4, 8e-4),
]

all_nn_t = []; all_nn_m = []
for label_nn, hs, drop, wd, lr_nn in nn_configs:
    print(f"\n  {label_nn}", flush=True)
    pvs_t = []; pvs_m = []
    for s in [42, 56, 70]:
        torch.manual_seed(s); np.random.seed(s)
        m = MLP(FT_NN, hs, drop)
        pv_t, pv_e, pv_m = train_mlp(m, X_tr_nn, y_tr, X_es_nn, y_es,
                                     ep=30, lr=lr_nn, wd=wd, bs=1024, pat=6, smooth=0.05)
        auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
        print(f"    s={s}: MV={auc_m:.4f} TE={auc_t:.4f}", flush=True)
        pvs_t.append(pv_t); pvs_m.append(pv_m); del m; gc.collect()
    all_nn_t.extend(pvs_t); all_nn_m.extend(pvs_m)

pv_nn_te = rank_agg(all_nn_t); pv_nn_mv = rank_agg(all_nn_m)
auc_nn_te = roc_auc_score(y_te, pv_nn_te); auc_nn_mv = roc_auc_score(y_mv, pv_nn_mv)
print(f"\n  ★ NN-ENS: MV={auc_nn_mv:.4f} TE={auc_nn_te:.4f}", flush=True)
del X_tr_nn, X_es_nn, X_mv_nn, X_te_nn; gc.collect()

# ============================================
# V3-Stack: Tree-Ens + NN-Ens
# ============================================
print("\n" + "="*60, flush=True)
print("V3-Stack: Tree-Ens + NN-Ens", flush=True)
print("="*60, flush=True)

pv_tree_mv_r = np.argsort(np.argsort(pv_tree_mv)).astype(np.float64)/len(pv_tree_mv)
pv_nn_mv_r = np.argsort(np.argsort(pv_nn_mv)).astype(np.float64)/len(pv_nn_mv)
pv_tree_te_r = np.argsort(np.argsort(pv_tree_te)).astype(np.float64)/len(pv_tree_te)
pv_nn_te_r = np.argsort(np.argsort(pv_nn_te)).astype(np.float64)/len(pv_nn_te)

# Tail corr
corr_all = np.corrcoef(pv_tree_te_r, pv_nn_te_r)[0,1]
tree_sorted = np.argsort(-pv_tree_te_r)
tail1 = tree_sorted[:int(len(tree_sorted)*0.01)]
corr_tail1 = np.corrcoef(pv_tree_te_r[tail1], pv_nn_te_r[tail1])[0,1]
print(f"  Overall corr={corr_all:.4f}  Top-1% tail corr={corr_tail1:.4f}", flush=True)

best_w = 0.5; best_auc = 0
for w in np.arange(0.0, 1.01, 0.02):
    pv = w*pv_tree_mv_r + (1-w)*pv_nn_mv_r
    auc = roc_auc_score(y_mv, pv)
    if auc > best_auc:
        best_auc = auc; best_w = w
print(f"  Opt blend: wT={best_w:.2f} wN={1-best_w:.2f} AUC={best_auc:.4f}", flush=True)

pv_stack_te = best_w*pv_tree_te_r + (1-best_w)*pv_nn_te_r
auc_stack_te = roc_auc_score(y_te, pv_stack_te)
print(f"  ★ STACK V3 MV={best_auc:.4f} TE={auc_stack_te:.4f}", flush=True)

# Also: equal rank agg across ALL models (LGBM+CatBoost+3NN)
pv_all_te = rank_agg([pv_lgb_te, pv_cat_te, pv_nn_te])
pv_all_mv = rank_agg([pv_lgb_mv, pv_cat_mv, pv_nn_mv])
auc_all_te = roc_auc_score(y_te, pv_all_te)
auc_all_mv = roc_auc_score(y_mv, pv_all_mv)
print(f"  ★ ALL-RANK-AVG MV={auc_all_mv:.4f} TE={auc_all_te:.4f}", flush=True)

# ============================================
# Evaluation
# ============================================
print("\n" + "="*60, flush=True)
print("Evaluation", flush=True)
print("="*60, flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def topk_eval(name, pv, y, days):
    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/days
        flag = '🏆' if pct==1.0 and acc>=62 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

for name, pv in [('LGBM', pv_lgb_te), ('CatBoost', pv_cat_te),
                 ('TREE-ENS', pv_tree_te), ('NN-ENS', pv_nn_te),
                 ('STACK', pv_stack_te), ('ALL-RANK', pv_all_te)]:
    topk_eval(name, pv, y_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/v3.npz',
         pv_lgb_te=pv_lgb_te, pv_cat_te=pv_cat_te, pv_tree_te=pv_tree_te,
         pv_nn_te=pv_nn_te, pv_stack_te=pv_stack_te, pv_all_te=pv_all_te,
         pv_tree_mv=pv_tree_mv, pv_nn_mv=pv_nn_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved v3.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"V3 DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"{'='*60}", flush=True)
