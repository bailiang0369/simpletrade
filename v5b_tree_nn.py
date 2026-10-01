"""V5b: 用 V5 Tree-ENS (全量, best AUC=0.5430) + V4 ResMLP (800K) + Stack + rolling quantile eval."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import datetime as dtm, config

torch.manual_seed(42); np.random.seed(42)
t0_all = time.time()

# ============================================
# Data
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

# Tree: 全量, NN: 800K
tr_idx_all = np.where(tr_mask)[0]
np.random.seed(42)
tr_idx_nn = np.random.choice(tr_idx_all, 800_000, replace=False)

X_tr_all = X_all[tr_idx_all].copy()
X_tr_nn = X_all[tr_idx_nn].copy()
X_es = X_all[es_mask].copy()
X_mv = X_all[mv_mask].copy()
X_te = X_all[te_mask].copy()
y_tr_all = label[tr_idx_all]; r_tr_all = ret_future[tr_idx_all]
y_tr_nn = label[tr_idx_nn]; r_tr_nn = ret_future[tr_idx_nn]
y_es = label[es_mask]; y_mv = label[mv_mask]; y_te = label[te_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

FT = X_tr_all.shape[1]
# z-score using full train stats (NN subsample will use same stats)
mu = X_tr_all.mean(0)
sd = X_tr_all.std(0) + 1e-6
for j in range(FT):
    col = X_tr_all[:,j]; col[np.isnan(col)] = 0
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    m, s = col.mean(), col.std() + 1e-6
    X_tr_all[:,j] = np.clip((col - m) / s, -5, 5)
    X_tr_nn[:,j] = np.clip((np.nan_to_num(X_tr_nn[:,j], nan=0) - m) / s, -5, 5)
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()

# ============================================
# Tree: LGBM 5-seed full + CatBoost 5-seed full
# ============================================
print(f"\n{'='*60}", flush=True)
print("Tree-ENS: LGBM+CatBoost full train", flush=True)
print(f"{'='*60}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pos = y_tr_all.mean()
pw = np.where(y_tr_all>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr_all)*200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr_all)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)

# LGBM
lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_lgb_t = []; pvs_lgb_m = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr_all, label=y_tr_all, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(1000)])
    pvs_lgb_t.append(bst.predict(X_te)); pvs_lgb_m.append(bst.predict(X_mv))

pv_lgb_te = rank_agg(pvs_lgb_t); pv_lgb_mv = rank_agg(pvs_lgb_m)
print(f"  LGBM: MV={roc_auc_score(y_mv, pv_lgb_mv):.4f} TE={roc_auc_score(y_te, pv_lgb_te):.4f}", flush=True)
del pvs_lgb_t, pvs_lgb_m, bst; gc.collect()

# CatBoost
pvs_cat_t = []; pvs_cat_m = []
for s in [42, 49, 56, 63, 70]:
    cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3.0, random_seed=s, verbose=0,
                            early_stopping_rounds=200, eval_metric='AUC',
                            bagging_temperature=1.0, random_strength=0.5)
    t0s = time.time()
    cb.fit(X_tr_all, y_tr_all, eval_set=[(X_es, y_es)], sample_weight=sw, use_best_model=True)
    pvs_cat_t.append(cb.predict_proba(X_te)[:,1])
    pvs_cat_m.append(cb.predict_proba(X_mv)[:,1])
    print(f"  CatBoost s={s}: [{time.time()-t0s:.0f}s]", flush=True)
    del cb; gc.collect()

pv_cat_te = rank_agg(pvs_cat_t); pv_cat_mv = rank_agg(pvs_cat_m)
print(f"  CatBoost: MV={roc_auc_score(y_mv, pv_cat_mv):.4f} TE={roc_auc_score(y_te, pv_cat_te):.4f}", flush=True)
del pvs_cat_t, pvs_cat_m; gc.collect()

pv_tree_te = rank_agg([pv_lgb_te, pv_cat_te])
pv_tree_mv = rank_agg([pv_lgb_mv, pv_cat_mv])
auc_tree_te = roc_auc_score(y_te, pv_tree_te); auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv)
print(f"  ★ TREE-ENS: MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)

# Free full train data before NN
del X_tr_all, y_tr_all, r_tr_all, sw; gc.collect()

# ============================================
# NN: ResMLP top20 800K
# ============================================
print(f"\n{'='*60}", flush=True)
print("NN: ResMLP [256,128] top20 int 800K", flush=True)
print(f"{'='*60}", flush=True)

corrs = np.array([abs(np.corrcoef(X_tr_nn[:,j], y_tr_nn)[0,1]) for j in range(FT)])
top20 = np.argsort(-corrs)[:20]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn2 = add_interactions(X_tr_nn, top20)
X_es_nn = add_interactions(X_es, top20)
X_mv_nn = add_interactions(X_mv, top20)
X_te_nn = add_interactions(X_te, top20)
del X_tr_nn, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn2.shape[1]
print(f"  FT={FT_NN}", flush=True)

class ResMLP(nn.Module):
    def __init__(self, ft, hs, drop=0.3):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs)-1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hs[i], hs[i+1]), nn.BatchNorm1d(hs[i+1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x) if x.shape == b(x).shape else b(x)
        return self.head(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_resmlp(m, Xtr, ytr, Xes, yes, ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
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
        auc_es = roc_auc_score(yes, evaluate(m, Xes))
        if auc_es > best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return evaluate(m, X_te_nn), evaluate(m, X_mv_nn)

pvs_nn_t = []; pvs_nn_m = []
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = ResMLP(FT_NN, [256, 128], 0.3)
    t0s = time.time()
    pv_t, pv_m = train_resmlp(m, X_tr_nn2, y_tr_nn, X_es_nn, y_es,
                              ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  s={s}: MV={auc_m:.4f} TE={auc_t:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_nn_t.append(pv_t); pvs_nn_m.append(pv_m); del m; gc.collect()

pv_nn_te = rank_agg(pvs_nn_t); pv_nn_mv = rank_agg(pvs_nn_m)
auc_nn_te = roc_auc_score(y_te, pv_nn_te); auc_nn_mv = roc_auc_score(y_mv, pv_nn_mv)
print(f"\n  ★ NN: MV={auc_nn_mv:.4f} TE={auc_nn_te:.4f}", flush=True)
del X_tr_nn2, X_es_nn, X_mv_nn, X_te_nn, pvs_nn_t, pvs_nn_m; gc.collect()

# ============================================
# Stack
# ============================================
print(f"\n{'='*60}", flush=True)
print("Stack + Rolling Quantile Eval", flush=True)
print(f"{'='*60}", flush=True)

pv_tree_mv_r = np.argsort(np.argsort(pv_tree_mv)).astype(np.float64)/len(pv_tree_mv)
pv_nn_mv_r = np.argsort(np.argsort(pv_nn_mv)).astype(np.float64)/len(pv_nn_mv)
pv_tree_te_r = np.argsort(np.argsort(pv_tree_te)).astype(np.float64)/len(pv_tree_te)
pv_nn_te_r = np.argsort(np.argsort(pv_nn_te)).astype(np.float64)/len(pv_nn_te)

best_w = 0.5; best_auc = 0
for w in np.arange(0.0, 1.01, 0.02):
    pv = w*pv_tree_mv_r + (1-w)*pv_nn_mv_r
    auc = roc_auc_score(y_mv, pv)
    if auc > best_auc:
        best_auc = auc; best_w = w
print(f"  Opt blend: wT={best_w:.2f} wN={1-best_w:.2f} AUC={best_auc:.4f}", flush=True)

pv_stack_te = best_w*pv_tree_te_r + (1-best_w)*pv_nn_te_r
auc_stack_te = roc_auc_score(y_te, pv_stack_te)
print(f"  ★ STACK MV={best_auc:.4f} TE={auc_stack_te:.4f}", flush=True)

# ============================================
# Rolling quantile no-lookahead
# ============================================
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def rolling_eval(name, pv, y, ts, days, q_list=[97, 98, 99, 99.3, 99.5]):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400
    day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    window = 30
    
    print(f"\n  [{name}] AUC={roc_auc_score(y, pv):.4f}", flush=True)
    for q in q_list:
        trades = []
        for d in range(window, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - window]; hist_hi = day_lo
            hist_mask = (ts_arr >= hist_lo) & (ts_arr < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100: continue
            today_mask = (ts_arr >= day_lo) & (ts_arr < day_hi)
            thr = np.percentile(hist_pv, q)
            pick = pv_arr[today_mask] >= thr
            if pick.sum() > 0:
                trades.extend(y_arr[pick].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            print(f"    q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)}", flush=True)

for name, pv in [('LGBM', pv_lgb_te), ('CatBoost', pv_cat_te),
                 ('TREE-ENS', pv_tree_te), ('NN', pv_nn_te), ('STACK', pv_stack_te)]:
    rolling_eval(name, pv, y_te, ts_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/v5b.npz',
         pv_lgb_te=pv_lgb_te, pv_cat_te=pv_cat_te,
         pv_tree_te=pv_tree_te, pv_nn_te=pv_nn_te, pv_stack_te=pv_stack_te,
         pv_tree_mv=pv_tree_mv, pv_nn_mv=pv_nn_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved v5b.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"V5b DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  Tree-ENS AUC={auc_tree_te:.4f}", flush=True)
print(f"  NN   AUC={auc_nn_te:.4f}", flush=True)
print(f"  Stk  AUC={auc_stack_te:.4f}", flush=True)
print(f"{'='*60}", flush=True)
