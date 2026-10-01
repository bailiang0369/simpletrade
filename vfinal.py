"""V-FINAL: Tree+ResMLP Stack + rolling quantile no-lookahead eval.
800K train (fits memory). 真正的无前视评估.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
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
print(f"  TR={X_tr.shape} TE={X_te.shape}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

# ============================================
# Tree: LGBM 10-seed lr=0.03 metric=auc
# ============================================
print(f"\n{'='*60}", flush=True)
print("Tree: LGBM 10-seed lr=0.03 metric=auc", flush=True)
print(f"{'='*60}", flush=True)

pos = y_tr.mean()
pw = np.where(y_tr>0.5, (1-pos)/pos, pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10); hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, pw * rw * 0.3, pw * rw).astype(np.float32)

lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_tree_t = []; pvs_tree_m = []
for s in [42, 49, 56, 63, 70, 115, 122, 129, 136, 143]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(2000)])
    pvs_tree_t.append(bst.predict(X_te)); pvs_tree_m.append(bst.predict(X_mv))

pv_tree_te = rank_agg(pvs_tree_t); pv_tree_mv = rank_agg(pvs_tree_m)
print(f"  ★ TREE: MV={roc_auc_score(y_mv, pv_tree_mv):.4f} TE={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)
del pvs_tree_t, pvs_tree_m; gc.collect()

# ============================================
# NN: ResMLP top20 5-seed
# ============================================
print(f"\n{'='*60}", flush=True)
print("NN: ResMLP [256,128] top20 int 5-seed", flush=True)
print(f"{'='*60}", flush=True)

corrs = np.array([abs(np.corrcoef(X_tr[:,j], y_tr)[0,1]) for j in range(FT)])
top20 = np.argsort(-corrs)[:20]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn = add_interactions(X_tr, top20)
X_es_nn = add_interactions(X_es, top20)
X_mv_nn = add_interactions(X_mv, top20)
X_te_nn = add_interactions(X_te, top20)
del X_tr, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn.shape[1]
print(f"  FT: {FT} → {FT_NN}", flush=True)

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
    pv_t, pv_m = train_resmlp(m, X_tr_nn, y_tr, X_es_nn, y_es,
                              ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  s={s}: MV={auc_m:.4f} TE={auc_t:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_nn_t.append(pv_t); pvs_nn_m.append(pv_m); del m; gc.collect()

pv_nn_te = rank_agg(pvs_nn_t); pv_nn_mv = rank_agg(pvs_nn_m)
print(f"  ★ NN: MV={roc_auc_score(y_mv, pv_nn_mv):.4f} TE={roc_auc_score(y_te, pv_nn_te):.4f}", flush=True)
del X_tr_nn, X_es_nn, X_mv_nn, X_te_nn, pvs_nn_t, pvs_nn_m; gc.collect()

# ============================================
# Stack
# ============================================
print(f"\n{'='*60}", flush=True)
print("Stack: Tree + NN", flush=True)
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
# Rolling quantile no-lookahead eval
# ============================================
print(f"\n{'='*60}", flush=True)
print("Rolling quantile no-lookahead (30d window)", flush=True)
print(f"{'='*60}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def rolling_eval(name, pv, y, ts, days, q_list=[97, 98, 99, 99.3, 99.5, 99.7]):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400
    all_ts = np.arange(ts_arr.min(), ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    window = 30
    
    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
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
            flag = '🏆' if acc>=65 else ('✅' if acc>=62 else '')
            print(f"    q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)} {flag}", flush=True)

for name, pv in [('TREE', pv_tree_te), ('NN', pv_nn_te), ('STACK', pv_stack_te)]:
    rolling_eval(name, pv, y_te, ts_te, DAYS)

# Also: 等权 rank avg
pv_equal = rank_agg([pv_tree_te, pv_nn_te])
rolling_eval("EQUAL-RANK", pv_equal, y_te, ts_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/vfinal.npz',
         pv_tree_te=pv_tree_te, pv_nn_te=pv_nn_te, pv_stack_te=pv_stack_te,
         pv_tree_mv=pv_tree_mv, pv_nn_mv=pv_nn_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved vfinal.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"V-FINAL DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  Tree AUC={roc_auc_score(y_te, pv_tree_te):.4f}", flush=True)
print(f"  NN   AUC={roc_auc_score(y_te, pv_nn_te):.4f}", flush=True)
print(f"  Stk  AUC={auc_stack_te:.4f}", flush=True)
print(f"{'='*60}", flush=True)
