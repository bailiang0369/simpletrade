"""V-NN3: ResMLP 10-seed + 800K + top20 interactions. 高效版."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

torch.manual_seed(42); np.random.seed(42)
t0_all = time.time()

# ============================================
# Data pipeline
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
print(f"  TR={X_tr_nn.shape} TE={X_te_nn.shape} FT={FT_NN}", flush=True)

# ============================================
# ResMLP + AdamW + 10 seeds
# ============================================
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

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

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

# 10-seed ResMLP
print(f"\n{'='*60}", flush=True)
print("ResMLP [256,128] × 10 seeds", flush=True)
print(f"{'='*60}", flush=True)

pvs_t = []; pvs_m = []
seeds = [42, 49, 56, 63, 70, 115, 122, 129, 136, 143]
for i, s in enumerate(seeds):
    torch.manual_seed(s); np.random.seed(s)
    m = ResMLP(FT_NN, [256, 128], 0.3)
    np_ = sum(p.numel() for p in m.parameters())
    t0s = time.time()
    pv_t, pv_m = train_resmlp(m, X_tr_nn, y_tr, X_es_nn, y_es,
                              ep=35, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  [{i+1}/10] s={s}: MV={auc_m:.4f} TE={auc_t:.4f} p={np_:,} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_t.append(pv_t); pvs_m.append(pv_m); del m; gc.collect()

pv_resmlp_te = rank_agg(pvs_t)
pv_resmlp_mv = rank_agg(pvs_m)
auc_resmlp_te = roc_auc_score(y_te, pv_resmlp_te)
auc_resmlp_mv = roc_auc_score(y_mv, pv_resmlp_mv)
print(f"\n  ★ ResMLP-10: MV={auc_resmlp_mv:.4f} TE={auc_resmlp_te:.4f}", flush=True)

# ============================================
# Tree baseline (from V2 results, already saved)
# ============================================
print(f"\n{'='*60}", flush=True)
print("Loading V2 Tree baseline for stacking", flush=True)
print(f"{'='*60}", flush=True)

d = np.load(f'{config.PROJECT_DIR}/results/v2.npz')
pv_lgb_te = d['pv_tree_te']; pv_lgb_mv = d['pv_tree_mv']
print(f"  LGBM V2: MV={roc_auc_score(d['y_mv'], pv_lgb_mv):.4f} TE={roc_auc_score(d['y_te'], pv_lgb_te):.4f}", flush=True)
auc_lgb_te = roc_auc_score(y_te, pv_lgb_te)

# ============================================
# Stacking: Tree + ResMLP
# ============================================
print(f"\n{'='*60}", flush=True)
print("Stacking: Tree + ResMLP", flush=True)
print(f"{'='*60}", flush=True)

# Rank transform
pv_tree_mv_r = np.argsort(np.argsort(pv_lgb_mv)).astype(np.float64)/len(pv_lgb_mv)
pv_nn_mv_r = np.argsort(np.argsort(pv_resmlp_mv)).astype(np.float64)/len(pv_resmlp_mv)
pv_tree_te_r = np.argsort(np.argsort(pv_lgb_te)).astype(np.float64)/len(pv_lgb_te)
pv_nn_te_r = np.argsort(np.argsort(pv_resmlp_te)).astype(np.float64)/len(pv_resmlp_te)

# Tail correlation
corr_all = np.corrcoef(pv_tree_te_r, pv_nn_te_r)[0,1]
tree_sorted = np.argsort(-pv_tree_te_r)
tail1 = tree_sorted[:int(len(tree_sorted)*0.01)]
corr_tail1 = np.corrcoef(pv_tree_te_r[tail1], pv_nn_te_r[tail1])[0,1]
print(f"  Overall corr={corr_all:.4f}  Top-1% tail corr={corr_tail1:.4f}", flush=True)

# Opt blend
best_w = 0.5; best_auc = 0
for w in np.arange(0.0, 1.01, 0.02):
    pv = w*pv_tree_mv_r + (1-w)*pv_nn_mv_r
    auc = roc_auc_score(d['y_mv'], pv)
    if auc > best_auc:
        best_auc = auc; best_w = w
print(f"  Opt blend: wT={best_w:.2f} wN={1-best_w:.2f} AUC={best_auc:.4f}", flush=True)

pv_stack_te = best_w*pv_tree_te_r + (1-best_w)*pv_nn_te_r
auc_stack_te = roc_auc_score(y_te, pv_stack_te)
print(f"  ★ STACK MV={best_auc:.4f} TE={auc_stack_te:.4f}", flush=True)

# ============================================
# Evaluation
# ============================================
print(f"\n{'='*60}", flush=True)
print("Top-k accuracy", flush=True)
print(f"{'='*60}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def topk_eval(name, pv, y, days):
    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/days
        flag = '🏆' if pct==1.0 and acc>=62 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

topk_eval("TREE", pv_lgb_te, y_te, DAYS)
topk_eval("ResMLP-10", pv_resmlp_te, y_te, DAYS)
topk_eval("STACK", pv_stack_te, y_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/vnn3_resmlp.npz',
         pv_resmlp_te=pv_resmlp_te, pv_resmlp_mv=pv_resmlp_mv,
         pv_stack_te=pv_stack_te,
         y_te=y_te, y_mv=d['y_mv'], ts_te=ts_te)
print(f"\nSaved vnn3_resmlp.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"V-NN3 DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  Tree AUC={auc_lgb_te:.4f}", flush=True)
print(f"  ResMLP AUC={auc_resmlp_te:.4f}", flush=True)
print(f"  Stack AUC={auc_stack_te:.4f}", flush=True)
print(f"{'='*60}", flush=True)
