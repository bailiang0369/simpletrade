"""Stack v3: 修复 rolling_eval bug + 用原始概率混合(而非rank后混合) + 多策略对比"""
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

tr_idx_all = np.where(tr_mask)[0]
np.random.seed(42)
tr_idx_nn = np.random.choice(tr_idx_all, 1_200_000, replace=False)

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
print(f"FT={FT}, train_all={len(X_tr_all)}, train_nn={len(X_tr_nn)}, te={len(X_te)}", flush=True)

# z-score + clip
mu = X_tr_all.mean(0)
sd = X_tr_all.std(0) + 1e-6
for j in range(FT):
    col = X_tr_all[:,j]; col[np.isnan(col)] = 0
    m, s = col.mean(), col.std() + 1e-6
    X_tr_all[:,j] = np.clip((col - m) / s, -5, 5)
    X_tr_nn[:,j] = np.clip((np.nan_to_num(X_tr_nn[:,j], nan=0) - m) / s, -5, 5)
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()

# ============================================
# Tree: LGBM 5-seed full
# ============================================
print(f"\n{'='*60}", flush=True)
print("Tree: LGBM 5-seed full train", flush=True)
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

lgb_params = dict(objective='binary', metric='auc', learning_rate=0.03,
                  num_leaves=255, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                  verbose=-1, n_jobs=-1)

pvs_tr_t = []; pvs_tr_m = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr_all, label=y_tr_all, weight=sw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_tr_t.append(bst.predict(X_te)); pvs_tr_m.append(bst.predict(X_mv))
    print(f"  s={s}: best_iter={bst.best_iteration}", flush=True)

pv_tree_te_raw = rank_agg(pvs_tr_t)   # rank-aggregated probability
pv_tree_mv_raw = rank_agg(pvs_tr_m)
auc_tree_te = roc_auc_score(y_te, pv_tree_te_raw); auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv_raw)
print(f"  ★ TREE: MV={auc_tree_mv:.4f} TE={auc_tree_te:.4f}", flush=True)
del pvs_tr_t, pvs_tr_m; gc.collect()

# ============================================
# NN: MLP [256,128] top10² + top10两两交互
# ============================================
print(f"\n{'='*60}", flush=True)
print("NN: MLP [256,128] top10² + top10两两交互 1.2M", flush=True)
print(f"{'='*60}", flush=True)

corrs = np.array([abs(np.corrcoef(X_tr_nn[:,j], y_tr_nn)[0,1]) for j in range(FT)])
top10 = np.argsort(-corrs)[:10]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn2 = add_interactions(X_tr_nn, top10)
X_es_nn = add_interactions(X_es, top10)
X_mv_nn = add_interactions(X_mv, top10)
X_te_nn = add_interactions(X_te, top10)
del X_tr_nn, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn2.shape[1]
print(f"  FT_NN={FT_NN}", flush=True)

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

pv_nn_te_raw = rank_agg(pvs_nn_t)
pv_nn_mv_raw = rank_agg(pvs_nn_m)
auc_nn_te = roc_auc_score(y_te, pv_nn_te_raw); auc_nn_mv = roc_auc_score(y_mv, pv_nn_mv_raw)
print(f"\n  ★ NN: MV={auc_nn_mv:.4f} TE={auc_nn_te:.4f}", flush=True)
del X_tr_nn2, X_es_nn, X_mv_nn, X_te_nn, pvs_nn_t, pvs_nn_m; gc.collect()

# ============================================
# Correlation analysis
# ============================================
print(f"\n{'='*60}", flush=True)
print("Correlation Analysis (rank-based)", flush=True)
print(f"{'='*60}", flush=True)

def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)/len(a)
    rb = np.argsort(np.argsort(b)).astype(np.float64)/len(b)
    return np.corrcoef(ra, rb)[0,1]

corr_full = rank_corr(pv_tree_te_raw, pv_nn_te_raw)
print(f"  Full rank corr: {corr_full:.4f}", flush=True)

for q in [0.01, 0.02, 0.05, 0.10, 0.20]:
    n = int(len(pv_tree_te_raw) * q)
    t_mask = pv_tree_te_raw >= np.percentile(pv_tree_te_raw, 100*(1-q))
    n_mask = pv_nn_te_raw >= np.percentile(pv_nn_te_raw, 100*(1-q))
    i_mask = t_mask & n_mask
    if i_mask.sum() > 100:
        c = rank_corr(pv_tree_te_raw[i_mask], pv_nn_te_raw[i_mask])
        print(f"  top-{q:.0%} overlap ({i_mask.sum()}): corr={c:.4f}", flush=True)

for q in [0.005, 0.01, 0.02]:
    n = int(len(pv_tree_te_raw) * q)
    top_idx_t = np.argsort(-pv_tree_te_raw)[:n]
    top_idx_n = np.argsort(-pv_nn_te_raw)[:n]
    overlap = len(set(top_idx_t) & set(top_idx_n))
    print(f"  top-{q:.1%} overlap count: {overlap}/{n} ({overlap/n:.1%})", flush=True)

# ============================================
# Stacking: 用原始概率混合, 然后对混合结果算 AUC
# ============================================
print(f"\n{'='*60}", flush=True)
print("Stacking (on raw probs, then rank for eval)", flush=True)
print(f"{'='*60}", flush=True)

# 方法: 直接混合原始概率, 然后用混合结果算 AUC
# 注意: Tree 和 NN 的概率刻度不同, 但 AUC 对单调变换不敏感
# 但为了公平, 我们也试试先 rank-normalize 再混合

methods = {}

# 1. Tree only (raw prob)
methods['Tree_ONLY'] = (pv_tree_te_raw, pv_tree_mv_raw)

# 2. NN only (raw prob)
methods['NN_ONLY'] = (pv_nn_te_raw, pv_nn_mv_raw)

# 3. Opt blend on RAW PROBS (scan w on mv)
best_w_raw = 0.5; best_auc_raw = 0
for w in np.arange(0.0, 1.01, 0.01):
    pv = w*pv_tree_mv_raw + (1-w)*pv_nn_mv_raw
    auc = roc_auc_score(y_mv, pv)
    if auc > best_auc_raw:
        best_auc_raw = auc; best_w_raw = w
pv_blend_raw_te = best_w_raw*pv_tree_te_raw + (1-best_w_raw)*pv_nn_te_raw
pv_blend_raw_mv = best_w_raw*pv_tree_mv_raw + (1-best_w_raw)*pv_nn_mv_raw
methods[f'OPT_BLEND_RAW(wT={best_w_raw:.2f})'] = (pv_blend_raw_te, pv_blend_raw_mv)
print(f"  Opt blend raw: wT={best_w_raw:.2f} wN={1-best_w_raw:.2f} MV={best_auc_raw:.4f}", flush=True)

# 4. Opt blend on RANK-NORMALIZED probs
pv_tree_mv_r = np.argsort(np.argsort(pv_tree_mv_raw)).astype(np.float64)/len(pv_tree_mv_raw)
pv_nn_mv_r = np.argsort(np.argsort(pv_nn_mv_raw)).astype(np.float64)/len(pv_nn_mv_raw)
pv_tree_te_r = np.argsort(np.argsort(pv_tree_te_raw)).astype(np.float64)/len(pv_tree_te_raw)
pv_nn_te_r = np.argsort(np.argsort(pv_nn_te_raw)).astype(np.float64)/len(pv_nn_te_raw)

best_w_rank = 0.5; best_auc_rank = 0
for w in np.arange(0.0, 1.01, 0.01):
    pv = w*pv_tree_mv_r + (1-w)*pv_nn_mv_r
    auc = roc_auc_score(y_mv, pv)
    if auc > best_auc_rank:
        best_auc_rank = auc; best_w_rank = w
pv_blend_rank_te = best_w_rank*pv_tree_te_r + (1-best_w_rank)*pv_nn_te_r
pv_blend_rank_mv = best_w_rank*pv_tree_mv_r + (1-best_w_rank)*pv_nn_mv_r
methods[f'OPT_BLEND_RANK(wT={best_w_rank:.2f})'] = (pv_blend_rank_te, pv_blend_rank_mv)
print(f"  Opt blend rank: wT={best_w_rank:.2f} wN={1-best_w_rank:.2f} MV={best_auc_rank:.4f}", flush=True)

# 5. Prob avg (raw)
pv_prob_avg_te = (pv_tree_te_raw + pv_nn_te_raw) / 2
pv_prob_avg_mv = (pv_tree_mv_raw + pv_nn_mv_raw) / 2
methods['PROB_AVG'] = (pv_prob_avg_te, pv_prob_avg_mv)

# 6. Rank avg
pv_rank_avg_te = (pv_tree_te_r + pv_nn_te_r) / 2
pv_rank_avg_mv = (pv_tree_mv_r + pv_nn_mv_r) / 2
methods['RANK_AVG'] = (pv_rank_avg_te, pv_rank_avg_mv)

# 7. Product (confidence multiplicative)
pv_prod_te = pv_tree_te_raw * pv_nn_te_raw
pv_prod_mv = pv_tree_mv_raw * pv_nn_mv_raw
methods['PRODUCT'] = (pv_prod_te, pv_prod_mv)

# 8. Log-odds blending (更合理的概率混合方式)
def prob_to_logodds(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p))
def logodds_to_prob(l):
    return 1 / (1 + np.exp(-l))

lo_tree_mv = prob_to_logodds(pv_tree_mv_raw)
lo_nn_mv = prob_to_logodds(pv_nn_mv_raw)
lo_tree_te = prob_to_logodds(pv_tree_te_raw)
lo_nn_te = prob_to_logodds(pv_nn_te_raw)

best_w_lo = 0.5; best_auc_lo = 0
for w in np.arange(0.0, 1.01, 0.01):
    pv = logodds_to_prob(w*lo_tree_mv + (1-w)*lo_nn_mv)
    auc = roc_auc_score(y_mv, pv)
    if auc > best_auc_lo:
        best_auc_lo = auc; best_w_lo = w
pv_lo_te = logodds_to_prob(best_w_lo*lo_tree_te + (1-best_w_lo)*lo_nn_te)
pv_lo_mv = logodds_to_prob(best_w_lo*lo_tree_mv + (1-best_w_lo)*lo_nn_mv)
methods[f'LOGODDS(wT={best_w_lo:.2f})'] = (pv_lo_te, pv_lo_mv)
print(f"  Logodds blend: wT={best_w_lo:.2f} MV={best_auc_lo:.4f}", flush=True)

# 9. LR stacking (简单线性回归 on mv)
from sklearn.linear_model import LogisticRegression
X_stack_mv = np.column_stack([pv_tree_mv_raw, pv_nn_mv_raw])
X_stack_te = np.column_stack([pv_tree_te_raw, pv_nn_te_raw])
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_stack_mv, y_mv)
pv_lr_te = lr.predict_proba(X_stack_te)[:, 1]
pv_lr_mv = lr.predict_proba(X_stack_mv)[:, 1]
methods['LR_STACK'] = (pv_lr_te, pv_lr_mv)
print(f"  LR stack coefs: Tree={lr.coef_[0][0]:.3f} NN={lr.coef_[0][1]:.3f} inter={lr.intercept_[0]:.3f}", flush=True)

# Print all results
print(f"\n{'Method':<28} {'MV_AUC':>8} {'TE_AUC':>8} {'ΔTE':>8}", flush=True)
print("-" * 56, flush=True)
base = roc_auc_score(y_te, pv_tree_te_raw)
for name, (pte, pmv) in sorted(methods.items(), key=lambda x: -roc_auc_score(y_te, x[1][0])):
    auc_mv = roc_auc_score(y_mv, pmv)
    auc_te = roc_auc_score(y_te, pte)
    delta = auc_te - base
    marker = " ★" if auc_te > base + 0.001 else ""
    print(f"{name:<28} {auc_mv:>8.4f} {auc_te:>8.4f} {delta:>+8.4f}{marker}", flush=True)

# ============================================
# Rolling quantile eval (修复 bug)
# ============================================
print(f"\n{'='*60}", flush=True)
print("Rolling Quantile No-Lookahead Eval", flush=True)
print(f"{'='*60}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def rolling_eval(name, pv, y, ts, days, q_list=[97, 98, 99, 99.2, 99.5]):
    pv_arr = np.array(pv, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int64)
    ts_arr = np.array(ts, dtype=np.int64)
    n = len(pv_arr)
    assert n == len(y_arr) == len(ts_arr), f"shape mismatch: {n} vs {len(y_arr)} vs {len(ts_arr)}"
    
    day_sec = 86400
    day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    window = 30
    
    print(f"\n  [{name}] AUC={roc_auc_score(y_arr, pv_arr):.4f}", flush=True)
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
            pick_mask = (ts_arr >= day_lo) & (ts_arr < day_hi) & (pv_arr >= thr)
            if pick_mask.sum() > 0:
                trades.extend(y_arr[pick_mask].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            print(f"    q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)}", flush=True)

# 选最好的几个方法做 rolling eval
top_methods = sorted(methods.items(), key=lambda x: -roc_auc_score(y_te, x[1][0]))[:6]
for name, (pte, pmv) in top_methods:
    rolling_eval(name, pte, y_te, ts_te, DAYS)

# Save
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/stack_v3.npz',
         pv_tree_te_raw=pv_tree_te_raw, pv_tree_mv_raw=pv_tree_mv_raw,
         pv_nn_te_raw=pv_nn_te_raw, pv_nn_mv_raw=pv_nn_mv_raw,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te,
         best_w_raw=best_w_raw, best_w_rank=best_w_rank, best_w_lo=best_w_lo)

print(f"\n{'='*60}", flush=True)
print(f"DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  Tree TE AUC={auc_tree_te:.4f}", flush=True)
print(f"  NN   TE AUC={auc_nn_te:.4f}", flush=True)
print(f"  Corr full={corr_full:.4f}, top1%={rank_corr(pv_tree_te_raw[pv_tree_te_raw>=np.percentile(pv_tree_te_raw,99)], pv_nn_te_raw[pv_nn_te_raw>=np.percentile(pv_nn_te_raw,99)]):.4f}", flush=True)
print(f"{'='*60}", flush=True)
