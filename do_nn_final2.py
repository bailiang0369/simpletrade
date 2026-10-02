"""纯 NN 最终版 v2: 用 v3 原始配置 (top10²+C(10,2)=45, 1.2M, float32) + 完整 rolling eval"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
print("Loading data (float32)...", flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ts_eth = eth['ts'].to_numpy().astype(np.int64)
C_eth = eth['close'].to_numpy().astype(np.float64)
ts_btc = btc['ts'].to_numpy().astype(np.int64)
C_btc = btc['close'].to_numpy().astype(np.float64)
del eth, btc; gc.collect()

idx = np.searchsorted(ts_btc, ts_eth, side="right") - 1
idx = np.clip(idx, 0, len(C_btc)-1)
B_lr1 = np.zeros(len(ts_eth), dtype=np.float32)
B_lr1[1:] = np.log(np.maximum(C_btc[idx[1:]],1e-8)/np.maximum(C_btc[idx[:-1]],1e-8)).astype(np.float32)
del ts_btc, C_btc; gc.collect()

eth_full = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth_full)
del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float32)
del feats; gc.collect()

H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int8)
del C_eth; gc.collect()

X = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
gc.collect()

FT = X.shape[1]
ts_u = ts_eth[:-H].copy()
del ts_eth; gc.collect()
print(f"FT={FT}, X mem={X.nbytes/1e9:.2f}GB", flush=True)

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts_u < tre)[0]
es_idx = np.where((ts_u >= tre) & (ts_u < es_end))[0]
mv_idx = np.where((ts_u >= es_end) & (ts_u < meta_end))[0]
te_idx = np.where(ts_u >= meta_end)[0]
ts_te_vals = ts_u[te_idx].copy()
del ts_u; gc.collect()

np.random.seed(42)
tr_samp = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
mu = X[tr_samp].mean(0)
sd = X[tr_samp].std(0) + 1e-6
del tr_samp; gc.collect()

def norm(idx):
    return np.clip((X[idx] - mu) / sd, -5, 5).astype(np.float32)

# v3 配置: top10² + C(10,2)=45, 67+10+45=122 维
TOPK = 10
print(f"FT={FT}, te={len(te_idx)}, es={len(es_idx)}, mv={len(mv_idx)}", flush=True)

# 预先计算 split 的 normalized + 加交互的版本 (省得每次 train 都重算)
print("Pre-computing eval splits...", flush=True)

X_es_norm = norm(es_idx)
X_mv_norm = norm(mv_idx)
X_te_norm = norm(te_idx)

corrs_tmp = np.array([abs(np.corrcoef(X_es_norm[:, j], label[es_idx].astype(np.float32))[0,1]) for j in range(FT)])
# 不行, top_idx 应该用 train 算. 但 norm 已经用 train stats 了
# 在 train_one 里算

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

def add_feats(X, top_idx):
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X_sq = np.clip(X[:, top_idx] ** 2, 0, 25)
    X_cross = np.concatenate([X[:, top_idx[i:i+1]] * X[:, top_idx[j:j+1]]
                              for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cross = np.clip(X_cross, -25, 25)
    return np.concatenate([X, X_sq, X_cross], axis=1)

def train_one(tr_idx_sub, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr_norm = norm(tr_idx_sub)
    y_tr = label[tr_idx_sub].astype(np.float32)
    
    corrs = np.array([abs(np.corrcoef(X_tr_norm[:, j], y_tr)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:TOPK]
    
    Xtr_f = add_feats(X_tr_norm, top_idx)
    del X_tr_norm; gc.collect()
    Xes_f = add_feats(X_es_norm, top_idx)
    Xmv_f = add_feats(X_mv_norm, top_idx)
    Xte_f = add_feats(X_te_norm, top_idx)
    gc.collect()
    
    m = ResMLP(Xtr_f.shape[1], [256, 128], 0.3)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    
    best_es = 0.0; bst = None; ni = 0
    for e in range(40):
        m.train(); idxs = np.random.permutation(len(Xtr_f))
        for i in range(0, len(idxs), 1024):
            bi = idxs[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr_f[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb*(1-0.05)+0.025)
            opt.zero_grad(); loss.backward(); opt.step()
        
        m.eval()
        with torch.no_grad():
            pv_es = torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xes_f)))).numpy()
        auc_es = roc_auc_score(label[es_idx], pv_es)
        
        if auc_es > best_es + 1e-5:
            best_es = auc_es; bst = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni = 0
        else:
            ni += 1
            if ni >= 8: break
    
    if bst: m.load_state_dict(bst)
    m.eval()
    def pred(Xf):
        pv=[]
        with torch.no_grad():
            for i in range(0,len(Xf),4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        return np.concatenate(pv)
    
    pv_te = pred(Xte_f); pv_mv = pred(Xmv_f)
    del m, Xtr_f, Xes_f, Xmv_f, Xte_f; gc.collect()
    return best_es, roc_auc_score(label[mv_idx], pv_mv), roc_auc_score(label[te_idx], pv_te), pv_te, pv_mv

# ============================================
# 运行: 1.2M + 5-seed
# ============================================
print(f"\n{'='*70}", flush=True)
print("5-seed NN ensemble (topk=10, [256,128], 1.2M)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_final = np.random.choice(tr_idx, min(1_200_000, len(tr_idx)), replace=False)
print(f"Train: {len(tr_final)}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pvs_t = []; pvs_m = []
for seed in [42, 49, 56, 63, 70]:
    t0s = time.time()
    es_a, mv_a, te_a, pv_t, pv_m = train_one(tr_final, seed)
    pvs_t.append(pv_t); pvs_m.append(pv_m)
    print(f"  s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    del pv_t, pv_m; gc.collect()

pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
auc_e_t = roc_auc_score(label[te_idx], pv_e_t); auc_e_m = roc_auc_score(label[mv_idx], pv_e_m)
print(f"\n★ ENSEMBLE: MV={auc_e_m:.4f} TE={auc_e_t:.4f}", flush=True)

# Direct top-%
print(f"\nDirect Top-%:", flush=True)
for pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
    n = int(len(pv_e_t) * pct)
    ti = np.argsort(-pv_e_t)[:n]
    acc = label[te_idx][ti].mean() * 100
    n_days = len(te_idx) / 1440
    tpd = n / n_days
    print(f"  top-{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)

# Rolling eval
print(f"\n{'='*70}", flush=True)
print("Rolling Quantile Eval (no-lookahead)", flush=True)
print(f"{'='*70}", flush=True)

DAYS = len(te_idx) / 1440

def rolling_eval(name, pv, y, ts, days, q_list=[97, 98, 99, 99.2, 99.5]):
    pv_arr = np.array(pv, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int8)
    ts_arr = np.array(ts, dtype=np.int64)
    
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
            thr = np.percentile(hist_pv, q)
            pick_mask = (ts_arr >= day_lo) & (ts_arr < day_hi) & (pv_arr >= thr)
            if pick_mask.sum() > 0:
                trades.extend(y_arr[pick_mask].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            print(f"    q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)}", flush=True)

rolling_eval("NN_5SEED_ENSEMBLE", pv_e_t, label[te_idx], ts_te_vals, DAYS)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/nn_final2.npz',
         pv_nn_te=pv_e_t, pv_nn_mv=pv_e_m,
         y_te=label[te_idx], y_mv=label[mv_idx], ts_te=ts_te_vals)
print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
