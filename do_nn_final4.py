"""纯 NN: 最终对比 ResMLP+交互 vs DCN"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()

# Data
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ts_eth = eth['ts'].to_numpy().astype(np.int64)
C_eth = eth['close'].to_numpy().astype(np.float64)
ts_btc = btc['ts'].to_numpy().astype(np.int64)
C_btc = btc['close'].to_numpy().astype(np.float64)
del eth, btc; gc.collect()
idx = np.searchsorted(ts_btc, ts_eth, side="right") - 1
idx = np.clip(idx, 0, len(C_btc)-1)
B_lr1 = np.zeros(len(ts_eth), dtype=np.float16)
B_lr1[1:] = np.log(np.maximum(C_btc[idx[1:]],1e-8)/np.maximum(C_btc[idx[:-1]],1e-8)).astype(np.float16)
del ts_btc, C_btc; gc.collect()

eth_full = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth_full); del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float16); del feats; gc.collect()

H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int8); del C_eth; gc.collect()

X = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)

FT = X.shape[1]
ts_u = ts_eth[:-H].copy(); del ts_eth; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts_u < tre)[0]
es_idx = np.where((ts_u >= tre) & (ts_u < es_end))[0]
mv_idx = np.where((ts_u >= es_end) & (ts_u < meta_end))[0]
te_idx = np.where(ts_u >= meta_end)[0]
ts_te_vals = ts_u[te_idx].copy(); del ts_u; gc.collect()

np.random.seed(42)
tr_samp = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
mu = X[tr_samp].astype(np.float32).mean(0)
sd = X[tr_samp].astype(np.float32).std(0) + 1e-6
del tr_samp; gc.collect()

def norm(idx):
    return np.clip((X[idx].astype(np.float32) - mu) / sd, -5, 5).astype(np.float32)

print(f"FT={FT}, te={len(te_idx)}", flush=True)

# 准备 splits + 加交互
Xes_b = norm(es_idx); Xmv_b = norm(mv_idx); Xte_b = norm(te_idx)

np.random.seed(42)
tr_for_corr = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
X_tr_c = norm(tr_for_corr)
y_tr_c = label[tr_for_corr].astype(np.float32)
corrs = np.array([abs(np.corrcoef(X_tr_c[:, j], y_tr_c)[0,1]) for j in range(FT)])
top10 = np.argsort(-corrs)[:10]
del X_tr_c, y_tr_c; gc.collect()

def addint(X):
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X_sq = np.clip(X[:, top10] ** 2, 0, 25)
    X_cross = np.concatenate([X[:, top10[i:i+1]] * X[:, top10[j:j+1]]
                              for i in range(len(top10)) for j in range(i+1, len(top10))], axis=1)
    X_cross = np.clip(X_cross, -25, 25)
    return np.concatenate([X, X_sq, X_cross], axis=1)

Xes_i = addint(Xes_b.copy()); Xmv_i = addint(Xmv_b.copy()); Xte_i = addint(Xte_b.copy())
FT_INTER = Xes_i.shape[1]

# Models
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x0, x):
        return x0 * (x @ self.w).unsqueeze(1) + self.b + x

class DCN(nn.Module):
    def __init__(self, ft, cross=3, deep=[256,128], drop=0.3):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(ft) for _ in range(cross)])
        self.deep = nn.Sequential()
        prev = ft
        for h in deep:
            self.deep.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        self.head = nn.Linear(ft + prev, 1)
    def forward(self, x0):
        x_cross = x0
        for cl in self.cross_layers: x_cross = cl(x0, x_cross)
        x_deep = self.deep(x0)
        return self.head(torch.cat([x_cross, x_deep], dim=1)).squeeze(-1)

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
        for b in self.blocks: x = x + b(x) if x.shape == b(x).shape else b(x)
        return self.head(x).squeeze(-1)

def train_eval(tr_idx_sub, seed, m, Xes_f, Xmv_f, Xte_f):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = norm(tr_idx_sub)
    y_tr = label[tr_idx_sub].astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    best_es = 0.0; bst = None; ni = 0
    for e in range(40):
        m.train(); idxs = np.random.permutation(len(X_tr))
        for i in range(0, len(idxs), 1024):
            bi = idxs[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb*0.95+0.025)
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
    del m, X_tr; gc.collect()
    return best_es, roc_auc_score(label[mv_idx], pv_mv), roc_auc_score(label[te_idx], pv_te), pv_te, pv_mv

# ============================================
# PHASE 1: 3 模型对比 (800K, 单 seed)
# ============================================
print(f"\n{'='*70}", flush=True)
print("PHASE 1: 3 模型对比 (800K, 单 seed)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)
n1 = int(len(te_idx) * 0.01)

configs = [
    ("ResMLP_inter", ResMLP(FT_INTER, [256,128], 0.3), Xes_i.copy(), Xmv_i.copy(), Xte_i.copy()),
    ("DCN_3cross",   DCN(FT, 3, [256,128], 0.3),       Xes_b.copy(), Xmv_b.copy(), Xte_b.copy()),
    ("DCN_5cross",   DCN(FT, 5, [256,128], 0.3),       norm(es_idx), norm(mv_idx), norm(te_idx)),
]

for mn, m, Xes_f, Xmv_f, Xte_f in configs:
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()
    es_a, mv_a, te_a, pv_t, pv_m = train_eval(tr_sub, 42, m, Xes_f, Xmv_f, Xte_f)
    elapsed = time.time() - t0
    t1 = np.argsort(-pv_t)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    print(f"  {mn:<15} TE={te_a:.4f} MV={mv_a:.4f} top1={top1:.1f}% | {elapsed:.0f}s", flush=True)
    del pv_t, pv_m, Xes_f, Xmv_f, Xte_f; gc.collect()

# ============================================
# PHASE 2: Top2 × 1.2M × 5-seed
# ============================================
print(f"\n{'='*70}", flush=True)
print("PHASE 2: Top2 × 1.2M × 5-seed ensemble", flush=True)
print(f"{'='*70}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

np.random.seed(42)
tr_final = np.random.choice(tr_idx, min(1_200_000, len(tr_idx)), replace=False)

for mn, make_m, Xes_f, Xmv_f, Xte_f in configs[:2]:
    print(f"\n--- {mn} ---", flush=True)
    pvs_t = []; pvs_m = []
    for seed in [42, 49, 56, 63, 70]:
        m = make_m if callable(make_m) and not isinstance(make_m, nn.Module) else None
        # make_m 已经是实例, 不能复用
        if mn == 'ResMLP_inter':
            m = ResMLP(FT_INTER, [256,128], 0.3)
        elif mn == 'DCN_3cross':
            m = DCN(FT, 3, [256,128], 0.3)
        t0s = time.time()
        es_a, mv_a, te_a, pv_t, pv_m = train_eval(tr_final, seed, m, Xes_f.copy(), Xmv_f.copy(), Xte_f.copy())
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-t0s:.0f}s]", flush=True)
        del pv_t, pv_m; gc.collect()
    
    pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
    auc_te_f = roc_auc_score(label[te_idx], pv_e_t)
    auc_mv_f = roc_auc_score(label[mv_idx], pv_e_m)
    print(f"\n    ★ ENSEMBLE: MV={auc_mv_f:.4f} TE={auc_te_f:.4f}", flush=True)
    for pct in [0.005, 0.01, 0.015]:
        n = int(len(pv_e_t) * pct)
        ti = np.argsort(-pv_e_t)[:n]
        acc = label[te_idx][ti].mean() * 100
        n_days = len(te_idx) / 1440
        print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={n/n_days:.1f}", flush=True)
    
    # Rolling q=99
    DAYS = len(te_idx) / 1440
    pv_arr = np.array(pv_e_t, dtype=np.float64)
    day_start = ts_te_vals.min()
    all_ts = np.arange(day_start, ts_te_vals.max() + 86400, 86400)
    n_days = len(all_ts) - 1
    for q in [99, 99.5]:
        trades = []
        for d in range(30, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - 30]; hist_hi = day_lo
            hist_mask = (ts_te_vals >= hist_lo) & (ts_te_vals < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100: continue
            thr = np.percentile(hist_pv, q)
            pick_mask = (ts_te_vals >= day_lo) & (ts_te_vals < day_hi) & (pv_arr >= thr)
            if pick_mask.sum() > 0:
                trades.extend(label[te_idx][pick_mask].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / DAYS
            print(f"      q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)}", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
