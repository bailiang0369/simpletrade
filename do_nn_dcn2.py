"""纯 NN: DCN vs ResMLP - float16 省内存版"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
print("Loading data (float16)...", flush=True)

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
feats = fe.build_features(eth_full)
del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float16)
del feats; gc.collect()

H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int8)
del C_eth; gc.collect()

X = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
gc.collect()

FT = X.shape[1]
ts_u = ts_eth[:-H].copy()
del ts_eth; gc.collect()

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
mu = X[tr_samp].astype(np.float32).mean(0)
sd = X[tr_samp].astype(np.float32).std(0) + 1e-6
del tr_samp; gc.collect()

def norm(idx):
    return np.clip((X[idx].astype(np.float32) - mu) / sd, -5, 5).astype(np.float32)

print(f"FT={FT}, te={len(te_idx)}, X={X.nbytes/1e9:.2f}GB", flush=True)

# ============================================
# Models
# ============================================
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x0, x):
        return x0 * (x @ self.w).unsqueeze(1) + self.b + x

class DCN(nn.Module):
    def __init__(self, ft, cross_layers=3, deep_hs=[256, 128], drop=0.3):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(ft) for _ in range(cross_layers)])
        self.deep = nn.Sequential()
        prev = ft
        for h in deep_hs:
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

def train_and_eval(tr_idx_sub, seed, model_fn, Xes_f, Xmv_f, Xte_f):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = norm(tr_idx_sub)
    y_tr = label[tr_idx_sub].astype(np.float32)
    
    m = model_fn(X_tr.shape[1])
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
# 逐个模型跑
# ============================================
print(f"\n{'='*70}", flush=True)
print("Model comparison (800K, single seed)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)

# ResMLP base (无交互)
print("\n--- ResMLP base ---", flush=True)
Xes_b = norm(es_idx); Xmv_b = norm(mv_idx); Xte_b = norm(te_idx)
es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_sub, 42, 
    lambda ft: ResMLP(ft, [256,128], 0.3), Xes_b, Xmv_b, Xte_b)
print(f"  TE={te_a:.4f} MV={mv_a:.4f}", flush=True)
n1 = int(len(pv_t) * 0.01)
t1 = np.argsort(-pv_t)[:n1]
print(f"  top1%={label[te_idx][t1].mean()*100:.1f}%", flush=True)
del Xes_b, Xmv_b, Xte_b, pv_t, pv_m; gc.collect()

# ResMLP + 手工交互 (v3 配置)
print("\n--- ResMLP + top10²+C(10,2) ---", flush=True)
X_tr_corr = norm(tr_sub)
y_tr_corr = label[tr_sub].astype(np.float32)
corrs = np.array([abs(np.corrcoef(X_tr_corr[:, j], y_tr_corr)[0,1]) for j in range(FT)])
top10 = np.argsort(-corrs)[:10]
del X_tr_corr, y_tr_corr; gc.collect()

def addint(normed_X, t=top10):
    X = np.nan_to_num(normed_X, nan=0.0).astype(np.float32)
    X_sq = np.clip(X[:, t] ** 2, 0, 25)
    X_cross = np.concatenate([X[:, t[i:i+1]] * X[:, t[j:j+1]]
                              for i in range(len(t)) for j in range(i+1, len(t))], axis=1)
    X_cross = np.clip(X_cross, -25, 25)
    return np.concatenate([X, X_sq, X_cross], axis=1)

Xes_i = addint(norm(es_idx)); Xmv_i = addint(norm(mv_idx)); Xte_i = addint(norm(te_idx))
es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_sub, 42,
    lambda ft: ResMLP(ft, [256,128], 0.3), Xes_i, Xmv_i, Xte_i)
print(f"  TE={te_a:.4f} MV={mv_a:.4f}", flush=True)
t1 = np.argsort(-pv_t)[:n1]
print(f"  top1%={label[te_idx][t1].mean()*100:.1f}%", flush=True)
del Xes_i, Xmv_i, Xte_i, pv_t, pv_m; gc.collect()

# DCN
print("\n--- DCN (3 cross, [256,128] deep) ---", flush=True)
Xes_b = norm(es_idx); Xmv_b = norm(mv_idx); Xte_b = norm(te_idx)
es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_sub, 42,
    lambda ft: DCN(ft, cross_layers=3, deep_hs=[256,128], drop=0.3), Xes_b, Xmv_b, Xte_b)
print(f"  TE={te_a:.4f} MV={mv_a:.4f}", flush=True)
t1 = np.argsort(-pv_t)[:n1]
print(f"  top1%={label[te_idx][t1].mean()*100:.1f}%", flush=True)
del Xes_b, Xmv_b, Xte_b, pv_t, pv_m; gc.collect()

# DCN deeper
print("\n--- DCN (3 cross, [512,256,128] deep) ---", flush=True)
Xes_b = norm(es_idx); Xmv_b = norm(mv_idx); Xte_b = norm(te_idx)
es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_sub, 42,
    lambda ft: DCN(ft, cross_layers=3, deep_hs=[512,256,128], drop=0.3), Xes_b, Xmv_b, Xte_b)
print(f"  TE={te_a:.4f} MV={mv_a:.4f}", flush=True)
t1 = np.argsort(-pv_t)[:n1]
print(f"  top1%={label[te_idx][t1].mean()*100:.1f}%", flush=True)
del Xes_b, Xmv_b, Xte_b, pv_t, pv_m; gc.collect()

# ============================================
# 最好的 2 个 × 1.2M × 5-seed
# ============================================
print(f"\n{'='*70}", flush=True)
print("Best 2 models × 1.2M × 5-seed ensemble", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_final = np.random.choice(tr_idx, min(1_200_000, len(tr_idx)), replace=False)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

# ResMLP + 交互 (已知最好)
print("\n--- ResMLP + top10²+C(10,2) ---", flush=True)
pvs_t = []; pvs_m = []
for seed in [42, 49, 56, 63, 70]:
    Xes_i = addint(norm(es_idx)); Xmv_i = addint(norm(mv_idx)); Xte_i = addint(norm(te_idx))
    es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_final, seed,
        lambda ft: ResMLP(ft, [256,128], 0.3), Xes_i, Xmv_i, Xte_i)
    pvs_t.append(pv_t); pvs_m.append(pv_m)
    print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f}", flush=True)
    del Xes_i, Xmv_i, Xte_i; gc.collect()

pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
print(f"\n    ★ ENSEMBLE: MV={roc_auc_score(label[mv_idx], pv_e_m):.4f} TE={roc_auc_score(label[te_idx], pv_e_t):.4f}", flush=True)
for pct in [0.005, 0.01, 0.015]:
    n = int(len(pv_e_t) * pct)
    ti = np.argsort(-pv_e_t)[:n]
    acc = label[te_idx][ti].mean() * 100
    n_days = len(te_idx) / 1440
    print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={n/n_days:.1f}", flush=True)

# DCN
print("\n--- DCN (3 cross, [256,128]) ---", flush=True)
pvs_t = []; pvs_m = []
for seed in [42, 49, 56, 63, 70]:
    Xes_b = norm(es_idx); Xmv_b = norm(mv_idx); Xte_b = norm(te_idx)
    es_a, mv_a, te_a, pv_t, pv_m = train_and_eval(tr_final, seed,
        lambda ft: DCN(ft, cross_layers=3, deep_hs=[256,128], drop=0.3), Xes_b, Xmv_b, Xte_b)
    pvs_t.append(pv_t); pvs_m.append(pv_m)
    print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f}", flush=True)
    del Xes_b, Xmv_b, Xte_b; gc.collect()

pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
print(f"\n    ★ ENSEMBLE: MV={roc_auc_score(label[mv_idx], pv_e_m):.4f} TE={roc_auc_score(label[te_idx], pv_e_t):.4f}", flush=True)
for pct in [0.005, 0.01, 0.015]:
    n = int(len(pv_e_t) * pct)
    ti = np.argsort(-pv_e_t)[:n]
    acc = label[te_idx][ti].mean() * 100
    n_days = len(te_idx) / 1440
    print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={n/n_days:.1f}", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
