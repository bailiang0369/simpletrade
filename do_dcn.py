"""
DCN (Deep & Cross Network) for ETH H=15
显式 cross layer + deep network, 和 MLP 有不同归纳偏置
如果 tail 相关性低, ensemble 能提升
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
print("Loading...", flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')

ts_e = eth['ts'].to_numpy()
C_e = eth['close'].to_numpy()
ts_b = btc['ts'].to_numpy()
C_b = btc['close'].to_numpy()
del eth, btc; gc.collect()

idx_b = np.searchsorted(ts_b, ts_e, side="right") - 1
idx_b = np.clip(idx_b, 0, len(C_b)-1)
B_lr1 = np.zeros(len(ts_e), dtype=np.float32)
B_lr1[1:] = np.log(np.maximum(C_b[idx_b[1:]],1e-8)/np.maximum(C_b[idx_b[:-1]],1e-8))
del ts_b, C_b; gc.collect()

eth_full = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth_full); del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float32); del feats; gc.collect()

H = 15
label = (C_e[H:] > C_e[:-H]).astype(np.int8); del C_e; gc.collect()

X_RAW = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X_RAW = np.nan_to_num(X_RAW, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()

FT = X_RAW.shape[1]
ts_u = ts_e[:-H].copy(); del ts_e; gc.collect()
print(f"X_RAW={X_RAW.shape}", flush=True)

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts_u < tre)[0]
es_idx = np.where((ts_u >= tre) & (ts_u < es_end))[0]
mv_idx = np.where((ts_u >= es_end) & (ts_u < meta_end))[0]
te_idx = np.where(ts_u >= meta_end)[0]
ts_te_vals = ts_u[te_idx].copy(); del ts_u; gc.collect()

np.random.seed(42)
samp = np.random.choice(tr_idx, 300_000, replace=False)
MU = X_RAW[samp].mean(0); SD = X_RAW[samp].std(0) + 1e-6
del samp; gc.collect()

def norm(arr): return np.clip((arr - MU) / SD, -5, 5)

def make_inter(Xn, top_idx):
    X_sq = np.clip(Xn[:, top_idx] ** 2, 0, 25)
    X_cr = np.concatenate([Xn[:, top_idx[i:i+1]] * Xn[:, top_idx[j:j+1]]
                           for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cr = np.clip(X_cr, -25, 25)
    return np.concatenate([Xn, X_sq, X_cr], axis=1)

# DCN
class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d) * 0.01)
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x0, x):
        return x0 * (x @ self.w).unsqueeze(1) + self.b + x

class DCN(nn.Module):
    def __init__(self, ft, cross_n=3, deep=[256,128], drop=0.3):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(ft) for _ in range(cross_n)])
        deep_layers = []; prev = ft
        for h in deep:
            deep_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(ft + prev, 1)
    def forward(self, x0):
        x_c = x0
        for cl in self.cross: x_c = cl(x0, x_c)
        x_d = self.deep(x0)
        return self.head(torch.cat([x_c, x_d], dim=1)).squeeze(-1)

def train_one(tr_sub, seed, cross_n, deep, drop, lr, wd, topk, smooth, epochs, pat, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    
    np.random.seed(seed)
    cs = np.random.choice(tr_sub, min(300_000, len(tr_sub)), replace=False)
    X_cs = norm(X_RAW[cs])
    y_cs = label[cs].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_cs[:, j], y_cs)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:topk]
    del X_cs, y_cs, corrs; gc.collect()
    
    print(f"    build train ({len(tr_sub)})...", flush=True)
    X_tr = make_inter(norm(X_RAW[tr_sub]), top_idx)
    y_tr = label[tr_sub].astype(np.float32)
    print(f"    X_tr={X_tr.shape}, mem={X_tr.nbytes/1e9:.2f}GB", flush=True)
    
    X_es = make_inter(norm(X_RAW[es_idx]), top_idx)
    
    m = DCN(X_tr.shape[1], cross_n, deep, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_es = 0.0; bst = None; ni = 0
    for e in range(epochs):
        m.train(); p = np.random.permutation(len(X_tr))
        for i in range(0, len(p), bs):
            bi = p[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        
        m.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(X_es), 4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X_es[i:i+4096])))).numpy())
        auc_es = roc_auc_score(label[es_idx], np.concatenate(pv))
        
        if auc_es > best_es + 1e-5:
            best_es = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat: break
    
    if bst: m.load_state_dict(bst)
    del X_tr, X_es, y_tr; gc.collect()
    
    def pred(idx):
        Xf = make_inter(norm(X_RAW[idx]), top_idx)
        m.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf), 4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        del Xf; gc.collect()
        return np.concatenate(pv)
    
    pv_te = pred(te_idx); pv_mv = pred(mv_idx)
    auc_te = roc_auc_score(label[te_idx], pv_te)
    auc_mv = roc_auc_score(label[mv_idx], pv_mv)
    del m, top_idx; gc.collect()
    return best_es, auc_mv, auc_te, pv_te, pv_mv

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

# ============================================
# 快速搜索: DCN vs ResMLP
# ============================================
print(f"\n{'='*70}", flush=True)
print("DCN Search (800K train, 单 seed)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)

configs = [
    ("DCN1", 3, [256,128], 0.3, 5e-4, 1e-3, 15, 0.05, 40, 8, 1024),
    ("DCN2", 5, [256,128], 0.3, 5e-4, 1e-3, 15, 0.05, 40, 8, 1024),
    ("DCN3", 3, [512,256], 0.3, 5e-4, 1e-3, 15, 0.05, 40, 8, 1024),
    ("DCN4", 5, [512,256], 0.3, 5e-4, 1e-3, 15, 0.05, 40, 8, 1024),
]

for name, cn, dp, dr, lr, wd, tk, sm, ep, pa, bs in configs:
    torch.manual_seed(42); np.random.seed(42)
    tc = time.time()
    es_a, mv_a, te_a, pv_t, _ = train_one(tr_sub, 42, cn, dp, dr, lr, wd, tk, sm, ep, pa, bs)
    elapsed = time.time() - tc
    n1 = int(len(pv_t) * 0.01)
    t1 = np.argsort(-pv_t)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    print(f"  {name}: TE={te_a:.4f} MV={mv_a:.4f} top1={top1:.1f}% | {elapsed:.0f}s", flush=True)
    del pv_t; gc.collect()

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
