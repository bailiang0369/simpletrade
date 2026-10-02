"""纯 NN: DCN (Deep & Cross Network) - 显式交叉层 替代 手工特征交互"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
print("Loading data...", flush=True)

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

X_es_norm = norm(es_idx)
X_mv_norm = norm(mv_idx)
X_te_norm = norm(te_idx)
gc.collect()

print(f"FT={FT}, te={len(te_idx)}", flush=True)

# ============================================
# DCN: Cross Network (显式特征交叉) + Deep Network
# ============================================
class CrossLayer(nn.Module):
    """DCN Cross Layer: x_{l+1} = x0 * x_l^T * w + b + x_l"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x0, x):
        # x0, x: (B, D)
        inter = x0 * (x @ self.w).unsqueeze(1) + self.b + x
        return inter

class DCN(nn.Module):
    def __init__(self, ft, cross_layers=3, deep_hs=[256, 128], drop=0.3):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(ft) for _ in range(cross_layers)])
        self.deep = nn.Sequential()
        prev = ft
        for h in deep_hs:
            self.deep.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)
            ])
            prev = h
        # 输出: cross out + deep out concat -> head
        self.head = nn.Linear(ft + prev, 1)
    
    def forward(self, x0):
        x_cross = x0
        for cl in self.cross_layers:
            x_cross = cl(x0, x_cross)
        x_deep = self.deep(x0)
        x_cat = torch.cat([x_cross, x_deep], dim=1)
        return self.head(x_cat).squeeze(-1)

# 对比: ResMLP (baseline, 无显式交叉)
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

def train_model(tr_idx_sub, seed, model_name, tr_norm_fn):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = norm(tr_idx_sub)
    y_tr = label[tr_idx_sub].astype(np.float32)
    y_es_f = label[es_idx].astype(np.float32)
    
    if model_name == 'resmlp_base':
        m = ResMLP(FT, [256, 128], 0.3)
        Xes = X_es_norm; Xmv = X_mv_norm; Xte = X_te_norm
    elif model_name == 'resmlp_inter':
        # 加手工交互
        corrs = np.array([abs(np.corrcoef(X_tr[:, j], y_tr)[0,1]) for j in range(FT)])
        top10 = np.argsort(-corrs)[:10]
        def addint(X, t=top10):
            X = np.nan_to_num(X, nan=0.0).astype(np.float32)
            X_sq = np.clip(X[:, t] ** 2, 0, 25)
            X_cross = np.concatenate([X[:, t[i:i+1]] * X[:, t[j:j+1]]
                                      for i in range(len(t)) for j in range(i+1, len(t))], axis=1)
            X_cross = np.clip(X_cross, -25, 25)
            return np.concatenate([X, X_sq, X_cross], axis=1)
        X_tr = addint(X_tr)
        Xes = addint(X_es_norm); Xmv = addint(X_mv_norm); Xte = addint(X_te_norm)
        m = ResMLP(X_tr.shape[1], [256, 128], 0.3)
    elif model_name == 'dcn':
        m = DCN(FT, cross_layers=3, deep_hs=[256, 128], drop=0.3)
        Xes = X_es_norm; Xmv = X_mv_norm; Xte = X_te_norm
    elif model_name == 'dcn_deep':
        m = DCN(FT, cross_layers=3, deep_hs=[512, 256, 128], drop=0.3)
        Xes = X_es_norm; Xmv = X_mv_norm; Xte = X_te_norm
    elif model_name == 'dcn_more_cross':
        m = DCN(FT, cross_layers=5, deep_hs=[256, 128], drop=0.3)
        Xes = X_es_norm; Xmv = X_mv_norm; Xte = X_te_norm
    
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    
    best_es = 0.0; bst = None; ni = 0
    for e in range(40):
        m.train(); idxs = np.random.permutation(len(X_tr))
        for i in range(0, len(idxs), 1024):
            bi = idxs[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb*(1-0.05)+0.025)
            opt.zero_grad(); loss.backward(); opt.step()
        
        m.eval()
        with torch.no_grad():
            pv_es = torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xes)))).numpy()
        auc_es = roc_auc_score(y_es_f, pv_es)
        
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
    
    pv_te = pred(Xte); pv_mv = pred(Xmv)
    auc_te = roc_auc_score(label[te_idx], pv_te)
    auc_mv = roc_auc_score(label[mv_idx], pv_mv)
    del m, X_tr, Xes, Xmv, Xte; gc.collect()
    return best_es, auc_mv, auc_te, pv_te, pv_mv

# ============================================
# 搜索: 各模型 × 单 seed (800K)
# ============================================
print(f"\n{'='*70}", flush=True)
print("Model comparison (800K train, single seed)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)

models = ['resmlp_base', 'resmlp_inter', 'dcn', 'dcn_deep', 'dcn_more_cross']
results = []
for mn in models:
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()
    es_a, mv_a, te_a, pv_t, pv_m = train_model(tr_sub, 42, mn, None)
    elapsed = time.time() - t0
    n1 = int(len(pv_t) * 0.01)
    t1 = np.argsort(-pv_t)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    results.append((mn, es_a, mv_a, te_a, top1, elapsed))
    print(f"  {mn:<20} ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} top1={top1:.1f}% | {elapsed:.0f}s", flush=True)
    del pv_t, pv_m; gc.collect()

print(f"\n{'Model':<20} {'TE_AUC':>8} {'top1%':>8} {'MV_AUC':>8}", flush=True)
print("-" * 44, flush=True)
for mn, es_a, mv_a, te_a, top1, _ in sorted(results, key=lambda x: -x[3]):
    print(f"{mn:<20} {te_a:>8.4f} {top1:>7.1f}% {mv_a:>8.4f}", flush=True)

# ============================================
# Top2 × 1.2M × 5-seed
# ============================================
print(f"\n{'='*70}", flush=True)
print("Top2 models × 1.2M × 5-seed ensemble", flush=True)
print(f"{'='*70}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

np.random.seed(42)
tr_final = np.random.choice(tr_idx, min(1_200_000, len(tr_idx)), replace=False)

top2_models = sorted(results, key=lambda x: -x[3])[:2]
for mn, _, _, _, _, _ in top2_models:
    print(f"\n--- {mn} ---", flush=True)
    pvs_t = []; pvs_m = []
    for seed in [42, 49, 56, 63, 70]:
        t0s = time.time()
        es_a, mv_a, te_a, pv_t, pv_m = train_model(tr_final, seed, mn, None)
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-t0s:.0f}s]", flush=True)
        del pv_t, pv_m; gc.collect()
    
    pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
    auc_e_t = roc_auc_score(label[te_idx], pv_e_t); auc_e_m = roc_auc_score(label[mv_idx], pv_e_m)
    print(f"\n    ★ ENSEMBLE: MV={auc_e_m:.4f} TE={auc_e_t:.4f}", flush=True)
    for pct in [0.005, 0.01, 0.015]:
        n = int(len(pv_e_t) * pct)
        ti = np.argsort(-pv_e_t)[:n]
        acc = label[te_idx][ti].mean() * 100
        n_days = len(te_idx) / 1440
        tpd = n / n_days
        print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)
    
    # Rolling eval
    print(f"\n    Rolling quantile:", flush=True)
    DAYS = len(te_idx) / 1440
    pv_arr = np.array(pv_e_t, dtype=np.float64)
    ts_arr = np.array(ts_te_vals, dtype=np.int64)
    day_sec = 86400
    day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    for q in [99, 99.2, 99.5]:
        trades = []
        for d in range(30, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - 30]; hist_hi = day_lo
            hist_mask = (ts_arr >= hist_lo) & (ts_arr < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100: continue
            thr = np.percentile(hist_pv, q)
            pick_mask = (ts_arr >= day_lo) & (ts_arr < day_hi) & (pv_arr >= thr)
            if pick_mask.sum() > 0:
                trades.extend(label[te_idx][pick_mask].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / DAYS
            print(f"      q={q}: acc={acc:.1f}% tpd={tpd:.1f} n={len(trades)}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/nn_dcn.npz', search_results=results)
print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
