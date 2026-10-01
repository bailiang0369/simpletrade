"""V6: 纯神经网络强力优化 - 0% 树模型
策略:
  A) Wide ResMLP [1024,512,256] × 15 seeds
  B) Deep ResMLP [512,256,256,128] × 15 seeds  
  C) TCN1D (特征当序列) × 15 seeds
  + top30交互特征 (30+435=465个新特征)
  + CosineAnnealingWarmRestarts + 样本权重 + 标签平滑 + 梯度累积
  + 三架构 rank ensemble
"""
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

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
if len(tr_idx) > 1_000_000:
    tr_idx = np.random.choice(tr_idx, 1_000_000, replace=False)

X_tr = X_all[tr_idx].copy()
X_es = X_all[es_mask].copy()
X_mv = X_all[mv_mask].copy()
X_te = X_all[te_mask].copy()
y_tr = label[tr_idx]; r_tr = ret_future[tr_idx]
y_es = label[es_mask]; y_mv = label[mv_mask]; y_te = label[te_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

# NaN handling + robust z-score
FT = X_tr.shape[1]
for j in range(FT):
    col = X_tr[:,j]; col[np.isnan(col)] = 0
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    col = np.clip(col, lo, hi)
    m, s = col.mean(), col.std() + 1e-6
    X_tr[:,j] = (col - m) / s
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()

# Sample weights: extreme returns downweighted
ret_abs = np.abs(r_tr)
med_ret = np.median(ret_abs)
sw = np.where(ret_abs > np.percentile(ret_abs, 90), 0.3, 1.0).astype(np.float32)
sw = sw * np.clip(ret_abs / (med_ret + 1e-8), 0.5, 3.0).astype(np.float32)
print(f"Sample weights: mean={sw.mean():.3f}, extreme_down={(ret_abs > np.percentile(ret_abs,90)).mean():.1%}", flush=True)
del r_tr; gc.collect()

# Interaction features: top30
corrs = np.array([abs(np.corrcoef(X_tr[:,j], y_tr)[0,1]) for j in range(FT)])
TOPK = 30
topk_idx = np.argsort(-corrs)[:TOPK]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cb = X[:, topk] ** 3  # cubic for good measure
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, X_cb, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn = add_interactions(X_tr, topk_idx)
X_es_nn = add_interactions(X_es, topk_idx)
X_mv_nn = add_interactions(X_mv, topk_idx)
X_te_nn = add_interactions(X_te, topk_idx)
del X_tr, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn.shape[1]
print(f"TR={X_tr_nn.shape} TE={X_te_nn.shape} FT_BASE={FT} FT_NN={FT_NN}", flush=True)

# Pre-push to device for speed
X_tr_d = torch.from_numpy(X_tr_nn).to(device)
X_es_d = torch.from_numpy(X_es_nn).to(device)
X_mv_d = torch.from_numpy(X_mv_nn).to(device)
X_te_d = torch.from_numpy(X_te_nn).to(device)
y_tr_d = torch.from_numpy(y_tr).float().to(device)
sw_d = torch.from_numpy(sw).to(device)
del X_tr_nn, X_es_nn, X_mv_nn, X_te_nn; gc.collect()

# ============================================
# Models
# ============================================
class WideResMLP(nn.Module):
    """宽版: [1024, 512, 256] + PreNorm残差"""
    def __init__(self, ft, drop=0.3):
        super().__init__()
        hs = [1024, 512, 256]
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        prev = hs[0]
        for h in hs:
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(prev),
                nn.Linear(prev, h), nn.GELU(), nn.Dropout(drop),
                nn.Linear(h, h), nn.Dropout(drop)))
            prev = h
        self.norm_out = nn.LayerNorm(hs[-1])
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x)
        return self.head(self.norm_out(x)).squeeze(-1)

class DeepResMLP(nn.Module):
    """深版: [512, 256, 256, 128] + PreNorm残差"""
    def __init__(self, ft, drop=0.25):
        super().__init__()
        hs = [512, 256, 256, 128]
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        prev = hs[0]
        for h in hs:
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(prev),
                nn.Linear(prev, h), nn.GELU(), nn.Dropout(drop),
                nn.Linear(h, h), nn.Dropout(drop)))
            prev = h
        self.norm_out = nn.LayerNorm(hs[-1])
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x)
        return self.head(self.norm_out(x)).squeeze(-1)

class TCN1D(nn.Module):
    """把特征当 1×FT_NN 序列, TCN提取局部模式"""
    def __init__(self, ft, chs=[32, 64, 128, 64], drop=0.3):
        super().__init__()
        prev = 1; d = 1; blocks = []
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(chs[-1]//2, 1))
    def forward(self, x):
        x = x.unsqueeze(1)
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

class FusedNet(nn.Module):
    """WideResMLP + TCN 特征融合"""
    def __init__(self, ft, drop=0.3):
        super().__init__()
        # MLP branch
        self.mlp_input = nn.Linear(ft, 512)
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, 512)),
            nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 256)),
        ])
        self.mlp_reduce = nn.Linear(256, 128)
        # TCN branch
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=2, dilation=1), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=4, dilation=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=8, dilation=4), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.tcn_reduce = nn.Linear(64, 128)
        # Fusion
        self.fuse = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 1))
    def forward(self, x):
        # MLP path
        m = F.gelu(self.mlp_input(x))
        for b in self.mlp_blocks: m = m + b(m)
        m = self.mlp_reduce(m)
        # TCN path  
        t = self.tcn(x.unsqueeze(1)).flatten(1)
        t = self.tcn_reduce(t)
        # Fuse
        return self.fuse(torch.cat([m, t], 1)).squeeze(-1)

# ============================================
# Training utilities
# ============================================
def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            chunk = X[i:i+bs] if X.device != device else X[i:i+bs]
            pv.append(torch.sigmoid(m(chunk)).cpu().numpy())
    return np.concatenate(pv)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def train_model(m, Xtr, ytr, sw_tr, Xes, yes,
                ep=40, lr=3e-3, wd=1e-3, bs=2048, pat=10, smooth=0.05,
                grad_acc=1, use_sw=True):
    """带 CosineAnnealingWarmRestarts + label smoothing + sample weights"""
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=lr*0.01)
    best=0.0; bst=None; ni=0; best_ep=0
    n = len(Xtr)
    for e in range(ep):
        m.train()
        # 每次epoch重排
        perm = torch.randperm(n, device=device)
        tot_loss = 0.0; cnt = 0
        for i in range(0, n, bs):
            bi = perm[i:i+bs]
            xb = Xtr[bi]
            yb = ytr[bi]
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb)
            if use_sw:
                wb = sw_tr[bi]
                loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss = loss / grad_acc
            loss.backward()
            tot_loss += loss.item() * grad_acc; cnt += 1
            if (cnt % grad_acc == 0) or (i + bs >= n):
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        sched.step()
        # Evaluate ES (do it on CPU to save GPU mem during training)
        pv_es = evaluate(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te_d)
    pv_mv = evaluate(m, X_mv_d)
    return pv_te, pv_mv

def run_architecture(label, model_fn, seeds, train_kwargs):
    print(f"\n{'='*65}\n{label}\n{'='*65}", flush=True)
    pvs_t, pvs_m = [], []
    for i, s in enumerate(seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = model_fn().to(device)
        np_ = sum(p.numel() for p in m.parameters())
        t0s = time.time()
        pv_t, pv_m = train_model(m, X_tr_d, y_tr_d, sw_d, X_es_d, y_es, **train_kwargs)
        auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
        print(f"  [{i+1}/{len(seeds)}] s={s}: MV={auc_m:.4f} TE={auc_t:.4f} p={np_:,} [{time.time()-t0s:.0f}s]", flush=True)
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pv_ens_t = rank_agg(pvs_t); pv_ens_m = rank_agg(pvs_m)
    auc_te = roc_auc_score(y_te, pv_ens_t); auc_mv = roc_auc_score(y_mv, pv_ens_m)
    print(f"\n  ★ {label} ENSEMBLE: MV={auc_mv:.4f} TE={auc_te:.4f}", flush=True)
    return pv_ens_t, pv_ens_m, auc_te, auc_mv

SEEDS = [42, 49, 56, 63, 70, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178]

# ============================================
# Architecture A: Wide ResMLP
# ============================================
pv_wide_t, pv_wide_m, auc_wide_t, auc_wide_m = run_architecture(
    "A) WideResMLP [1024,512,256] ×15",
    lambda: WideResMLP(FT_NN, drop=0.30),
    SEEDS,
    dict(ep=45, lr=2e-3, wd=5e-4, bs=2048, pat=12, smooth=0.05, grad_acc=1, use_sw=True))

# ============================================
# Architecture B: Deep ResMLP
# ============================================
pv_deep_t, pv_deep_m, auc_deep_t, auc_deep_m = run_architecture(
    "B) DeepResMLP [512,256,256,128] ×15",
    lambda: DeepResMLP(FT_NN, drop=0.25),
    SEEDS,
    dict(ep=50, lr=2e-3, wd=5e-4, bs=2048, pat=14, smooth=0.05, grad_acc=1, use_sw=True))

# ============================================
# Architecture C: TCN1D
# ============================================
pv_tcn_t, pv_tcn_m, auc_tcn_t, auc_tcn_m = run_architecture(
    "C) TCN1D chs=[32,64,128,64] ×15",
    lambda: TCN1D(FT_NN, chs=[32,64,128,64], drop=0.30),
    SEEDS,
    dict(ep=50, lr=3e-3, wd=1e-3, bs=2048, pat=14, smooth=0.05, grad_acc=1, use_sw=True))

# ============================================
# Architecture D: Fused Net (MLP + TCN)
# ============================================
pv_fused_t, pv_fused_m, auc_fused_t, auc_fused_m = run_architecture(
    "D) FusedNet (MLP+TCN) ×15",
    lambda: FusedNet(FT_NN, drop=0.30),
    SEEDS,
    dict(ep=45, lr=2e-3, wd=5e-4, bs=2048, pat=12, smooth=0.05, grad_acc=1, use_sw=True))

# ============================================
# Cross-architecture ensemble (rank blend)
# ============================================
print(f"\n{'='*65}", flush=True)
print("Cross-architecture rank ensemble", flush=True)
print(f"{'='*65}", flush=True)

def rrank(p):
    return np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)

cands = [('Wide', pv_wide_t, pv_wide_m), ('Deep', pv_deep_t, pv_deep_m),
         ('TCN', pv_tcn_t, pv_tcn_m), ('Fused', pv_fused_t, pv_fused_m)]

# Correlation matrix on test (tail)
print("\nPairwise rank corr on test:")
for i in range(len(cands)):
    for j in range(i+1, len(cands)):
        ri, rj = rrank(cands[i][1]), rrank(cands[j][1])
        ci = np.argsort(-ri)[:int(len(ri)*0.01)]
        corr_all = np.corrcoef(ri, rj)[0,1]
        corr_tail = np.corrcoef(ri[ci], rj[ci])[0,1]
        print(f"  {cands[i][0]} vs {cands[j][0]}: all={corr_all:.4f} tail1%={corr_tail:.4f}", flush=True)

# Try all pairs + triples + full ensemble
best_auc = 0; best_combo = None; best_names = None
from itertools import combinations
for k in range(2, len(cands)+1):
    for combo in combinations(cands, k):
        p_mv = np.mean([rrank(c[2]) for c in combo], axis=0)
        auc_mv = roc_auc_score(y_mv, p_mv)
        if auc_mv > best_auc:
            best_auc = auc_mv
            best_combo = combo
            best_names = [c[0] for c in combo]

print(f"\nBest combo ({len(best_combo)} models): {' + '.join(best_names)} MV_AUC={best_auc:.4f}", flush=True)

# Evaluate best combo on test
pv_best_t = np.mean([rrank(c[1]) for c in best_combo], axis=0)
pv_best_m = np.mean([rrank(c[2]) for c in best_combo], axis=0)
auc_best_t = roc_auc_score(y_te, pv_best_t)
auc_best_m = roc_auc_score(y_mv, pv_best_m)
print(f"  ★ CROSS-ARCH ENSEMBLE: MV={auc_best_m:.4f} TE={auc_best_t:.4f}", flush=True)

# ============================================
# Full ensemble: all 4 models
# ============================================
pv_all_t = np.mean([rrank(c[1]) for c in cands], axis=0)
pv_all_m = np.mean([rrank(c[2]) for c in cands], axis=0)
auc_all_t = roc_auc_score(y_te, pv_all_t); auc_all_m = roc_auc_score(y_mv, pv_all_m)
print(f"  ALL-4 Ensemble: MV={auc_all_m:.4f} TE={auc_all_t:.4f}", flush=True)

# ============================================
# Top-k accuracy evaluation
# ============================================
print(f"\n{'='*65}", flush=True)
print("Top-k accuracy (test set)", flush=True)
print(f"{'='*65}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def topk_eval(name, pv, y, days=DAYS):
    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/days
        flag = '🏆' if pct==1.0 and acc>=62 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# Each architecture solo
for nm, pvt, _ in cands:
    topk_eval(f"NN-{nm}", pvt, y_te)

# Best combo
topk_eval(f"CROSS-ARCH ({'+'.join(best_names)})", pv_best_t, y_te)
topk_eval("ALL-4 Ensemble", pv_all_t, y_te)

# ============================================
# Save results
# ============================================
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/v6_nn_only.npz',
         pv_wide_te=pv_wide_t, pv_wide_mv=pv_wide_m,
         pv_deep_te=pv_deep_t, pv_deep_mv=pv_deep_m,
         pv_tcn_te=pv_tcn_t, pv_tcn_mv=pv_tcn_m,
         pv_fused_te=pv_fused_t, pv_fused_mv=pv_fused_m,
         pv_best_te=pv_best_t, pv_best_mv=pv_best_m,
         pv_all_te=pv_all_t, pv_all_mv=pv_all_m,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te,
         topk_idx=topk_idx)
print(f"\nSaved v6_nn_only.npz", flush=True)

print(f"\n{'='*65}", flush=True)
print(f"V6 DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"  WideResMLP  TE={auc_wide_t:.4f}", flush=True)
print(f"  DeepResMLP  TE={auc_deep_t:.4f}", flush=True)
print(f"  TCN1D       TE={auc_tcn_t:.4f}", flush=True)
print(f"  FusedNet    TE={auc_fused_t:.4f}", flush=True)
print(f"  BEST COMBO  TE={auc_best_t:.4f}", flush=True)
print(f"  ALL-4       TE={auc_all_t:.4f}", flush=True)
print(f"{'='*65}", flush=True)
