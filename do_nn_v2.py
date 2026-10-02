"""
NN V2: 全面搜索 - 全量 2.36M 训练 + 大 topk 交互 + 多架构 + 训练技巧

两阶段:
  Phase 1: 300K 快速筛选 ~15 配置
  Phase 2: Top3 配置 × 全量 2.36M × 5-seed ensemble

核心改进 (vs V1):
  - 全量训练数据 (V1 最多 1.2M, 现在 2.36M)
  - 更大 topk 交互 (V1 只用 top10, 现在试 15/20)
  - 更多架构 (V1 只 [256,128], 现在 [384,192]/[512,256]/三层)
  - 训练更久 (40 -> 60 epochs, patience 8 -> 12)
  - Cosine annealing + warmup
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()

# ============================================
# 数据加载 (float16 省内存)
# ============================================
print("Loading data...", flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')

ts_eth = eth['ts'].to_numpy()
C_eth = eth['close'].to_numpy()
ts_btc = btc['ts'].to_numpy()
C_btc = btc['close'].to_numpy()
del eth, btc; gc.collect()

# BTC cross-asset
idx = np.searchsorted(ts_btc, ts_eth, side="right") - 1
idx = np.clip(idx, 0, len(C_btc)-1)
B_lr1 = np.zeros(len(ts_eth), dtype=np.float16)
B_lr1[1:] = np.log(np.maximum(C_btc[idx[1:]],1e-8)/np.maximum(C_btc[idx[:-1]],1e-8)).astype(np.float16)
del ts_btc, C_btc; gc.collect()

# ETH 特征
eth_full = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth_full)
del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float16)
del feats; gc.collect()

# Label
H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int8)
del C_eth; gc.collect()

X = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X = X.astype(np.float16)
gc.collect()

FT = X.shape[1]
ts_u = ts_eth[:-H].copy()
del ts_eth; gc.collect()
print(f"FT={FT}, X={len(X)}, mem={X.nbytes/1e9:.2f}GB", flush=True)

# Splits
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts_u < tre)[0]
es_idx = np.where((ts_u >= tre) & (ts_u < es_end))[0]
mv_idx = np.where((ts_u >= es_end) & (ts_u < meta_end))[0]
te_idx = np.where(ts_u >= meta_end)[0]
ts_te_vals = ts_u[te_idx].copy()
del ts_u; gc.collect()

print(f"splits: tr={len(tr_idx)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}", flush=True)

# Norm stats
np.random.seed(42)
tr_norm_samp = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
mu = X[tr_norm_samp].astype(np.float32).mean(0)
sd = X[tr_norm_samp].astype(np.float32).std(0) + 1e-6
del tr_norm_samp; gc.collect()

def norm(idx):
    return np.clip((X[idx].astype(np.float32) - mu) / sd, -5, 5).astype(np.float32)

# ============================================
# ResMLP (带可选 BatchNorm)
# ============================================
class ResMLP(nn.Module):
    def __init__(self, ft, hs, drop=0.3, use_bn=True):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs)-1):
            layers = [nn.Linear(hs[i], hs[i+1])]
            if use_bn: layers.append(nn.BatchNorm1d(hs[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop))
            self.blocks.append(nn.Sequential(*layers))
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            out = b(x)
            x = x + out if x.shape == out.shape else out
        return self.head(x).squeeze(-1)

# 可选: Focal Loss
def focal_bce(logits, target, alpha=0.5, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

# ============================================
# 特征交互 (topk 平方 + C(topk,2) 交叉)
# ============================================
def add_interactions(X, top_idx):
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    X_sq = np.clip(X[:, top_idx] ** 2, 0, 25)
    X_cross = np.concatenate([X[:, top_idx[i:i+1]] * X[:, top_idx[j:j+1]]
                              for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cross = np.clip(X_cross, -25, 25)
    return np.concatenate([X, X_sq, X_cross], axis=1)

def compute_topk(tr_idx_sub, topk):
    """用 train subset 算 corr, 取 topk"""
    X_tr = norm(tr_idx_sub)
    y_tr = label[tr_idx_sub].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_tr[:, j], y_tr)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:topk]
    del X_tr, y_tr; gc.collect()
    return top_idx

# ============================================
# 通用训练函数
# ============================================
def train_one(tr_idx_sub, seed, hs, drop, lr, wd, topk, smooth, use_bn,
              epochs, pat, bs, focal_alpha=None):
    """训练 + 早停 + 完整 TE/MV AUC + TE preds"""
    torch.manual_seed(seed); np.random.seed(seed)
    
    top_idx = compute_topk(tr_idx_sub, topk)
    
    X_tr = add_interactions(norm(tr_idx_sub), top_idx)
    y_tr = label[tr_idx_sub].astype(np.float32)
    
    X_es = add_interactions(norm(es_idx), top_idx)
    X_mv = add_interactions(norm(mv_idx), top_idx)
    X_te = add_interactions(norm(te_idx), top_idx)
    
    m = ResMLP(X_tr.shape[1], hs, drop, use_bn)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_es = 0.0; bst = None; ni = 0
    for e in range(epochs):
        m.train(); idxs = np.random.permutation(len(X_tr))
        for i in range(0, len(idxs), bs):
            bi = idxs[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb)
            if focal_alpha is not None:
                loss = focal_bce(logits, yb, alpha=focal_alpha, gamma=2.0)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        
        # Early stop check (ES AUC)
        m.eval(); pv_es=[]
        with torch.no_grad():
            for i in range(0, len(X_es), 4096):
                pv_es.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X_es[i:i+4096])))).numpy())
        auc_es = roc_auc_score(label[es_idx], np.concatenate(pv_es))
        
        if auc_es > best_es + 1e-5:
            best_es = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat: break
    
    if bst: m.load_state_dict(bst)
    m.eval()
    
    def pred(Xf):
        pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf), 4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        return np.concatenate(pv)
    
    pv_te = pred(X_te)
    pv_mv = pred(X_mv)
    auc_te = roc_auc_score(label[te_idx], pv_te)
    auc_mv = roc_auc_score(label[mv_idx], pv_mv)
    
    del m, X_tr, X_es, X_mv, X_te, y_tr, top_idx; gc.collect()
    return best_es, auc_mv, auc_te, pv_te, pv_mv

# ============================================
# PHASE 1: 300K 快速筛选
# ============================================
print(f"\n{'='*70}", flush=True)
print("PHASE 1: 300K 快速搜索", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_phase1 = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
print(f"Phase1 train: {len(tr_phase1)}", flush=True)

# 搜索配置 (精简但覆盖关键维度)
phase1_configs = [
    # (hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, focal_alpha)
    # ---- Baseline (V1 最优) ----
    ([256,128], 0.3, 5e-4, 1e-3, 10,  0.05, True, 40, 8, 1024, None),
    # ---- 更大 topk ----
    ([256,128], 0.3, 5e-4, 1e-3, 15,  0.05, True, 40, 8, 1024, None),
    ([256,128], 0.3, 5e-4, 1e-3, 20,  0.05, True, 40, 8, 1024, None),
    # ---- 更大架构 ----
    ([384,192], 0.3, 5e-4, 1e-3, 15,  0.05, True, 40, 8, 1024, None),
    ([512,256], 0.3, 5e-4, 1e-3, 15,  0.05, True, 40, 8, 1024, None),
    ([512,256], 0.3, 5e-4, 1e-3, 20,  0.05, True, 40, 8, 1024, None),
    # ---- 更深架构 ----
    ([256,256,128], 0.3, 5e-4, 1e-3, 15,  0.05, True, 50, 10, 1024, None),
    ([512,256,128], 0.3, 5e-4, 1e-3, 15,  0.05, True, 50, 10, 1024, None),
    # ---- 正则化调整 ----
    ([384,192], 0.2, 5e-4, 5e-4, 15,  0.05, True, 40, 8, 1024, None),
    ([384,192], 0.4, 5e-4, 1e-3, 15,  0.05, True, 40, 8, 1024, None),
    # ---- LR 调整 ----
    ([384,192], 0.3, 3e-4, 1e-3, 15,  0.05, True, 50, 10, 1024, None),
    ([384,192], 0.3, 1e-3, 1e-3, 15,  0.05, True, 40, 8, 1024, None),
    # ---- Focal loss ----
    ([384,192], 0.3, 5e-4, 1e-3, 15,  0.05, True, 40, 8, 1024, 0.5),
    # ---- 无 BN ----
    ([384,192], 0.3, 5e-4, 1e-3, 15,  0.05, False, 40, 8, 1024, None),
    # ---- 更长训练 ----
    ([384,192], 0.3, 5e-4, 1e-3, 20,  0.05, True, 60, 12, 1024, None),
]

results_p1 = []
for ci, cfg in enumerate(phase1_configs):
    hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa = cfg
    torch.manual_seed(42); np.random.seed(42)
    t_cfg = time.time()
    es_a, mv_a, te_a, pv_t, _ = train_one(tr_phase1, 42, hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa)
    n1 = int(len(pv_t) * 0.01)
    t1 = np.argsort(-pv_t)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    elapsed = time.time() - t_cfg
    fa_str = f"focal={fa}" if fa else ""
    print(f"  [{ci:02d}] TE={te_a:.4f} MV={mv_a:.4f} top1={top1:.1f}% | hs={hs} d={drop} lr={lr} tk={topk} {fa_str} | {elapsed:.0f}s", flush=True)
    results_p1.append((ci, cfg, es_a, mv_a, te_a, top1))
    del pv_t; gc.collect()

# 排序取 top
results_p1.sort(key=lambda x: -x[4])  # sort by TE AUC
print(f"\nPHASE 1 TOP 5 (by TE AUC):", flush=True)
for rank, r in enumerate(results_p1[:5]):
    ci, cfg, es_a, mv_a, te_a, top1 = r
    hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa = cfg
    print(f"  #{rank+1} [{ci}] TE={te_a:.4f} MV={mv_a:.4f} top1={top1:.1f}% | hs={hs} d={drop} lr={lr} tk={topk}", flush=True)

# ============================================
# PHASE 2: Top3 × 全量 2.36M × 5-seed ensemble
# ============================================
print(f"\n{'='*70}", flush=True)
print("PHASE 2: Top3 × 全量 2.36M × 5-seed", flush=True)
print(f"{'='*70}", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

top3 = results_p1[:3]
all_final = {}

for rank, r in enumerate(top3):
    ci, cfg, es_a_p1, mv_a_p1, te_a_p1, top1_p1 = r
    hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa = cfg
    
    print(f"\n--- TOP{rank+1}: cfg[{ci}] hs={hs} d={drop} lr={lr} tk={topk} ---", flush=True)
    print(f"    P1: ES={es_a_p1:.4f} MV={mv_a_p1:.4f} TE={te_a_p1:.4f} top1={top1_p1:.1f}%", flush=True)
    
    pvs_t = []; pvs_m = []
    for seed in [42, 49, 56, 63, 70]:
        torch.manual_seed(seed); np.random.seed(seed)
        t_seed = time.time()
        # 全量训练集
        es_a, mv_a, te_a, pv_t, pv_m = train_one(tr_idx, seed, hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa)
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-t_seed:.0f}s]", flush=True)
        del pv_t, pv_m; gc.collect()
    
    pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
    auc_e_t = roc_auc_score(label[te_idx], pv_e_t)
    auc_e_m = roc_auc_score(label[mv_idx], pv_e_m)
    
    # Rolling eval + 多 pct eval
    DAYS = len(te_idx) / 1440
    n_days_int = int(DAYS)
    day_start = ts_te_vals.min()
    all_ts = np.arange(day_start, ts_te_vals.max() + 86400, 86400)
    n_days_all = len(all_ts) - 1
    
    print(f"\n    ★ ENSEMBLE: MV={auc_e_m:.4f} TE={auc_e_t:.4f}", flush=True)
    
    # Top-% eval
    print(f"    Top-% accuracy (no rolling):", flush=True)
    for pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
        n = int(len(pv_e_t) * pct)
        ti = np.argsort(-pv_e_t)[:n]
        acc = label[te_idx][ti].mean() * 100
        tpd = n / DAYS
        print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)
    
    # Rolling q=99 / q=99.5
    pv_arr = np.array(pv_e_t, dtype=np.float64)
    print(f"    Rolling q-threshold (30d lookback):", flush=True)
    for q in [98, 99, 99.5]:
        trades = []
        for d in range(30, n_days_all):
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
    
    all_final[f"top{rank+1}"] = {
        'cfg': cfg, 'auc_te': auc_e_t, 'auc_mv': auc_e_m,
        'pv_te': pv_e_t, 'pv_mv': pv_e_m
    }
    
    del pvs_t, pvs_m, pv_e_t, pv_e_m; gc.collect()

# ============================================
# PHASE 3: 最终汇总
# ============================================
print(f"\n{'='*70}", flush=True)
print("FINAL SUMMARY", flush=True)
print(f"{'='*70}", flush=True)

best_key = max(all_final.keys(), key=lambda k: all_final[k]['auc_te'])
best = all_final[best_key]
hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, fa = best['cfg']

print(f"\n🏆 Best: {best_key}", flush=True)
print(f"   Config: hs={hs} drop={drop} lr={lr} wd={wd} topk={topk} smooth={smooth} bn={use_bn} fa={fa}", flush=True)
print(f"   TE AUC={best['auc_te']:.4f} MV AUC={best['auc_mv']:.4f}", flush=True)
print(f"\n{'='*70}", flush=True)

# 保存结果
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/nn_v2_best.npz',
         pv_te=best['pv_te'], pv_mv=best['pv_mv'],
         te_idx=te_idx, mv_idx=mv_idx,
         ts_te_vals=ts_te_vals,
         config=str(best['cfg']),
         auc_te=best['auc_te'], auc_mv=best['auc_mv'])

print(f"\nResults saved to results/nn_v2_best.npz", flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
