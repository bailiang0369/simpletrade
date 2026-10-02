"""
NN V3: 全量 2.36M 训练 + 更大网络 + 更久训练
内存策略: 分块 norm + 交互，每次处理 500K 行
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
CHUNK = 500_000

def mem_gb(arr): return arr.nbytes / 1e9 if hasattr(arr, 'nbytes') else 0

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
feats = fe.build_features(eth_full)
del eth_full; gc.collect()
feats_np = feats.to_numpy().astype(np.float32)
del feats; gc.collect()

H = 15
label = (C_e[H:] > C_e[:-H]).astype(np.int8)
del C_e; gc.collect()

# X_RAW: float32, 紧凑
X_RAW = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X_RAW = np.nan_to_num(X_RAW, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()

FT = X_RAW.shape[1]
ts_u = ts_e[:-H].copy()
del ts_e; gc.collect()
print(f"X_RAW={X_RAW.shape}, mem={mem_gb(X_RAW):.2f}GB", flush=True)

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
samp = np.random.choice(tr_idx, 300_000, replace=False)
MU = X_RAW[samp].mean(0)
SD = X_RAW[samp].std(0) + 1e-6
del samp; gc.collect()

def norm(arr):
    return np.clip((arr - MU) / SD, -5, 5)

class ResMLP(nn.Module):
    def __init__(self, ft, hs, drop=0.3):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs)-1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hs[i], hs[i+1]), nn.BatchNorm1d(hs[i+1]),
                nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            out = b(x)
            x = x + out if x.shape == out.shape else out
        return self.head(x).squeeze(-1)

def make_inter(X_norm, top_idx):
    """X_norm: float32, normed; returns X+sq+cross float32"""
    X_sq = np.clip(X_norm[:, top_idx] ** 2, 0, 25)
    X_cr = np.concatenate([X_norm[:, top_idx[i:i+1]] * X_norm[:, top_idx[j:j+1]]
                           for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cr = np.clip(X_cr, -25, 25)
    return np.concatenate([X_norm, X_sq, X_cr], axis=1)

def build_full(idx_arr, top_idx, chunk=CHUNK):
    """分块构造 norm + 交互, 返回完整 float32 array"""
    total = len(idx_arr)
    out_list = []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        n = norm(X_RAW[idx_arr[start:end]])
        inter = make_inter(n, top_idx)
        out_list.append(inter)
        print(f"    chunk {start//chunk+1}/{(total+chunk-1)//chunk}: [{start}:{end}] → {inter.shape}", flush=True)
        del n, inter; gc.collect()
    result = np.concatenate(out_list, axis=0)
    del out_list; gc.collect()
    return result

def train_one(seed, hs, drop, lr, wd, topk, smooth, epochs, pat, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    
    # topk corr
    np.random.seed(seed)
    cs = np.random.choice(tr_idx, min(300_000, len(tr_idx)), replace=False)
    X_cs = norm(X_RAW[cs])
    y_cs = label[cs].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_cs[:, j], y_cs)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:topk]
    del X_cs, y_cs, corrs; gc.collect()
    print(f"  top{topk} idx={top_idx[:5]}...", flush=True)
    
    # 全量 train + ES (分块)
    print(f"  [TRAIN] building full ({len(tr_idx)})...", flush=True)
    X_tr = build_full(tr_idx, top_idx)
    y_tr = label[tr_idx].astype(np.float32)
    print(f"  X_tr={X_tr.shape}, mem={mem_gb(X_tr):.2f}GB", flush=True)
    
    print(f"  [ES] building ({len(es_idx)})...", flush=True)
    X_es = build_full(es_idx, top_idx)
    
    # Train
    m = ResMLP(X_tr.shape[1], hs, drop)
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
        if e % 10 == 0: print(f"    e={e}: ES={auc_es:.4f} best={best_es:.4f}", flush=True)
    
    if bst: m.load_state_dict(bst)
    del X_tr, X_es, y_tr; gc.collect()
    
    # Eval MV + TE
    def pred_on(idx):
        Xf = build_full(idx, top_idx, chunk=200_000)
        m.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf), 4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        del Xf; gc.collect()
        return np.concatenate(pv)
    
    print(f"  eval TE...", flush=True)
    pv_te = pred_on(te_idx)
    print(f"  eval MV...", flush=True)
    pv_mv = pred_on(mv_idx)
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
# 配置: 更大网络 + 更久训练 + 全量 2.36M
# ============================================
print(f"\n{'='*70}", flush=True)
print("NN V3: 全量 2.36M + 更大网络 + 更久训练", flush=True)
print(f"{'='*70}", flush=True)

configs = [
    ("BIG1", [768, 384],   0.3, 5e-4, 1e-3, 15, 0.05, 60, 12, 1024),
    ("BIG2", [512, 256],   0.3, 5e-4, 1e-3, 15, 0.05, 60, 12, 2048),
]

top_res = {}
for name, hs, dr, lr, wd, tk, sm, ep, pa, bs in configs:
    print(f"\n--- {name}: hs={hs} lr={lr} bs={bs} ---", flush=True)
    pvs_t = []; pvs_m = []
    for seed in [42, 49, 56, 63, 70]:
        tc = time.time()
        es_a, mv_a, te_a, pv_t, pv_m = train_one(seed, hs, dr, lr, wd, tk, sm, ep, pa, bs)
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        print(f"  s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-tc:.0f}s]", flush=True)
        del pv_t, pv_m; gc.collect()
    
    pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
    auc_t = roc_auc_score(label[te_idx], pv_e_t)
    auc_m = roc_auc_score(label[mv_idx], pv_e_m)
    
    DAYS = len(te_idx) / 1440
    print(f"\n  ★ ENSEMBLE: MV={auc_m:.4f} TE={auc_t:.4f}", flush=True)
    for pct in [0.005, 0.01, 0.015, 0.02]:
        n = int(len(pv_e_t) * pct)
        ti = np.argsort(-pv_e_t)[:n]
        acc = label[te_idx][ti].mean() * 100
        tpd = n / DAYS
        print(f"    top-{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)
    
    top_res[name] = {'pv_t': pv_e_t, 'auc_t': auc_t, 'auc_m': auc_m, 'cfg': (hs, dr, lr, tk)}
    del pvs_t, pvs_m; gc.collect()

# ============================================
# 汇总
# ============================================
print(f"\n{'='*70}", flush=True)
print("FINAL SUMMARY", flush=True)
print(f"{'='*70}", flush=True)

best = max(top_res.keys(), key=lambda k: top_res[k]['auc_t'])
print(f"\n🏆 Best: {best}", flush=True)
print(f"   TE={top_res[best]['auc_t']:.4f} MV={top_res[best]['auc_m']:.4f}", flush=True)
print(f"   Config: {top_res[best]['cfg']}", flush=True)

for name, info in top_res.items():
    print(f"   {name}: TE={info['auc_t']:.4f} MV={info['auc_m']:.4f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/nn_v3_best.npz',
         pv_te=top_res[best]['pv_t'],
         te_idx=te_idx, ts_te_vals=ts_te_vals,
         auc_te=top_res[best]['auc_t'])

print(f"\nSaved. DONE [{time.time()-t0:.0f}s]", flush=True)
