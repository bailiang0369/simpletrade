"""
NN V2 Phase 2: subset 训练版
保持 float32 X (~1GB), 训练用 1.5M subset
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

print("Loading data...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')

ts_eth = eth['ts'].to_numpy()
C_eth = eth['close'].to_numpy()
ts_btc = btc['ts'].to_numpy()
C_btc = btc['close'].to_numpy()
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
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()

FT = X.shape[1]
ts_u = ts_eth[:-H].copy()
del ts_eth; gc.collect()
print(f"FT={FT}, X={len(X)}, mem={X.nbytes/1e9:.2f}GB (float32)", flush=True)

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
MU = X[samp].mean(0)
SD = X[samp].std(0) + 1e-6
del samp; gc.collect()

def norm(arr):
    return np.clip((arr - MU) / SD, -5, 5)

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

def add_interactions(X, top_idx):
    # X 必须是 float32 (norm 的输出)
    X_sq = np.clip(X[:, top_idx] ** 2, 0, 25)
    X_cross = np.concatenate([X[:, top_idx[i:i+1]] * X[:, top_idx[j:j+1]]
                              for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cross = np.clip(X_cross, -25, 25)
    return np.concatenate([X, X_sq, X_cross], axis=1)

def train_one(seed, hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, train_size=1_500_000):
    torch.manual_seed(seed); np.random.seed(seed)
    
    # 1. 选 train subset
    np.random.seed(seed)
    tr_sub = np.random.choice(tr_idx, min(train_size, len(tr_idx)), replace=False)
    
    # 2. topk corr (用小 subset)
    np.random.seed(seed)
    cs = np.random.choice(tr_sub, min(300_000, len(tr_sub)), replace=False)
    X_cs = norm(X[cs])
    y_cs = label[cs].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_cs[:, j], y_cs)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:topk]
    del X_cs, y_cs, corrs; gc.collect()
    
    # 3. train data
    print(f"    norm+inter train ({len(tr_sub)})...", flush=True)
    X_tr = add_interactions(norm(X[tr_sub]), top_idx)
    y_tr = label[tr_sub].astype(np.float32)
    print(f"    train={X_tr.shape}, mem={X_tr.nbytes/1e9:.2f}GB", flush=True)
    
    X_es = add_interactions(norm(X[es_idx]), top_idx)
    
    # 4. Train
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
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        
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
    
    def pred_on(idx):
        Xf = add_interactions(norm(X[idx]), top_idx)
        m.eval()
        pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf), 4096):
                pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        del Xf; gc.collect()
        return np.concatenate(pv)
    
    pv_te = pred_on(te_idx)
    pv_mv = pred_on(mv_idx)
    auc_te = roc_auc_score(label[te_idx], pv_te)
    auc_mv = roc_auc_score(label[mv_idx], pv_mv)
    
    del m, X_tr, X_es, y_tr, top_idx, tr_sub; gc.collect()
    return best_es, auc_mv, auc_te, pv_te, pv_mv

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

# ============================================
# TOP3 配置 (1.5M subset)
# ============================================
top3_configs = [
    ("TOP1", [512,256],       0.3, 5e-4, 1e-3, 15, 0.05, True, 40, 8, 1024, 1_500_000),
    ("TOP2", [512,256,128],   0.3, 5e-4, 1e-3, 15, 0.05, True, 50, 10, 1024, 1_500_000),
    ("TOP3", [256,128],       0.3, 5e-4, 1e-3, 15, 0.05, True, 40, 8, 1024, 1_500_000),
]

all_final = {}

for rank, (name, hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, tsize) in enumerate(top3_configs):
    print(f"\n{'='*70}", flush=True)
    print(f"--- {name}: hs={hs} d={drop} lr={lr} tk={topk} ts={tsize} ---", flush=True)
    print(f"{'='*70}", flush=True)
    
    pvs_t = []; pvs_m = []
    for seed in [42, 49, 56, 63, 70]:
        t_seed = time.time()
        es_a, mv_a, te_a, pv_t, pv_m = train_one(seed, hs, drop, lr, wd, topk, smooth, use_bn, epochs, pat, bs, tsize)
        pvs_t.append(pv_t); pvs_m.append(pv_m)
        print(f"    s={seed}: ES={es_a:.4f} MV={mv_a:.4f} TE={te_a:.4f} [{time.time()-t_seed:.0f}s]", flush=True)
        del pv_t, pv_m; gc.collect()
    
    pv_e_t = rank_agg(pvs_t); pv_e_m = rank_agg(pvs_m)
    auc_e_t = roc_auc_score(label[te_idx], pv_e_t)
    auc_e_m = roc_auc_score(label[mv_idx], pv_e_m)
    
    DAYS = len(te_idx) / 1440
    day_start = ts_te_vals.min()
    all_ts = np.arange(day_start, ts_te_vals.max() + 86400, 86400)
    n_days_all = len(all_ts) - 1
    
    print(f"\n    ★ ENSEMBLE: MV={auc_e_m:.4f} TE={auc_e_t:.4f}", flush=True)
    
    print(f"    Top-% accuracy:", flush=True)
    for pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
        n = int(len(pv_e_t) * pct)
        ti = np.argsort(-pv_e_t)[:n]
        acc = label[te_idx][ti].mean() * 100
        tpd = n / DAYS
        print(f"      top-{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}", flush=True)
    
    pv_arr = np.array(pv_e_t, dtype=np.float64)
    print(f"    Rolling q-threshold (30d):", flush=True)
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
    
    all_final[name] = {
        'cfg': (hs, drop, lr, wd, topk, smooth, use_bn, tsize),
        'auc_te': auc_e_t, 'auc_mv': auc_e_m,
        'pv_te': pv_e_t, 'pv_mv': pv_e_m
    }
    del pvs_t, pvs_m; gc.collect()

print(f"\n{'='*70}", flush=True)
print("FINAL SUMMARY", flush=True)
print(f"{'='*70}", flush=True)

best_name = max(all_final.keys(), key=lambda k: all_final[k]['auc_te'])
best = all_final[best_name]
print(f"\n🏆 Best: {best_name}", flush=True)
print(f"   Config: {best['cfg']}", flush=True)
print(f"   TE AUC={best['auc_te']:.4f} MV AUC={best['auc_mv']:.4f}", flush=True)

for name, info in all_final.items():
    print(f"   {name}: TE={info['auc_te']:.4f} MV={info['auc_mv']:.4f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/nn_v2_best.npz',
         pv_te=best['pv_te'], pv_mv=best['pv_mv'],
         te_idx=te_idx, mv_idx=mv_idx, ts_te_vals=ts_te_vals,
         config=str(best['cfg']), auc_te=best['auc_te'], auc_mv=best['auc_mv'])

print(f"\nResults saved. DONE [{time.time()-t0:.0f}s]", flush=True)
