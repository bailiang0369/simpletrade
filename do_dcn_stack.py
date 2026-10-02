"""Train DCN + stack with existing ResMLP pv"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

t0 = time.time()
print("Loading...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ts_e = eth['ts'].to_numpy(); C_e = eth['close'].to_numpy()
ts_b = btc['ts'].to_numpy(); C_b = btc['close'].to_numpy()
del eth, btc; gc.collect()
idx_b = np.searchsorted(ts_b, ts_e, side="right") - 1
idx_b = np.clip(idx_b, 0, len(C_b)-1)
B_lr1 = np.zeros(len(ts_e), dtype=np.float32)
B_lr1[1:] = np.log(np.maximum(C_b[idx_b[1:]],1e-8)/np.maximum(C_b[idx_b[:-1]],1e-8))
del ts_b, C_b; gc.collect()

eth_full = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe_mod
feats_df = fe_mod.build_features(eth_full); del eth_full; gc.collect()
feats_np = feats_df.to_numpy().astype(np.float32); del feats_df; gc.collect()

H = 15
label = (C_e[H:] > C_e[:-H]).astype(np.int8); del C_e; gc.collect()
X_RAW = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X_RAW = np.nan_to_num(X_RAW, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()

FT = X_RAW.shape[1]
ts_u = ts_e[:-H].copy(); del ts_e; gc.collect()
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
tr_idx = np.where(ts_u < tre)[0]
es_idx = np.where((ts_u >= tre) & (ts_u < es_end))[0]
mv_idx = np.where((ts_u >= es_end) & (ts_u < meta_end))[0]
te_idx = np.where(ts_u >= meta_end)[0]
del ts_u; gc.collect()

np.random.seed(42)
samp = np.random.choice(tr_idx, 300_000, replace=False)
MU = X_RAW[samp].mean(0); SD = X_RAW[samp].std(0) + 1e-6
del samp; gc.collect()
def norm(a): return np.clip((a-MU)/SD, -5, 5)
def make_inter(Xn, top_idx):
    X_sq = np.clip(Xn[:, top_idx] ** 2, 0, 25)
    X_cr = np.concatenate([Xn[:, top_idx[i:i+1]] * Xn[:, top_idx[j:j+1]]
                           for i in range(len(top_idx)) for j in range(i+1, len(top_idx))], axis=1)
    X_cr = np.clip(X_cr, -25, 25)
    return np.concatenate([Xn, X_sq, X_cr], axis=1)

class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d) * 0.01)
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x0, x): return x0 * (x @ self.w).unsqueeze(1) + self.b + x

class DCN(nn.Module):
    def __init__(self, ft, cross_n=3, deep=[256,128], drop=0.3):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(ft) for _ in range(cross_n)])
        dl = []; prev = ft
        for h in deep:
            dl.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        self.deep = nn.Sequential(*dl)
        self.head = nn.Linear(ft + prev, 1)
    def forward(self, x0):
        x_c = x0
        for cl in self.cross: x_c = cl(x0, x_c)
        x_d = self.deep(x0)
        return self.head(torch.cat([x_c, x_d], dim=1)).squeeze(-1)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    np.random.seed(seed)
    tr_sub = np.random.choice(tr_idx, min(1_500_000, len(tr_idx)), replace=False)
    cs = np.random.choice(tr_sub, min(300_000, len(tr_sub)), replace=False)
    X_cs = norm(X_RAW[cs]); y_cs = label[cs].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_cs[:, j], y_cs)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:15]
    del X_cs, y_cs, corrs, cs; gc.collect()
    
    X_tr = make_inter(norm(X_RAW[tr_sub]), top_idx)
    y_tr = label[tr_sub].astype(np.float32)
    X_es = make_inter(norm(X_RAW[es_idx]), top_idx)
    del tr_sub; gc.collect()
    
    model = DCN(188, 3, [256,128], 0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    best_es = 0.0; bst = None; ni = 0
    for e in range(40):
        model.train(); p = np.random.permutation(len(X_tr))
        for i in range(0, len(p), 1024):
            bi = p[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            yb = yb * 0.95 + 0.025
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(X_es), 4096):
                pv.append(torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(X_es[i:i+4096])))).numpy())
        auc_es = roc_auc_score(label[es_idx], np.concatenate(pv))
        if auc_es > best_es + 1e-5:
            best_es = auc_es; bst = {k: v.detach().clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
            if ni >= 8: break
    if bst: model.load_state_dict(bst)
    del X_tr, X_es, y_tr; gc.collect()
    
    def pred(idx):
        Xf = make_inter(norm(X_RAW[idx]), top_idx)
        model.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf), 4096):
                pv.append(torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(Xf[i:i+4096])))).numpy())
        del Xf; gc.collect()
        return np.concatenate(pv)
    return pred(te_idx)

# 5 seeds
pvs_dcn = []
for seed in [42, 49, 56, 63, 70]:
    tc = time.time()
    pv = train_seed(seed)
    auc = roc_auc_score(label[te_idx], pv)
    print(f"  s={seed}: TE={auc:.4f} [{time.time()-tc:.0f}s]", flush=True)
    pvs_dcn.append(pv)
    del pv; gc.collect()

pv_dcn_ens = rank_agg(pvs_dcn)
auc_dcn = roc_auc_score(label[te_idx], pv_dcn_ens)
print(f"  DCN ENSEMBLE TE AUC={auc_dcn:.4f}", flush=True)

# 加载 ResMLP
r = np.load('results/nn_v2_best.npz', allow_pickle=True)
pv_res_ens = r['pv_te']
auc_res = float(r['auc_te'])
print(f"  ResMLP ENSEMBLE TE AUC={auc_res:.4f}", flush=True)

# Stacking analysis
print(f"\n{'='*70}", flush=True)
print("STACKING ANALYSIS (ResMLP ensemble + DCN ensemble)", flush=True)
print(f"{'='*70}", flush=True)

DAYS = len(te_idx) / 1440
corr_all = np.corrcoef(pv_res_ens, pv_dcn_ens)[0,1]
print(f"Overall corr: {corr_all:.4f}", flush=True)

for pct in [0.005, 0.01, 0.02, 0.05]:
    n = int(len(pv_res_ens) * pct)
    t_r = set(np.argsort(-pv_res_ens)[:n])
    t_d = set(np.argsort(-pv_dcn_ens)[:n])
    inter = len(t_r & t_d)
    print(f"  top-{pct*100:.1f}%: overlap={inter}/{n} ({inter/n*100:.0f}%)", flush=True)

print(f"\nWeighted Stack:", flush=True)
best_w = 0.5; best_top1 = 0
for w_r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pv_st = w_r * pv_res_ens + (1 - w_r) * pv_dcn_ens
    auc_st = roc_auc_score(label[te_idx], pv_st)
    n1 = int(len(pv_st) * 0.01)
    t1 = np.argsort(-pv_st)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    if top1 > best_top1: best_top1 = top1; best_w = w_r
    print(f"  w_res={w_r}: AUC={auc_st:.4f} top1={top1:.1f}%", flush=True)

# Rank aggregation of ranks
print(f"\nRank Aggregation of ranks:", flush=True)
pv_res_r = np.argsort(np.argsort(pv_res_ens)).astype(np.float64) / (len(pv_res_ens)-1)
pv_dcn_r = np.argsort(np.argsort(pv_dcn_ens)).astype(np.float64) / (len(pv_dcn_ens)-1)
pv_rank = (pv_res_r + pv_dcn_r) / 2
auc_rank = roc_auc_score(label[te_idx], pv_rank)
n1 = int(len(pv_rank) * 0.01)
top1_rank = label[te_idx][np.argsort(-pv_rank)[:n1]].mean() * 100
print(f"  rank_agg: AUC={auc_rank:.4f} top1={top1_rank:.1f}%", flush=True)

print(f"\nBaselines:", flush=True)
n1 = int(len(pv_res_ens) * 0.01)
print(f"  ResMLP alone: AUC={auc_res:.4f} top1={label[te_idx][np.argsort(-pv_res_ens)[:n1]].mean()*100:.1f}%", flush=True)
print(f"  DCN alone: AUC={auc_dcn:.4f} top1={label[te_idx][np.argsort(-pv_dcn_ens)[:n1]].mean()*100:.1f}%", flush=True)

# Save
np.savez('results/nn_v3_dcn.npz', pv_dcn=pv_dcn_ens, pv_res=pv_res_ens, label=label[te_idx], te_idx=te_idx)

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
