"""ResMLP + DCN 堆叠分析"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config as cfg

t0 = time.time()
eth = pl.read_parquet(f'{cfg.DS_DIR}/raw_ETH.parquet')
btc = pl.read_parquet(f'{cfg.DS_DIR}/raw_BTC.parquet')
ts_e = eth['ts'].to_numpy(); C_e = eth['close'].to_numpy()
ts_b = btc['ts'].to_numpy(); C_b = btc['close'].to_numpy()
del eth, btc; gc.collect()
idx_b = np.searchsorted(ts_b, ts_e, side="right") - 1
idx_b = np.clip(idx_b, 0, len(C_b)-1)
B_lr1 = np.zeros(len(ts_e), dtype=np.float32)
B_lr1[1:] = np.log(np.maximum(C_b[idx_b[1:]],1e-8)/np.maximum(C_b[idx_b[:-1]],1e-8))
del ts_b, C_b; gc.collect()

eth_full = pl.read_parquet(f'{cfg.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe_mod
feats_df = fe_mod.build_features(eth_full)
del eth_full; gc.collect()
feats_np = feats_df.to_numpy().astype(np.float32)

H = 15
label = (C_e[H:] > C_e[:-H]).astype(np.int8); del C_e; gc.collect()
X_RAW = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()
X_RAW = np.nan_to_num(X_RAW, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()

FT = X_RAW.shape[1]
ts_u = ts_e[:-H].copy(); del ts_e; gc.collect()
tre = int(dtm.datetime.strptime(cfg.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(cfg.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(cfg.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
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
            out = b(x); x = x + out if x.shape == out.shape else out
        return self.head(x).squeeze(-1)

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

def train_quick(tr_sub, seed, model, smooth=0.05, epochs=40, pat=8, bs=1024, lr=5e-4, wd=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    np.random.seed(seed)
    cs = np.random.choice(tr_sub, min(300_000, len(tr_sub)), replace=False)
    X_cs = norm(X_RAW[cs]); y_cs = label[cs].astype(np.float32)
    corrs = np.array([abs(np.corrcoef(X_cs[:, j], y_cs)[0,1]) for j in range(FT)])
    top_idx = np.argsort(-corrs)[:15]
    del X_cs, y_cs, corrs; gc.collect()
    
    X_tr = make_inter(norm(X_RAW[tr_sub]), top_idx)
    y_tr = label[tr_sub].astype(np.float32)
    X_es = make_inter(norm(X_RAW[es_idx]), top_idx)
    gc.collect()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_es = 0.0; bst = None; ni = 0
    for e in range(epochs):
        model.train(); p = np.random.permutation(len(X_tr))
        for i in range(0, len(p), bs):
            bi = p[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr[bi].copy()))
            yb = torch.from_numpy(y_tr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
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
            if ni >= pat: break
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

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, 800_000, replace=False)

print("Training ResMLP [512,256]...", flush=True); tc = time.time()
m_res = ResMLP(188, [512,256], 0.3)
pv_res = train_quick(tr_sub, 42, m_res)
print(f"  ResMLP TE AUC={roc_auc_score(label[te_idx], pv_res):.4f} [{time.time()-tc:.0f}s]", flush=True)
del m_res; gc.collect()

print("Training DCN cross=3 [256,128]...", flush=True); tc = time.time()
m_dcn = DCN(188, 3, [256,128], 0.3)
pv_dcn = train_quick(tr_sub, 42, m_dcn)
print(f"  DCN TE AUC={roc_auc_score(label[te_idx], pv_dcn):.4f} [{time.time()-tc:.0f}s]", flush=True)
del m_dcn; gc.collect()

print(f"\n{'='*70}", flush=True)
print("Correlation + Stacking Analysis", flush=True)
print(f"{'='*70}", flush=True)
corr_all = np.corrcoef(pv_res, pv_dcn)[0,1]
print(f"Overall corr: {corr_all:.4f}", flush=True)

for pct in [0.005, 0.01, 0.02, 0.05, 0.1]:
    n = int(len(pv_res) * pct)
    t_res = set(np.argsort(-pv_res)[:n])
    t_dcn = set(np.argsort(-pv_dcn)[:n])
    inter = len(t_res & t_dcn)
    print(f"  top-{pct*100:.1f}%: overlap={inter}/{n} ({inter/n*100:.0f}%) jacc={inter/len(t_res|t_dcn):.3f}", flush=True)

print(f"\nStacking (ResMLP w + DCN 1-w):", flush=True)
for w_r in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pv_st = w_r * pv_res + (1 - w_r) * pv_dcn
    auc_st = roc_auc_score(label[te_idx], pv_st)
    n1 = int(len(pv_st) * 0.01)
    t1 = np.argsort(-pv_st)[:n1]
    top1 = label[te_idx][t1].mean() * 100
    print(f"  w_res={w_r}: AUC={auc_st:.4f} top1={top1:.1f}%", flush=True)

n1 = int(len(pv_res) * 0.01)
print(f"\n  ResMLP alone: top1={label[te_idx][np.argsort(-pv_res)[:n1]].mean()*100:.1f}%", flush=True)
print(f"  DCN alone: top1={label[te_idx][np.argsort(-pv_dcn)[:n1]].mean()*100:.1f}%", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
