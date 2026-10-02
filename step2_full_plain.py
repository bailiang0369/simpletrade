"""Step 2 FULL PLAIN: ETH NN, raw 68-dim feats, full 2.4M train, larger [512,256] model."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings, datetime as dtm
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import config, resource

torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

def memnow(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print("Loading ETH + features (f16)...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
import features as fe
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()
feats_np = feats.to_numpy().astype(np.float16)
del feats; gc.collect()

print("Loading BTC lr1 cross-asset...", flush=True)
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').select(['ts','close']).sort('ts')
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_ts) - 1)
B_lr1 = np.zeros(len(ts_all), dtype=np.float16)
B_lr1[1:] = np.log(np.maximum(BTC_C[idx[1:]], 1e-8) / np.maximum(BTC_C[idx[:-1]], 1e-8)).astype(np.float16)
del BTC_ts, BTC_C; gc.collect()

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
del C_all; gc.collect()
ts_u = ts_all[:-H]; del ts_all; gc.collect()

X_all = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis]], axis=1)
del feats_np, B_lr1; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_m = ts_u < tre
es_m = (ts_u >= tre) & (ts_u < es_end)
mv_m = (ts_u >= es_end) & (ts_u < meta_end)
te_m = ts_u >= meta_end
ts_te = ts_u[te_m]
del ts_u; gc.collect()

X_tr = X_all[tr_m].copy().astype(np.float32); y_tr = label[tr_m]
X_es = X_all[es_m].copy().astype(np.float32); y_es = label[es_m]
X_mv = X_all[mv_m].copy().astype(np.float32); y_mv = label[mv_m]
X_te = X_all[te_m].copy().astype(np.float32); y_te = label[te_m]
del X_all, label, tr_m, es_m, mv_m, te_m; gc.collect()

FT = X_tr.shape[1]
print(f"Split: tr={X_tr.shape} es={X_es.shape} mv={X_mv.shape} te={X_te.shape}", flush=True)
print(f"  RSS={memnow():.0f}MB", flush=True)

# Norm (f32 compute, store f32 - we have the budget)
print("Normalizing (f32)...", flush=True)
mu = np.zeros(FT, dtype=np.float32); sd = np.zeros(FT, dtype=np.float32)
for j in range(FT):
    col = X_tr[:, j].astype(np.float32); col[np.isnan(col)] = 0
    mu[j] = col.mean(); sd[j] = col.std() + 1e-6

for j in range(FT):
    X_tr[:, j] = np.clip((np.nan_to_num(X_tr[:, j], nan=0) - mu[j]) / sd[j], -5, 5)
    X_es[:, j] = np.clip((np.nan_to_num(X_es[:, j], nan=0) - mu[j]) / sd[j], -5, 5)
    X_mv[:, j] = np.clip((np.nan_to_num(X_mv[:, j], nan=0) - mu[j]) / sd[j], -5, 5)
    X_te[:, j] = np.clip((np.nan_to_num(X_te[:, j], nan=0) - mu[j]) / sd[j], -5, 5)
del mu, sd; gc.collect()

print(f"  After norm: RSS={memnow():.0f}MB  (tr={X_tr.nbytes/1e6:.0f}MB es={X_es.nbytes/1e6:.0f}MB)", flush=True)

# ---- Larger ResMLP, raw 68-dim ----
class ResMLP(nn.Module):
    def __init__(self, ft, hs, drop=0.3):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hs[i], hs[i + 1]), nn.LayerNorm(hs[i + 1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x) if x.shape == b(x).shape else b(x)
        return self.head(x).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs]))
            xb = torch.nan_to_num(xb, nan=0.0, posinf=5.0, neginf=-5.0)
            out = torch.sigmoid(m(xb))
            out = torch.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
            pv.append(out.numpy())
    result = np.concatenate(pv)
    return np.nan_to_num(result, nan=0.5)

def train(m, Xtr, ytr, Xes, yes, Xte, Xmv,
          ep=50, lr=5e-4, wd=1e-3, bs=2048, pat=12, smooth=0.05):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; best_ep = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi]))
            yb = torch.from_numpy(ytr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        auc_es = roc_auc_score(yes, evaluate(m, Xes))
        if auc_es > best + 1e-5:
            best = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    print(f"    best_ep={best_ep}/{ep}", flush=True)
    return evaluate(m, Xte), evaluate(m, Xmv)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

print(f"\nTraining ResMLP [{FT}→512→256→1] on FULL {len(X_tr):,} samples...", flush=True)
print(f"  RSS before train: {memnow():.0f}MB", flush=True)

pvs_t = []; pvs_m = []
for s in [42, 49, 56, 63, 70]:
    torch.manual_seed(s); np.random.seed(s)
    m = ResMLP(FT, [512, 256], 0.3)
    t0s = time.time()
    pv_t, pv_m = train(m, X_tr, y_tr, X_es, y_es, X_te, X_mv)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  s={s}: MV={auc_m:.4f} TE={auc_t:.4f} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_t.append(pv_t); pvs_m.append(pv_m)
    del m; gc.collect()

pv_eth_nn_te = rank_agg(pvs_t); pv_eth_nn_mv = rank_agg(pvs_m)
print(f"\n★ ETH NN-ENS (FULL, [512,256]): MV={roc_auc_score(y_mv, pv_eth_nn_mv):.4f} TE={roc_auc_score(y_te, pv_eth_nn_te):.4f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/eth_nn_full.npz',
         pv_eth_nn_te=pv_eth_nn_te, pv_eth_nn_mv=pv_eth_nn_mv,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"Saved eth_nn_full.npz [{time.time()-t0:.0f}s]", flush=True)
