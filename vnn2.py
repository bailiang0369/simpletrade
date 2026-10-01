"""V-NN part 2: 补跑 E4-E7 + 全集成. 内存优化版."""
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

# Data (same as vnn_optimize)
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
if len(tr_idx) > 600_000:
    tr_idx = np.random.choice(tr_idx, 600_000, replace=False)

X_tr = X_all[tr_idx].copy()
X_es = X_all[es_mask].copy()
X_mv = X_all[mv_mask].copy()
X_te = X_all[te_mask].copy()
y_tr = label[tr_idx]; r_tr = ret_future[tr_idx]
y_es = label[es_mask]; y_mv = label[mv_mask]; y_te = label[te_mask]
ts_te = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

FT = X_tr.shape[1]
for j in range(FT):
    col = X_tr[:,j]; col[np.isnan(col)] = 0
    lo, hi = np.percentile(col, 0.5), np.percentile(col, 99.5)
    m, s = col.mean(), col.std() + 1e-6
    X_tr[:,j] = np.clip((col - m) / s, -5, 5)
    X_es[:,j] = np.clip((np.nan_to_num(X_es[:,j], nan=0) - m) / s, -5, 5)
    X_mv[:,j] = np.clip((np.nan_to_num(X_mv[:,j], nan=0) - m) / s, -5, 5)
    X_te[:,j] = np.clip((np.nan_to_num(X_te[:,j], nan=0) - m) / s, -5, 5)
gc.collect()

corrs = np.array([abs(np.corrcoef(X_tr[:,j], y_tr)[0,1]) for j in range(FT)])
top20 = np.argsort(-corrs)[:20]

def add_interactions(X, topk):
    X_sq = X[:, topk] ** 2
    X_cross = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            X_cross.append((X[:, topk[i]] * X[:, topk[j]])[:, np.newaxis])
    return np.concatenate([X, X_sq, np.concatenate(X_cross, axis=1)], axis=1)

X_tr_nn = add_interactions(X_tr, top20)
X_es_nn = add_interactions(X_es, top20)
X_mv_nn = add_interactions(X_mv, top20)
X_te_nn = add_interactions(X_te, top20)
del X_tr, X_es, X_mv, X_te; gc.collect()
FT_NN = X_tr_nn.shape[1]
print(f"  TR={X_tr_nn.shape} TE={X_te_nn.shape} FT={FT_NN}", flush=True)

# Models
class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

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

class TCN1D(nn.Module):
    def __init__(self, C=1, chs=[32,64,128,64], drop=0.3):
        super().__init__()
        prev=C; d=1; blocks=[]
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev,c,3,padding=2*d,dilation=d), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c,c,3,padding=2*d,dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            prev=c; d*=2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))
    def forward(self, x):
        x = x.unsqueeze(1)
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

def train_adamw(m, Xtr, ytr, Xes, yes, ep=40, lr=5e-4, wd=1e-3, bs=1024, pat=8, smooth=0.05):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best=0.0; bst=None; ni=0; best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr))
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        auc_es = roc_auc_score(yes, evaluate(m, Xes))
        if auc_es > best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return evaluate(m, X_te_nn), evaluate(m, X_mv_nn)

def train_sgd_cos(m, Xtr, ytr, Xes, yes, ep=50, lr=1e-2, wd=5e-3, bs=1024, pat=10, smooth=0.05):
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    best=0.0; bst=None; ni=0; best_ep=0
    for e in range(ep):
        m.train(); idx=np.random.permutation(len(Xtr))
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            logits=m(xb); loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        auc_es = roc_auc_score(yes, evaluate(m, Xes))
        if auc_es > best+1e-5:
            best=auc_es; bst={k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0; best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return evaluate(m, X_te_nn), evaluate(m, X_mv_nn)

# ============================================
# 补跑 E4-E7 + ResMLP 更多 seed
# ============================================
all_nn_te = []; all_nn_mv = []

def run_nn_exp(label, model_fn, seeds, train_fn, **train_kw):
    global all_nn_te, all_nn_mv
    print(f"\n{'='*50}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*50}", flush=True)
    pvs_t = []; pvs_m = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = model_fn()
        np_ = sum(p.numel() for p in m.parameters())
        t0s = time.time()
        pv_t, pv_m = train_fn(m, X_tr_nn, y_tr, X_es_nn, y_es, **train_kw)
        auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
        print(f"    s={s}: MV={auc_m:.4f} TE={auc_t:.4f} p={np_:,} [{time.time()-t0s:.0f}s]", flush=True)
        pvs_t.append(pv_t); pvs_m.append(pv_m); del m; gc.collect()
    pv_ens_t = rank_agg(pvs_t); pv_ens_m = rank_agg(pvs_m)
    auc_ens_t = roc_auc_score(y_te, pv_ens_t); auc_ens_m = roc_auc_score(y_mv, pv_ens_m)
    print(f"    ★ ENS: MV={auc_ens_m:.4f} TE={auc_ens_t:.4f}", flush=True)
    all_nn_te.append(pv_ens_t); all_nn_mv.append(pv_ens_m)
    return auc_ens_t

# E3b: ResMLP 更多 seed (7 seeds)
run_nn_exp("E3b: ResMLP [256,128] 7seeds AdamW d0.3 wd1e-3",
           lambda: ResMLP(FT_NN, [256,128], 0.3),
           [42, 49, 56, 63, 70, 115, 122], train_adamw,
           ep=30, lr=5e-4, wd=1e-3, bs=1024, pat=6, smooth=0.05)

# E4: TCN1D 小 chs (防 OOM)
run_nn_exp("E4: TCN1D small chs[16,32,64,32] d0.3",
           lambda: TCN1D(1, [16,32,64,32], 0.3),
           [42, 56, 70], train_adamw,
           ep=40, lr=3e-3, wd=1e-3, bs=512, pat=10, smooth=0.05)

# E5: SGD+Cosine ResMLP
run_nn_exp("E5: ResMLP [256,128] SGD+Cosine lr=0.01",
           lambda: ResMLP(FT_NN, [256,128], 0.3),
           [42, 56, 70], train_sgd_cos,
           ep=50, lr=1e-2, wd=5e-3, bs=1024, pat=10, smooth=0.05)

# E6: Wider ResMLP [512,256]
run_nn_exp("E6: ResMLP [512,256] AdamW d0.3 wd3e-3",
           lambda: ResMLP(FT_NN, [512,256], 0.3),
           [42, 56, 70], train_adamw,
           ep=30, lr=5e-4, wd=3e-3, bs=1024, pat=6, smooth=0.05)

# E7: ResMLP deep [512,256,256]
run_nn_exp("E7: ResMLP [512,256,256] AdamW d0.25 wd1e-4",
           lambda: ResMLP(FT_NN, [512,256,256], 0.25),
           [42, 56, 70], train_adamw,
           ep=35, lr=8e-4, wd=1e-4, bs=1024, pat=8, smooth=0.05)

# E8: MLP [256,128] SGD+Cosine
run_nn_exp("E8: MLP [256,128] SGD+Cosine lr=0.01",
           lambda: MLP(FT_NN, [256,128], 0.3),
           [42, 56, 70], train_sgd_cos,
           ep=50, lr=1e-2, wd=5e-3, bs=1024, pat=10, smooth=0.05)

# ============================================
# 集成
# ============================================
print(f"\n{'='*60}", flush=True)
print("NN-ENS: rank agg of all", flush=True)
print(f"{'='*60}", flush=True)

pv_nn_ens_te = rank_agg(all_nn_te)
pv_nn_ens_mv = rank_agg(all_nn_mv)
auc_nn_ens_te = roc_auc_score(y_te, pv_nn_ens_te)
auc_nn_ens_mv = roc_auc_score(y_mv, pv_nn_ens_mv)
print(f"  ★ NN-ENS ALL: MV={auc_nn_ens_mv:.4f} TE={auc_nn_ens_te:.4f}", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

def topk_eval(name, pv, y, days):
    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/days
        flag = '🏆' if pct==1.0 and acc>=62 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

topk_eval("NN-ENS ALL", pv_nn_ens_te, y_te, DAYS)

for i, (pv_t, pv_m) in enumerate(zip(all_nn_te, all_nn_mv)):
    auc_t = roc_auc_score(y_te, pv_t)
    auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  E{i+1}: MV={auc_m:.4f} TE={auc_t:.4f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/vnn2.npz',
         pv_nn_ens_te=pv_nn_ens_te, pv_nn_ens_mv=pv_nn_ens_mv,
         all_nn_te=np.array(all_nn_te), all_nn_mv=np.array(all_nn_mv),
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved vnn2.npz", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"V-NN2 DONE [{time.time()-t0_all:.0f}s]", flush=True)
print(f"{'='*60}", flush=True)
