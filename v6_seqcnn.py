"""SeqCNN v4: numpy高级索引批量切片 - 极快!
channels保留在内存，TR+ES预生成float32，MV/TE用fast_slice批量eval
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm, config

torch.manual_seed(42); np.random.seed(42)
t0_all = time.time()

W = 128; HORIZON = 15

# ============================================
# Build channels (keep in memory)
# ============================================
print("[1] Building channels...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
C = eth['close'].to_numpy().astype(np.float64)
H = eth['high'].to_numpy().astype(np.float64)
L_arr = eth['low'].to_numpy().astype(np.float64)
ts = eth['ts'].to_numpy().astype(np.int64)
del eth; gc.collect()

N = len(C)

# Channel 1: Close rolling zscore (240min)
rm = np.convolve(C, np.ones(240)/240, mode='same')
close_z = np.zeros(N, dtype=np.float32)
for i in range(240, N):
    sd = np.std(C[max(0,i-240):i])
    if sd > 1e-9:
        close_z[i] = float((C[i] - rm[i]) / sd)
close_z = np.clip(close_z, -5, 5)
del rm; gc.collect()

# Channel 2: 1min return
close_r = np.zeros(N, dtype=np.float32)
close_r[1:] = (C[1:] - C[:-1]) / np.maximum(C[:-1], 1e-9)
close_r = np.clip(close_r, -0.05, 0.05)

# Channel 3: Stochastic %K (14-period)
stoch_k = np.zeros(N, dtype=np.float32)
for i in range(13, N):
    s = i - 13
    lo = np.min(L_arr[s:i+1])
    hi = np.max(H[s:i+1])
    if hi - lo > 1e-9:
        stoch_k[i] = float((C[i] - lo) / (hi - lo))
stoch_k = np.clip(stoch_k, 0, 1)
del L_arr, H; gc.collect()

# Label
label_all = (C[HORIZON:] > C[:-HORIZON]).astype(np.int64)
del C; gc.collect()

# Splits
min_end = W - 1
max_end = len(label_all) - 1
valid_end = np.arange(min_end, max_end + 1)
ts_label = ts[:-HORIZON]
del ts; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

ts_valid = ts_label[valid_end]
tr_idx = valid_end[ts_valid < tre]
es_idx = valid_end[(ts_valid >= tre) & (ts_valid < es_end)]
mv_idx = valid_end[(ts_valid >= es_end) & (ts_valid < meta_end)]
te_idx = valid_end[ts_valid >= meta_end]
ts_te_seq = ts_label[te_idx]
del ts_valid, valid_end, ts_label; gc.collect()

np.random.seed(42)
if len(tr_idx) > 200_000:
    tr_idx = np.random.choice(tr_idx, 200_000, replace=False)

y_tr = label_all[tr_idx]
y_es = label_all[es_idx]
y_mv = label_all[mv_idx]
y_te = label_all[te_idx]
del label_all; gc.collect()

print(f"  Split sizes: TR={len(tr_idx):,} ES={len(es_idx):,} MV={len(mv_idx):,} TE={len(te_idx):,}", flush=True)
print(f"  Label ratios: TR={y_tr.mean():.4f} TE={y_te.mean():.4f}", flush=True)

# ============================================
# Fast vectorized sequence slicing
# ============================================
def fast_slice(end_idx_batch, close_z, close_r, stoch_k, W):
    """Vectorized sequence slicing - much faster than Python for loop"""
    e = end_idx_batch
    s = e - W + 1
    offsets = np.arange(W)
    indices = s[:, None] + offsets[None, :]  # (bs, W)
    cz = close_z[indices]   # (bs, W)
    cr = close_r[indices]   # (bs, W)
    sk = stoch_k[indices]   # (bs, W)
    c0 = cz[:, 0:1]
    cn = (cz - c0) / (np.abs(c0) + 1e-6)
    return np.stack([cz, cn, cr, sk], axis=1).astype(np.float32)  # (bs, 4, W)

# ============================================
# Pre-generate TR + ES (for training)
# ============================================
print("\n[2] Pre-generating TR + ES sequences...", flush=True)
t_gen = time.time()
X_tr_seq = fast_slice(tr_idx, close_z, close_r, stoch_k, W)
print(f"  TR: {X_tr_seq.shape} [{time.time()-t_gen:.1f}s] mem={X_tr_seq.nbytes/1e6:.0f}MB", flush=True)
X_es_seq = fast_slice(es_idx, close_z, close_r, stoch_k, W)
print(f"  ES: {X_es_seq.shape} [{time.time()-t_gen:.1f}s] mem={X_es_seq.nbytes/1e6:.0f}MB", flush=True)
# Keep es_idx, mv_idx, te_idx for fast_eval; only del tr_idx
del tr_idx; gc.collect()
print(f"  Total [{time.time()-t0_all:.1f}s]", flush=True)

# ============================================
# Tiny CNN v3
# ============================================
class SeqCNN(nn.Module):
    def __init__(self, in_ch=4, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.MaxPool1d(2),  # 128 → 64
            nn.Conv1d(hidden, hidden*2, 5, padding=2), nn.BatchNorm1d(hidden*2), nn.GELU(),
            nn.Conv1d(hidden*2, hidden*2, 5, padding=2), nn.BatchNorm1d(hidden*2), nn.GELU(),
            nn.MaxPool1d(2),  # 64 → 32
            nn.Conv1d(hidden*2, hidden*3, 3, padding=1), nn.BatchNorm1d(hidden*3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden*3, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x).squeeze(-1)

def evaluate_fast(m, end_idx, close_z, close_r, stoch_k, W, bs=4096):
    m.eval(); pv = []; n = len(end_idx)
    with torch.no_grad():
        for i in range(0, n, bs):
            X = fast_slice(end_idx[i:i+bs], close_z, close_r, stoch_k, W)
            pv.append(torch.sigmoid(m(torch.from_numpy(X))).numpy())
    return np.concatenate(pv)

def train_seq(seed, Xtr, ytr, Xes, yes, ep=25, lr=1e-3, wd=1e-3, bs=1024, pat=8, smooth=0.05):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SeqCNN(hidden=48)
    np_ = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; best_ep = 0
    n = len(Xtr)
    for e in range(ep):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n, bs):
            bi = perm[i:i+bs]
            xb = torch.from_numpy(Xtr[bi].copy())
            yb = torch.from_numpy(ytr[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb); loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es = evaluate_fast(m, es_idx, close_z, close_r, stoch_k, W)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k: v.detach().clone() for k, v in m.state_dict().items()}; ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    pv_te = evaluate_fast(m, te_idx, close_z, close_r, stoch_k, W)
    pv_mv = evaluate_fast(m, mv_idx, close_z, close_r, stoch_k, W)
    return pv_te, pv_mv, best, np_, best_ep

# ============================================
# Train 5 seeds
# ============================================
print(f"\n{'='*60}\nSeqCNN v4 × 5 seeds\n{'='*60}", flush=True)

SEEDS = [42, 49, 56, 63, 70]
pvs_t = []; pvs_m = []
for seed in SEEDS:
    t0s = time.time()
    pv_t, pv_m, best_es, np_, bep = train_seq(seed, X_tr_seq, y_tr, X_es_seq, y_es,
                                               ep=25, lr=1e-3, wd=1e-3, bs=1024, pat=8, smooth=0.05)
    auc_t = roc_auc_score(y_te, pv_t); auc_m = roc_auc_score(y_mv, pv_m)
    print(f"  s={seed}: ep={bep} ES={best_es:.4f} MV={auc_m:.4f} TE={auc_t:.4f} p={np_:,} [{time.time()-t0s:.0f}s]", flush=True)
    pvs_t.append(pv_t); pvs_m.append(pv_m)
    del pv_t, pv_m; gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_cnn_t = rank_agg(pvs_t); pv_cnn_m = rank_agg(pvs_m)
auc_cnn_t = roc_auc_score(y_te, pv_cnn_t); auc_cnn_m = roc_auc_score(y_mv, pv_cnn_m)
print(f"\n★ SeqCNN ENSEMBLE: MV={auc_cnn_m:.4f} TE={auc_cnn_t:.4f}", flush=True)

DAYS = (ts_te_seq[-1] - ts_te_seq[0]) / 86400.0
for pct in [0.5, 1.0, 1.5]:
    kk = max(1, int(len(pv_cnn_t)*pct/100))
    acc = y_te[np.argsort(-pv_cnn_t)[:kk]].mean()*100
    tpd = kk/DAYS
    print(f"  top-{pct}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/v6_seqcnn.npz',
         pv_cnn_te=pv_cnn_t, pv_cnn_mv=pv_cnn_m,
         y_te=y_te, y_mv=y_mv, ts_te=ts_te_seq)
print(f"\nSaved v6_seqcnn.npz [{time.time()-t0_all:.0f}s]", flush=True)
