"""
TCN: Temporal Convolutional Network for ETH 1min H=15 prediction
Input: past WINDOW minutes of raw OHLCV + funding
Output: sigmoid probability
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config as cfg

t0 = time.time()

# ============================================
# Data loading
# ============================================
print("Loading raw data...", flush=True)
eth_df = pl.read_parquet(f'{cfg.DS_DIR}/raw_ETH.parquet')

ts_arr = eth_df['ts'].to_numpy()
close_arr = eth_df['close'].to_numpy()
open_arr = eth_df['open'].to_numpy()
high_arr = eth_df['high'].to_numpy()
low_arr = eth_df['low'].to_numpy()
buyvol_arr = eth_df['buy_vol'].to_numpy()
sellvol_arr = eth_df['sell_vol'].to_numpy()
fund_arr = eth_df['funding'].to_numpy()
del eth_df; gc.collect()

# BTC cross
btc_df = pl.read_parquet(f'{cfg.DS_DIR}/raw_BTC.parquet')
ts_btc = btc_df['ts'].to_numpy()
close_btc = btc_df['close'].to_numpy()
del btc_df; gc.collect()
idx_btc = np.searchsorted(ts_btc, ts_arr, side="right") - 1
idx_btc = np.clip(idx_btc, 0, len(close_btc) - 1)
btc_lr = np.zeros(len(ts_arr), dtype=np.float32)
btc_lr[1:] = np.log(np.maximum(close_btc[idx_btc[1:]], 1e-8) / np.maximum(close_btc[idx_btc[:-1]], 1e-8))
del ts_btc, close_btc; gc.collect()

# Build per-bar features
HORIZON = 15
TOTAL = len(close_arr)

log_ret = np.zeros(TOTAL, dtype=np.float32)
log_ret[1:] = np.log(np.maximum(close_arr[1:], 1e-8) / np.maximum(close_arr[:-1], 1e-8))

body_r = (close_arr - open_arr) / (high_arr - low_arr + 1e-8)
range_r = (high_arr - low_arr) / (np.abs(log_ret) + 1e-6)
cvd_r = (buyvol_arr - sellvol_arr) / (buyvol_arr + sellvol_arr + 1e-8)

# Shape: (T, 6)
FEAT = np.stack([log_ret, body_r, range_r, cvd_r, fund_arr.astype(np.float32), btc_lr], axis=1)
del log_ret, body_r, range_r, cvd_r, btc_lr; gc.collect()
FEAT = np.nan_to_num(FEAT, nan=0.0, posinf=0.0, neginf=0.0)

# Label
LABEL = (close_arr[HORIZON:] > close_arr[:-HORIZON]).astype(np.int8)
del close_arr, open_arr, high_arr, low_arr, buyvol_arr, sellvol_arr, fund_arr; gc.collect()

T = len(LABEL)
N_CH = FEAT.shape[1]
print(f"FEAT={FEAT.shape}, LABEL={LABEL.shape}", flush=True)

# Splits (label[i] corresponds to feat at time i+HORIZON)
ts_label = ts_arr[:-HORIZON]
del ts_arr; gc.collect()

tre = int(dtm.datetime.strptime(cfg.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(cfg.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(cfg.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts_label < tre)[0]
es_idx = np.where((ts_label >= tre) & (ts_label < es_end))[0]
mv_idx = np.where((ts_label >= es_end) & (ts_label < meta_end))[0]
te_idx = np.where(ts_label >= meta_end)[0]
ts_te_vals = ts_label[te_idx].copy()
del ts_label; gc.collect()

print(f"splits: tr={len(tr_idx)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}", flush=True)

# Norm stats
np.random.seed(42)
samp = np.random.choice(tr_idx, min(500_000, len(tr_idx)), replace=False)
_FMU = FEAT[samp].mean(0)
_FSD = FEAT[samp].std(0) + 1e-6
del samp; gc.collect()

def _norm(arr):
    return np.clip((arr - _FMU) / _FSD, -5, 5)

# ============================================
# TCN Model
# ============================================
class Chomp1d(nn.Module):
    def __init__(self, cs): super().__init__(); self.cs = cs
    def forward(self, x): return x[:, :, :-self.cs].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, ks, dilation, drop=0.2):
        super().__init__()
        pad = (ks - 1) * dilation
        self.c1 = nn.Conv1d(ic, oc, ks, padding=pad, dilation=dilation)
        self.ch1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, ks, padding=pad, dilation=dilation)
        self.ch2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(oc)
        self.drp = nn.Dropout(drop)
        self.net = nn.Sequential(
            self.c1, self.ch1, self.bn1, nn.ReLU(), self.drp,
            self.c2, self.ch2, self.bn2, nn.ReLU(), self.drp)
        self.down = nn.Conv1d(ic, oc, 1) if ic != oc else None
    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_ch, channels, kernel=3, drop=0.2):
        super().__init__()
        layers = []
        prev = in_ch
        for li, ch in enumerate(channels):
            dil = 2 ** (li % 4)
            layers.append(TemporalBlock(prev, ch, kernel, dil, drop))
            prev = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)
    def forward(self, x):
        out = self.net(x)
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)

# ============================================
# Sequence builder + train
# ============================================
WINDOW = 60
min_valid = max(0, WINDOW - HORIZON)

print(f"norm FEAT...", flush=True)
FEAT_NORM = _norm(FEAT)
del FEAT; gc.collect()

def make_seqs(idx_arr):
    valid = idx_arr[idx_arr >= min_valid]
    N = len(valid)
    S = np.zeros((N, WINDOW, N_CH), dtype=np.float32)
    for k in range(WINDOW):
        S[:, k, :] = FEAT_NORM[valid + HORIZON - WINDOW + k]
    return S, LABEL[valid].astype(np.float32)

def train_one(tr_sub, seed, channels, kernel, drop, lr, wd, epochs, pat, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    
    print(f"  build seqs ({len(tr_sub)})...", flush=True)
    X_tr, y_tr = make_seqs(tr_sub)
    print(f"  X_tr={X_tr.shape}, mem={X_tr.nbytes/1e9:.2f}GB", flush=True)
    
    es_v = es_idx[es_idx >= min_valid]
    X_es, y_es = make_seqs(es_v)
    print(f"  X_es={X_es.shape}", flush=True)
    
    m = TCN(N_CH, channels, kernel, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    X_tr_t = torch.from_numpy(X_tr).float().permute(0, 2, 1).contiguous()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_es_t = torch.from_numpy(X_es).float().permute(0, 2, 1).contiguous()
    del X_tr, y_tr, X_es; gc.collect()
    
    best_es = 0.0; bst = None; ni = 0
    for e in range(epochs):
        m.train(); p = np.random.permutation(len(X_tr_t))
        for i in range(0, len(p), bs):
            bi = p[i:i+bs]
            xb = X_tr_t[bi]; yb = y_tr_t[bi]
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb * 0.95 + 0.025)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        
        m.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(X_es_t), 512):
                pv.append(torch.sigmoid(m(X_es_t[i:i+512])).numpy())
        auc_es = roc_auc_score(y_es, np.concatenate(pv))
        
        if auc_es > best_es + 1e-5:
            best_es = auc_es
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat: break
    
    if bst: m.load_state_dict(bst)
    del X_tr_t, y_tr_t, X_es_t; gc.collect()
    
    def pred_all(idx):
        valid = idx[idx >= min_valid]
        Xf, _ = make_seqs(valid)
        Xf_t = torch.from_numpy(Xf).float().permute(0, 2, 1).contiguous()
        m.eval(); pv=[]
        with torch.no_grad():
            for i in range(0, len(Xf_t), 512):
                pv.append(torch.sigmoid(m(Xf_t[i:i+512])).numpy())
        full = np.full(len(idx), 0.5)
        full[idx >= min_valid] = np.concatenate(pv)
        del Xf, Xf_t; gc.collect()
        return full
    
    pv_te = pred_all(te_idx)
    pv_mv = pred_all(mv_idx)
    auc_te = roc_auc_score(LABEL[te_idx], pv_te)
    auc_mv = roc_auc_score(LABEL[mv_idx], pv_mv)
    
    del m; gc.collect()
    return best_es, auc_mv, auc_te, pv_te, pv_mv

# ============================================
# Search
# ============================================
print(f"\n{'='*70}", flush=True)
print("TCN Search (800K, 单 seed)", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(800_000, len(tr_idx)), replace=False)

configs = [
    ("TCN1", [32, 64], 3, 0.2, 5e-4, 1e-3, 40, 8, 256),
    ("TCN2", [64, 128], 3, 0.3, 5e-4, 1e-3, 40, 8, 256),
    ("TCN3", [32, 32, 64], 3, 0.2, 5e-4, 1e-3, 50, 10, 256),
]

for name, ch, ks, dr, lr, wd, ep, pa, bs in configs:
    torch.manual_seed(42); np.random.seed(42)
    tc = time.time()
    es_a, mv_a, te_a, pv_t, _ = train_one(tr_sub, 42, ch, ks, dr, lr, wd, ep, pa, bs)
    elapsed = time.time() - tc
    n1 = int(len(pv_t) * 0.01)
    t1 = np.argsort(-pv_t)[:n1]
    top1 = LABEL[te_idx][t1].mean() * 100
    print(f"  {name}: TE={te_a:.4f} MV={mv_a:.4f} top1={top1:.1f}% | {elapsed:.0f}s", flush=True)
    del pv_t; gc.collect()

print(f"\nDONE [{time.time()-t0:.0f}s]", flush=True)
