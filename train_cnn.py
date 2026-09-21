"""1D Dilated CNN on raw OHLCV windows — no hand-engineering!"""""
import numpy as np, pandas as pd, time, gc, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

t0 = time.time()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {DEVICE}', flush=True)

# ===== Build raw sequence data =====
print('[1] load raw data...', flush=True)
eth = pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
btc = pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')

# Align BTC to ETH
btc_idx = pd.Index(btc['ts'].values); eth_idx = pd.Index(eth['ts'].values)
reorder = btc_idx.get_indexer(eth_idx)
reorder = np.where(reorder < 0, np.arange(len(reorder)), reorder)

# 8 channels: OHLC, log(buy/sell), log(vol), funding_norm, btc_ret
N = len(eth)
close = eth['close'].values.astype(np.float32)
ret1 = np.full(N, np.nan, np.float32); ret1[1:] = close[1:]/close[:-1] - 1
# Normalized OHLC (window-relative) — we'll normalize per-window

feats = np.zeros((N, 8), np.float32)
feats[:, 0] = eth['open'].values.astype(np.float32)
feats[:, 1] = eth['high'].values.astype(np.float32)
feats[:, 2] = eth['low'].values.astype(np.float32)
feats[:, 3] = close
bv = eth['buy_vol'].values.astype(np.float32); sv = eth['sell_vol'].values.astype(np.float32)
tot = np.maximum(bv + sv, 1)
feats[:, 4] = np.log(np.maximum(bv / tot, 1e-6))  # log(buy_ratio)
feats[:, 5] = np.log(np.maximum(tot, 1e-6))      # log(volume)
feats[:, 6] = eth['funding'].values.astype(np.float32)
# BTC close return
btc_c = btc['close'].values.astype(np.float32)[reorder]
btc_r = np.full(N, np.nan, np.float32); btc_r[1:] = btc_c[1:]/btc_c[:-1] - 1
feats[:, 7] = np.nan_to_num(btc_r, nan=0.0)

print(f'  feats: {feats.shape}')

# ===== Build windows (CNN) =====
print('\n[2] build windows...', flush=True)
SEG = 64; STRIDE = 16
# Window-level log normalization
# shape: (N, SEG, 8) → transpose to (N, 8, SEG)
# For CNN we do (N, channels, seq_len)
def build_windows(arr, seg, stride):
    N = arr.shape[0]
    out = []
    for start in range(0, N - seg, stride):
        w = arr[start:start+seg]
        out.append(w)
    return np.array(out, np.float32)

# Build target labels (H=30 after window end)
H = 30
labels = np.full(N, np.nan, np.float32)
labels[:-H] = (close[H:]/close[:-H] - 1).astype(np.float32)

# Build windows with normalization
TOTAL_SEG = SEG
# Pre-compute all windows
rows = []; targets = []; ts_vals = []
for start in range(0, N - TOTAL_SEG - H, STRIDE):
    end = start + TOTAL_SEG
    if np.isnan(labels[end]): continue
    w = feats[start:end].copy()
    # Normalize: log-change relative to window start close
    c0 = close[start]
    if c0 <= 0: continue
    w[:, 0:4] = np.log(np.maximum(w[:, 0:4], 1e-9) / c0)  # OHLC
    # Funding: z-score per window
    fund_m = w[:, 6].mean(); fund_s = max(w[:, 6].std(), 1e-6)
    w[:, 6] = (w[:, 6] - fund_m) / fund_s
    rows.append(w.transpose())  # (8, 64)
    targets.append(1 if labels[end] > 0 else 0)
    ts_vals.append(eth['ts'].values[end])

X_all = np.array(rows, np.float32)
y_all = np.array(targets, np.int64)
ts_all = np.array(ts_vals, np.int64)
del rows, feats; gc.collect()
print(f'  windows: {X_all.shape}  labels: pos={y_all.mean():.3f}')

# ===== Split (strict time) =====
def ts_mask(s,e):
    a=int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b=int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all>=a)&(ts_all<b)
import datetime as dtm
tr_m = ts_mask('2020-01-01','2024-06-30')
es_m = ts_mask('2024-06-30','2024-09-30')
te_m = ts_mask('2025-09-30','2026-08-29')
tr_idx = np.where(tr_m)[0]; es_idx = np.where(es_m)[0]; te_idx = np.where(te_m)[0]
print(f'  TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,}')

# ===== CNN Model =====
print('\n[3] define CNN...', flush=True)
class DilatedCNN(nn.Module):
    def __init__(self, ch_in=8, seg=64):
        super().__init__()
        self.stem = nn.Conv1d(ch_in, 64, 3, padding=1)
        self.b1 = nn.Sequential(nn.Conv1d(64, 128, 3, dilation=2, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(128, 128, 3, dilation=4, padding=4), nn.BatchNorm1d(128), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(128, 128, 3, dilation=8, padding=8), nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

m = DilatedCNN(ch_in=8).to(DEVICE)
total_params = sum(p.numel() for p in m.parameters())
print(f'  params: {total_params:,}')

# ===== Train =====
print('\n[4] train...', flush=True)
BATCH = 512
EPOCHS = 15
LR = 3e-4

X_tr = torch.from_numpy(X_all[tr_idx])
y_tr_t = torch.from_numpy(y_all[tr_idx])
X_es = torch.from_numpy(X_all[es_idx])
y_es_t = torch.from_numpy(y_all[es_idx])
X_te = torch.from_numpy(X_all[te_idx])
y_te_t = torch.from_numpy(y_all[te_idx])

train_dl = DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=BATCH, shuffle=True, drop_last=False)
del X_all, y_all; gc.collect()

opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
best_es_auc = 0; best_state = None
early_stop_patience = 3; no_improve = 0

for ep in range(EPOCHS):
    m.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        logits = m(xb.to(DEVICE))
        loss = loss_fn(logits, yb.float().to(DEVICE))
        loss.backward()
        opt.step()

    # Eval
    m.eval()
    with torch.no_grad():
        pv_es = torch.sigmoid(m(X_es.to(DEVICE))).cpu().numpy()
        pv_te = torch.sigmoid(m(X_te.to(DEVICE))).cpu().numpy()
    auc_es = roc_auc_score(y_all[es_idx], pv_es)
    auc_te = roc_auc_score(y_all[te_idx], pv_te)

    if auc_es > best_es_auc:
        best_es_auc = auc_es; best_state = {k: v.clone() for k,v in m.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    top1 = pv_te.argsort()[-max(1,int(len(pv_te)*0.01)):]
    acc_top1 = y_all[te_idx][top1].mean()*100
    print(f'  ep{ep+1}: es_auc={auc_es:.4f} te_auc={auc_te:.4f} top1%={acc_top1:.1f}%  ({time.time()-t0:.0f}s)', flush=True)

    if no_improve >= early_stop_patience:
        print(f'  early stop at ep{ep+1}', flush=True)
        break

if best_state:
    m.load_state_dict(best_state)

# ===== EVAL FINAL =====
print('\n[5] FINAL EVAL...', flush=True)
m.eval()
with torch.no_grad():
    pv_te = torch.sigmoid(m(X_te.to(DEVICE))).cpu().numpy()
y_te = y_all[te_idx]
auc = roc_auc_score(y_te, pv_te)
print(f'  best te_auc={auc:.4f}')

daily = 1440
for pct in [0.5, 1, 2, 3, 5, 8, 10]:
    k = max(1, int(len(pv_te)*pct/100))
    idx = np.argsort(-pv_te)[:k]
    acc = y_te[idx].mean()*100; tpd = k*daily/len(pv_te)
    print(f'  top{pct:>3}%: acc={acc:.1f}%  tpd={tpd:.0f}')

# Monthly stability
dt_te = pd.to_datetime(ts_all[te_idx], unit='s', utc=True)
month = dt_te.to_period('M').values
all_months = sorted(pd.PeriodIndex(np.unique(month)))
bad = 0
for mm in all_months:
    hm = month == mm
    n = hm.sum(); k = max(1, int(n*0.01))
    acc_m = y_te[hm][np.argsort(-pv_te[hm])[:k]].mean()*100
    auc_m = roc_auc_score(y_te[hm], pv_te[hm])
    flag = 'BAD' if acc_m < 60 else '   '
    if acc_m < 60: bad += 1
    print(f'  {mm} n={n:>6,} AUC={auc_m:.4f} top1%={acc_m:>5.1f}% {flag}')
print(f'  BAD months: {bad}/{len(all_months)}')

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
