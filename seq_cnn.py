"""ETH 1min → H=30 direction (1D Dilated CNN + TA indicators)"""
import numpy as np, pandas as pd, time, gc, datetime as dtm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import ta

torch.manual_seed(42); np.random.seed(42)
torch.set_default_dtype(torch.float32)
def log(m): print(m,flush=True)
t0=time.time(); EPS=1e-12

SEG=64; STRIDE=16; H=30; BATCH=256; EPOCHS=30; LR=5e-2; CH=12
# CH = OHLC(4) + buyV/sellV/fund(3) + BTC(1) + RSI/MACD/Stoch/ATR(4)

log('[1] 数据...')
r=pd.read_parquet('data/datasets/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
b=pd.read_parquet('data/datasets/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
BTC=b[['ts','close']].rename(columns={'close':'BTC_close'})
r = pd.merge_asof(r.sort_values('ts'), BTC, on='ts', direction='backward')
n=len(r)
log(f'  n={n:,}')

# Tech indicators
log('[2] TA 指标...')
t=time.time()
r['rsi']   = ta.momentum.RSIIndicator(r['close'], 14).rsi()
r['macd']  = ta.trend.MACD(r['close']).macd_diff()
r['stoch'] = ta.momentum.StochasticOscillator(r['high'], r['low'], r['close']).stoch()
r['atr_n'] = ta.volatility.AverageTrueRange(r['high'], r['low'], r['close']).average_true_range() / r['close']
log(f'  {time.time()-t:.1f}s')

ts=r.ts.values.astype(np.int64)
O=r.open.values.astype(np.float32); Hv=r.high.values.astype(np.float32)
Lv=r.low.values.astype(np.float32); C=r.close.values.astype(np.float32)
buyV=r.buy_vol.values.astype(np.float32); sellV=r.sell_vol.values.astype(np.float32)
fund=r.funding.values.astype(np.float32); B=r.BTC_close.values.astype(np.float32)
RSI=r.rsi.values.astype(np.float32); MACD=r.macd.values.astype(np.float32)
STOCH=r.stoch.values.astype(np.float32); ATR_N=r.atr_n.values.astype(np.float32)
del r,b,BTC; gc.collect()

# Row-level z-score normalize
log('[3] row-level normalize...')
t=time.time()
def rolling_zscore(x, w=2880):  # 2 days
    s = pd.Series(x.astype(np.float64))
    mu = s.rolling(w, min_periods=2880).mean().values
    sd = s.rolling(w, min_periods=2880).std().values + 1e-6
    return ((x - mu) / sd).astype(np.float32)

buyV_z  = np.clip(rolling_zscore(buyV, 2880), -5, 5)
sellV_z = np.clip(rolling_zscore(sellV, 2880), -5, 5)
fund_s  = pd.Series(fund.astype(np.float64)).rolling(60, min_periods=60).mean()
fund_z  = np.clip(((fund_s - fund_s.mean()) / (fund_s.std() + 1e-6)).values, -5, 5).astype(np.float32)
rsi_z   = np.clip(rolling_zscore(RSI, 2880), -5, 5)
macd_z  = np.clip(rolling_zscore(MACD, 2880), -5, 5)
stoch_z = np.clip(rolling_zscore(STOCH, 2880), -5, 5)
atr_z   = np.clip(rolling_zscore(ATR_N, 2880), -5, 5)
del buyV, sellV, fund, RSI, MACD, STOCH, ATR_N, fund_s; gc.collect()

feat_cols = [
    O, Hv, Lv, C,                     # 4: OHLC (window-normalize)
    buyV_z, sellV_z, fund_z,           # 3: row-zscore
    B,                                  # 1: BTC close (window-normalize)
    rsi_z, macd_z, stoch_z, atr_z,     # 4: row-zscore
]
del O, Hv, Lv, C, buyV_z, sellV_z, fund_z, B, rsi_z, macd_z, stoch_z, atr_z
gc.collect()

feat_np = np.column_stack(feat_cols).astype(np.float32)
del feat_cols; gc.collect()
log(f'  feat_np={feat_np.shape} ({time.time()-t:.1f}s)')

row_valid = ~np.isnan(feat_np).any(axis=1)
_row_valid_int = row_valid.astype(np.int8)
_row_valid_cumsum = np.concatenate([[0], np.cumsum(_row_valid_int)])
del _row_valid_int; gc.collect()

retH = np.full(n, np.nan, np.float64)
C4label = feat_np[:, 3]
retH[:-H] = (C4label[H:] / (C4label[:-H] + EPS) - 1.0)
del C4label; gc.collect()

# ========= 4. unfold =========
log(f'[4] unfold SEG={SEG} STRIDE={STRIDE} ...')
t=time.time()
X_all = torch.from_numpy(feat_np).unfold(0, SEG, STRIDE)
N_total = X_all.shape[0]
del feat_np; gc.collect()

start_idx = np.arange(0, n - SEG + 1, STRIDE)
end_idx = start_idx + SEG
seg_valid = (_row_valid_cumsum[end_idx] - _row_valid_cumsum[start_idx]) == SEG
del _row_valid_cumsum, start_idx, end_idx; gc.collect()

log(f'  valid={int(seg_valid.sum()):,} total={N_total:,} ({time.time()-t:.0f}s)')

TR_end = int(dtm.datetime(2024,6,30,tzinfo=dtm.timezone.utc).timestamp())
ES_end = int(dtm.datetime(2024,9,30,tzinfo=dtm.timezone.utc).timestamp())
TE_end = int(dtm.datetime(2025,9,30,tzinfo=dtm.timezone.utc).timestamp())

def mask_window(mask, seg_valid):
    out = np.zeros(len(seg_valid), dtype=bool)
    out[mask & seg_valid] = True; return out

start_ts = ts[::STRIDE][:N_total]
tr_idx = mask_window(start_ts < TR_end, seg_valid)
es_idx = mask_window((start_ts >= TR_end) & (start_ts < ES_end), seg_valid)
te_idx = mask_window((start_ts >= ES_end) & (start_ts < TE_end), seg_valid)
del start_ts, ts, seg_valid; gc.collect()

# label: window 结束位置 j+SEG 的 H-period forward return
# label: window 结束位置 j+SEG 的 H-period forward return
# retH[i] = C[i+H]/C[i]-1, window j 结束在 row j+SEG, 所以 y[j] = retH[j+SEG]
y_all = (retH[SEG::STRIDE][:N_total] > 0).astype(np.float32)

# ========= 5. 切分 + normalize =========
log('[5] 切分 + window normalize...')
X_tr = X_all[tr_idx].clone(); y_tr = torch.from_numpy(y_all[tr_idx]).float()
X_es = X_all[es_idx].clone(); y_es = torch.from_numpy(y_all[es_idx]).float()
X_te = X_all[te_idx].clone(); y_te = torch.from_numpy(y_all[te_idx]).float()
del X_all; gc.collect()

def window_norm(X):
    c0 = X[:, 3:4, 0:1].clone(); btc0 = X[:, 7:8, 0:1].clone()
    for col in [0,1,2,3]:
        X[:, col:col+1, :] = torch.log(torch.clamp(X[:, col:col+1, :]/(c0+EPS), min=EPS)) * 100
    X[:, 7:8, :] = torch.log(torch.clamp(X[:, 7:8, :]/(btc0+EPS), min=EPS)) * 100
    return X

X_tr = window_norm(X_tr); X_es = window_norm(X_es); X_te = window_norm(X_te)

log(f'  TR={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}')
for i, cn in enumerate(['logO','logH','logL','logC','buyV','sellV','fund','logBTC','rsi','macd','stoch','atr']):
    c = X_tr[:, i, :].numpy().ravel()
    log(f'  {cn:6s}: mean={c.mean():+.4f} std={c.std():.4f} min={c.min():+.3f} max={c.max():+.3f}')

dl_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
dl_es = DataLoader(TensorDataset(X_es, y_es), batch_size=BATCH, shuffle=False)
dl_te = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH, shuffle=False)
del X_tr, y_tr; gc.collect()

# ========= 6. Dilated CNN =========
class DilatedCNN(nn.Module):
    def __init__(self, ch=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=2, dilation=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=8, dilation=8), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

m = DilatedCNN(CH)
log(f'\n[model] params={sum(p.numel() for p in m.parameters()):,}')

opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4, nesterov=True)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
best_auc=0; best_state=None; no_improve=0

def eval_auc(model, dl):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(torch.sigmoid(model(xb)).numpy()); ys.append(yb.numpy())
    p=np.concatenate(ps); y=np.concatenate(ys)
    return roc_auc_score(y,p), p, y

for ep in range(1, EPOCHS+1):
    t_ep=time.time(); m.train(); tl=0; tc=0; nb=0
    for xb, yb in dl_tr:
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(xb), yb)
        loss.backward(); opt.step()
        tl += loss.item()*xb.size(0)
        tc += ((m(xb)>0).float()==yb).sum().item()
        nb += xb.size(0)
    sch.step()
    es_auc,_,_ = eval_auc(m, dl_es)
    te_auc,_,_ = eval_auc(m, dl_te)
    log(f'  ep {ep}: loss={tl/nb:.4f} acc={tc/nb:.4f}  es={es_auc:.4f} te={te_auc:.4f}  {time.time()-t_ep:.0f}s')
    if es_auc > best_auc:
        best_auc = es_auc
        best_state = {k: v.clone() for k,v in m.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= 8:
            log(f'  early stop @ep{ep}'); break

if best_state: m.load_state_dict(best_state)
te_auc, te_p, te_y = eval_auc(m, dl_te)
log(f'\n[final] te_auc={te_auc:.4f}  best_es={best_auc:.4f}')

daily = 1440 // STRIDE
for pct in [1,2,3,5,8,10,15,20,25]:
    k = max(1, int(len(te_p)*pct/100))
    idx = np.argsort(-te_p)[:k]
    acc = te_y[idx].mean()*100
    tpd = k/daily
    log(f'  top{pct}%: acc={acc:.2f}% tpd={tpd:.1f}')

log(f'\n⏱ {time.time()-t0:.0f}s')
