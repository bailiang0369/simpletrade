"""金字塔分层切片 K线: 同结束点, 1min × 1min, 2min × 2min, 4min × 4min, 8min × 8min
比如 ts=14:32:00:
  L1(1min): [14:32, 14:31, 14:30, ..., 12:53] ← 100 根, 每步 1
  L2(2min): [14:32, 14:30, 14:28, ..., 11:12] ← 100 根, 每步 2
  L4(4min): [14:32, 14:28, 14:24, ..., 10:28] ← 100 根, 每步 4
  L8(8min): [14:32, 14:24, 14:16, ..., 04:48] ← 100 根, 每步 8
每层独立卷积, 最后 concat → fc
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'
print(f"torch={torch.__version__}", flush=True)

# ============= 1. 加载 raw 1min 并构建特征 =============
print("[1] Build 1min seq features (close_zscore + stoch_k14)...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
C = raw['close'].to_numpy().astype(np.float64)
H = raw['high'].to_numpy().astype(np.float64)
L = raw['low'].to_numpy().astype(np.float64)
rts = raw['ts'].to_numpy().astype(np.int64)
del raw; gc.collect()

# close zscore (4h rolling = 240 根)
rm = pd.Series(C).rolling(240, min_periods=60).mean().to_numpy()
rs = pd.Series(C).rolling(240, min_periods=60).std().to_numpy()
Cz = np.where(rs>1e-9, (C-rm)/rs, 0.0).astype(np.float32)

# stoch_k14 (1min)
L14 = pd.Series(L).rolling(14, min_periods=1).min().to_numpy()
H14 = pd.Series(H).rolling(14, min_periods=1).max().to_numpy()
Sk = np.where(H14-L14>1e-9, (C-L14)/(H14-L14)*100, 50.0).astype(np.float32)

print(f"  Raw 1min: {len(C):,} bars", flush=True)
del rm, rs, L14, H14; gc.collect()

# ============= 2. 对齐 ds ts =============
print("[2] Align ds timestamps...", flush=True)
ds = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
dts = ds['ts'].to_numpy().astype(np.int64)
labels = ds['label'].to_numpy().astype(np.int64)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
del ds; gc.collect()

# 找到每个 ds ts 在 1min raw 中的位置
idx = np.searchsorted(rts, dts)
# 安全: 最老的 ds ts 前面需要 8*100=800 根 → clip at 1000
idx = np.clip(idx, 1000, len(C)-1)
del rts; gc.collect()

def split_mask(lo, hi): return (dts>=lo)&(dts<hi)

tr_m = split_mask(0, TRAIN_END); es_m = split_mask(TRAIN_END, ES_END); te_m = split_mask(META_END, 10**18)
tr_idx = idx[tr_m]; tr_y = labels[tr_m]
es_idx = idx[es_m]; es_y = labels[es_m]
te_idx = idx[te_m]; te_y = labels[te_m]
del idx, dts, labels, tr_m, es_m, te_m; gc.collect()

# 下采样训练到 50 万 (CPU 限制)
np.random.seed(42)
sub = np.random.choice(len(tr_idx), min(500000, len(tr_idx)), replace=False)
tr_idx = tr_idx[sub]; tr_y = tr_y[sub]
print(f"  tr={len(tr_idx):,} (down)  es={len(es_idx):,}  te={len(te_idx):,}", flush=True)

# ============= 3. 金字塔数据集 =============
STEPS = [1, 2, 4, 8]  # 每层的步长 (bar)
W = 100               # 每层看 100 根 bar
print(f"\n[3] Pyramid: steps={STEPS}, W={W}", flush=True)
print(f"  e.g. idx=-1: L1=[-100..-1 step1] L2=[-200..-1 step2] ...", flush=True)

class PyramidDS(Dataset):
    """同结束点分层切片"""
    def __init__(self, idx, y, steps, window):
        self.idx = idx.astype(np.int64)
        self.y = y.astype(np.int64)
        self.steps = steps
        self.w = window
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        e = self.idx[i]           # 结束点 (同一根 1min bar)
        tensors = []
        for step in self.steps:
            start = e - (self.w - 1) * step
            sl = slice(start, e+1, step)  # 跳 step 根 → 恰好 W 根
            cz = Cz[sl]
            sk = Sk[sl] / 100.0   # 归一化到 0-1
            tensors.append(np.stack([cz, sk], axis=0).astype(np.float32))
        # (n_layers, 2, W) → flatten layers as batch dim → (2*n_layers, W)
        x = np.concatenate(tensors, axis=0)  # (8, 100) — 4 layers × 2 channels
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

# ============= 4. 金字塔 CNN =============
class PyramidCNN(nn.Module):
    def __init__(self, n_layers=4, in_ch_per_layer=2, h=24):
        super().__init__()
        self.n_layers = n_layers
        # 共享权重的单尺度 encoder (每层共用)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch_per_layer, h, 7, padding=3), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),   # 100→50
            nn.Conv1d(h, h, 5, padding=2),              nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),   # 50→25
            nn.Conv1d(h, h*2, 5, padding=2),           nn.BatchNorm1d(h*2), nn.SiLU(), nn.MaxPool1d(2),   # 25→12
            nn.Conv1d(h*2, h*2, 3, padding=1),          nn.BatchNorm1d(h*2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(
            nn.Linear(h*2*n_layers, 64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 2))
    def forward(self, x):
        # x: (B, n_layers*2, W) — 每层 2ch concat
        B = x.shape[0]
        # reshape to (B*n_layers, 2, W)
        x = x.view(B * self.n_layers, 2, -1)
        # encode each layer independently
        x = self.encoder(x)  # (B*n_layers, h*2, 1)
        x = x.view(B, self.n_layers, -1).flatten(1)  # (B, n_layers*h*2)
        return self.fc(x)

m = PyramidCNN(n_layers=len(STEPS)); tp = sum(p.numel() for p in m.parameters())
print(f"\n[4] PyramidCNN params={tp:,}", flush=True)

# ============= 5. Train =============
BATCH = 2048; EPOCHS = 15; LR = 1e-3
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader = DataLoader(PyramidDS(tr_idx, tr_y, STEPS, W), BATCH, shuffle=True, num_workers=0)
    es_loader = DataLoader(PyramidDS(es_idx, es_y, STEPS, W), BATCH*2, shuffle=False, num_workers=0)
    te_loader = DataLoader(PyramidDS(te_idx, te_y, STEPS, W), BATCH*2, shuffle=False, num_workers=0)
    m = PyramidCNN(len(STEPS))
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best = 0; best_state = None; t0 = time.time()
    for ep in range(1, EPOCHS+1):
        m.train(); tl = 0; n = 0
        for xb, yb in tr_loader:
            opt.zero_grad(); loss = ce(m(xb), yb); loss.backward(); opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        m.eval(); ps = []; ys = []
        with torch.no_grad():
            for xb, yb in es_loader:
                ps.append(F.softmax(m(xb), 1)[:,1].numpy())
                ys.append(yb.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        if auc > best: best = auc; best_state = {k:v.clone() for k,v in m.state_dict().items()}
        if ep % 5 == 0: print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.load_state_dict(best_state); m.eval(); ps = []
    with torch.no_grad():
        for xb, _ in te_loader: ps.append(F.softmax(m(xb), 1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[5] Train 3 seeds pyramid CNN (steps={STEPS}, W={W})...", flush=True)
pte = []
for sd in [42, 56, 70]:
    print(f"=== seed={sd} ===", flush=True)
    p, bes = run(sd); pte.append(p)
    print(f"  es_best={bes:.4f}", flush=True)

p = np.mean(pte, axis=0); auc = roc_auc_score(te_y, p)
print(f"\n{'='*60}")
print(f"  ★★ PYRAMID CNN (1,2,4,8 min, 同结束点分层切片, W=100)")
print(f"    TEST AUC = {auc:.4f}")
print(f"    对比:")
print(f"      LGBM ret+negw (2.4M, 5 seeds): AUC = 0.5428  ← 当前天花板")
print(f"      MultiScale Resample CNN (之前): AUC = 0.5405")
print(f"      SingleScale CNN (W=64):        AUC = 0.5363")
print(f"    {(auc-0.5428)*100:+.2f}pp vs LGBM  ({(auc-0.5405)*100:+.2f}pp vs Resample CNN)")
print(f"{'='*60}")

# tpd sweep
print(f"\n[6] tpd sweep...")
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'Δ65%':>8}")
best = None
for q in np.arange(0.985, 0.998, 0.0005):
    th = np.quantile(p, q); lm = p>th; sm = p<(1-th); tm = lm|sm; n = tm.sum()
    if n < 50: continue
    ss = sm[tm]; acc = (((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100
    tpd = n/332; diff = acc-65; mark = '◀' if abs(tpd-14.4)<0.5 else ''
    if 8 <= tpd <= 22: print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {diff:+7.1f}pp {mark}", flush=True)
    if abs(tpd-14.4)<0.5 and (best is None or acc>best[0]): best = (acc, q, tpd)
if best: print(f"\n★ tpd≈14.4: ACC={best[0]:.1f}%  (LGBM ≈61.6%)")

print(f"\nTOTAL: {time.time()-T0:.0f}s", flush=True)
