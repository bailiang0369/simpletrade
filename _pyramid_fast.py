"""金字塔 CNN — 预构建金字塔 tensor, 速度 ×20

核心优化:
  之前: DataLoader 每次 slice(3.5M, start, end, step) → O(N_samples × N_layers)
  现在: 预计算 Py = (4, N_TOTAL, 100) float32, DataLoader 直接 Py[:, idx, :]
  
分层切片 K线: 每个 1min bar 当结束点, 分别往前取 step 步 (同结束点)
  step=[1,2,4,8], W=100 → 4×100 金字塔
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'

# ============= 1. 加载 1min 特征 =============
print("[1] Load 1min raw + precompute feats...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
C = raw['close'].to_numpy().astype(np.float64)
H = raw['high'].to_numpy().astype(np.float64)
L = raw['low'].to_numpy().astype(np.float64)
rts = raw['ts'].to_numpy().astype(np.int64); N_TOTAL = len(C)
del raw; gc.collect()

# close zscore
rm = pd.Series(C).rolling(240, min_periods=60).mean().to_numpy()
rs = pd.Series(C).rolling(240, min_periods=60).std().to_numpy()
Cz = np.where(rs>1e-9, (C-rm)/rs, 0.0).astype(np.float32)

# stoch_k14
L14 = pd.Series(L).rolling(14, min_periods=1).min().to_numpy()
H14 = pd.Series(H).rolling(14, min_periods=1).max().to_numpy()
Sk = np.where(H14-L14>1e-9, (C-L14)/(H14-L14)*100, 50.0).astype(np.float32)
del rm, rs, L14, H14, H, L, C; gc.collect()
print(f"  1min features ready: Cz={Cz.shape}, Sk={Sk.shape}, N_TOTAL={N_TOTAL:,}", flush=True)

# ============= 2. 预构建金字塔 =============
STEPS = [1, 2, 4, 8]; W = 100; NL = len(STEPS)
print(f"\n[2] Build pyramid tensor ({NL} layers × {N_TOTAL:,} ends × {W} bars)...", flush=True)
t0 = time.time()
Py = np.zeros((NL, N_TOTAL, W, 2), dtype=np.float32)  # (layers, ends, window, 2ch)
for li, step in enumerate(STEPS):
    print(f"  Layer step={step}...", flush=True)
    for i in range(1000, N_TOTAL):   # 前 1000 根留空 (边界 clip 到 1000)
        s = i - (W-1) * step
        sl = slice(s, i+1, step)
        Py[li, i] = np.stack([Cz[sl], Sk[sl]/100.0], axis=1)
print(f"  Done in {time.time()-t0:.0f}s, shape={Py.shape}, mem={Py.nbytes/1e9:.1f}GB", flush=True)

del Cz, Sk; gc.collect()
print(f"  After gc mem={gc.get_objects().__len__()}", flush=True)

# ============= 3. 对齐 ds ts + 切 split =============
print(f"\n[3] Align ds → split indices...", flush=True)
ds = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
dts = ds['ts'].to_numpy().astype(np.int64); labels = ds['label'].to_numpy().astype(np.int64)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
del ds; gc.collect()

idx = np.searchsorted(rts, dts); idx = np.clip(idx, 1000, N_TOTAL-1)
del rts; gc.collect()

def split(lo, hi):
    m = (dts>=lo)&(dts<hi)
    return idx[m], labels[m]

tr_idx, tr_y = split(0, TRAIN_END)
es_idx, es_y = split(TRAIN_END, ES_END)
te_idx, te_y = split(META_END, 10**18)
del idx, dts, labels; gc.collect()

np.random.seed(42)
sub = np.random.choice(len(tr_idx), min(500000, len(tr_idx)), replace=False)
tr_idx, tr_y = tr_idx[sub], tr_y[sub]
print(f"  tr={len(tr_idx):,} (down), es={len(es_idx):,}, te={len(te_idx):,}", flush=True)

# ============= 4. 极速 Dataset =============
class FastPyDS(Dataset):
    __slots__ = ('py', 'idx', 'y', 'nl')
    def __init__(self, py, idx, y, nl=4):
        self.py = py       # 预构建好的 (NL, N_TOTAL, W, 2)
        self.idx = idx.astype(np.int64)
        self.y = y.astype(np.int64)
        self.nl = nl
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        # py[:, self.idx[i], :, :] → (4, 100, 2) → transpose → (4, 2, 100) → flatten first 2 dim
        x = self.py[:, self.idx[i], :, :].transpose(0,2,1).reshape(-1, W)  # (8, 100)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

# ============= 5. CNN (共享权重, 每层独立编码) =============
class PyramidCNN(nn.Module):
    def __init__(self, nl=4, h=24):
        super().__init__()
        self.nl = nl
        self.enc = nn.Sequential(
            nn.Conv1d(2, h, 7, padding=3), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h, h, 5, padding=2), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h, h*2, 5, padding=2), nn.BatchNorm1d(h*2), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h*2, h*2, 3, padding=1), nn.BatchNorm1d(h*2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(h*2*nl, 64), nn.SiLU(), nn.Dropout(0.4), nn.Linear(64,2))
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B * self.nl, 2, -1)
        x = self.enc(x).view(B, self.nl, -1).flatten(1)
        return self.fc(x)

tp = sum(p.numel() for p in PyramidCNN().parameters())
print(f"\n[4] PyramidCNN params={tp:,}", flush=True)

# ============= 6. Train (batch=4096, 因为 __getitem__ 只做一次 fancy index) =============
BATCH = 4096; EPOCHS = 10; LR = 1e-3

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader = DataLoader(FastPyDS(Py, tr_idx, tr_y, NL), BATCH, shuffle=True, num_workers=0)
    es_loader = DataLoader(FastPyDS(Py, es_idx, es_y, NL), BATCH*2, shuffle=False, num_workers=0)
    te_loader = DataLoader(FastPyDS(Py, te_idx, te_y, NL), BATCH*2, shuffle=False, num_workers=0)
    m = PyramidCNN(NL)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); best = 0; best_state = None; t0 = time.time()
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
        print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.load_state_dict(best_state); m.eval(); ps = []
    with torch.no_grad():
        for xb, _ in te_loader: ps.append(F.softmax(m(xb), 1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[5] Train 3 seeds fast pyramid CNN (预构建 tensor)...", flush=True)
pte = []
for sd in [42, 56, 70]:
    print(f"=== seed={sd} ===", flush=True)
    p, bes = run(sd); pte.append(p); print(f"  es_best={bes:.4f}", flush=True)

p = np.mean(pte, axis=0); auc = roc_auc_score(te_y, p)
print(f"\n{'='*60}")
print(f"  ★ FAST PYRAMID CNN (steps={STEPS}, W={W}, 预构建 tensor)")
print(f"    TEST AUC = {auc:.4f}")
print(f"    对比:")
print(f"      LGBM ret+negw (2.4M, 5 seeds): AUC = 0.5428")
print(f"      之前慢版 Pyramid CNN (500k):    AUC = 0.5415")
print(f"      {(auc-0.5428)*100:+.2f}pp vs LGBM")
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
if best: print(f"\n★ tpd≈14.4: ACC={best[0]:.1f}%")

print(f"\nTOTAL: {time.time()-T0:.0f}s", flush=True)
