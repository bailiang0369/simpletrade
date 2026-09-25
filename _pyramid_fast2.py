"""金字塔 CNN — 每个 split 独立预构建 tensor, 避免 OOM, 速度 ×20
Py 只存 split 内用到的 idx, 不是全量 3.5M
"""
import numpy as np, pandas as pd, time, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); T0=time.time(); DEVICE='cpu'
print(f"DEVICE={DEVICE}")

STEPS = [1, 2, 4, 8]; W = 100; NL = len(STEPS)

# ============= 1. 加载 1min 特征 + split idx =============
print("[1] Load 1min raw + split idx...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
C = raw['close'].to_numpy().astype(np.float64); H = raw['high'].to_numpy().astype(np.float64)
L = raw['low'].to_numpy().astype(np.float64); rts = raw['ts'].to_numpy().astype(np.int64); N = len(C)

rm = pd.Series(C).rolling(240, min_periods=60).mean().to_numpy()
rs = pd.Series(C).rolling(240, min_periods=60).std().to_numpy()
Cz = np.where(rs>1e-9, (C-rm)/rs, 0.0).astype(np.float32)
L14 = pd.Series(L).rolling(14, min_periods=1).min().to_numpy()
H14 = pd.Series(H).rolling(14, min_periods=1).max().to_numpy()
Sk = np.where(H14-L14>1e-9, (C-L14)/(H14-L14)*100, 50.0).astype(np.float32)
del raw, rm, rs, L14, H14, H, L, C; gc.collect()

ds = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label']).sort_values('ts')
dts = ds['ts'].to_numpy().astype(np.int64); labels = ds['label'].to_numpy().astype(np.int64)
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
del ds; gc.collect()
idx = np.searchsorted(rts, dts); idx = np.clip(idx, 1000, N-1)
del rts; gc.collect()

def split(lo, hi):
    m=(dts>=lo)&(dts<hi)
    return idx[m], labels[m]

tr_idx, tr_y = split(0, TRAIN_END); es_idx, es_y = split(TRAIN_END, ES_END); te_idx, te_y = split(META_END, 10**18)
del idx, dts, labels; gc.collect()
np.random.seed(42)
sub = np.random.choice(len(tr_idx), min(500000, len(tr_idx)), replace=False)
tr_idx, tr_y = tr_idx[sub], tr_y[sub]
print(f"  tr={len(tr_idx):,}, es={len(es_idx):,}, te={len(te_idx):,}", flush=True)

# ============= 2. 预构建金字塔 (只为 split 内的 idx) =============
def build_pyramid(idx_arr):
    """idx_arr: (N,) → return (N, NL, 2, W) float32"""
    Ns = len(idx_arr)
    out = np.zeros((Ns, NL, 2, W), dtype=np.float32)
    for li, step in enumerate(STEPS):
        s_arr = idx_arr - (W-1)*step    # 每个 idx 的起点
        for j in range(Ns):
            out[j, li, 0] = Cz[s_arr[j]:idx_arr[j]+1:step]   # zscore
            out[j, li, 1] = Sk[s_arr[j]:idx_arr[j]+1:step] / 100.0
    return out

print(f"\n[2] Build pyramid tensors...", flush=True)
t0 = time.time()
Py_tr = build_pyramid(tr_idx); print(f"  tr: {Py_tr.shape}  ({time.time()-t0:.0f}s)", flush=True); t0=time.time()
Py_es = build_pyramid(es_idx); print(f"  es: {Py_es.shape}  ({time.time()-t0:.0f}s)", flush=True); t0=time.time()
Py_te = build_pyramid(te_idx); print(f"  te: {Py_te.shape}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  Total build time={time.time()-t0:.0f}s", flush=True)
del Cz, Sk; gc.collect()

# ============= 3. Dataset + CNN =============
class FastPyDS(Dataset):
    def __init__(self, py, y): self.py=py; self.y=y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return torch.from_numpy(self.py[i]), torch.tensor(self.y[i], dtype=torch.long)

class PyramidCNN(nn.Module):
    def __init__(self, nl=4, h=24):
        super().__init__(); self.nl=nl
        self.enc=nn.Sequential(
            nn.Conv1d(2,h,7,padding=3), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h,h,5,padding=2), nn.BatchNorm1d(h), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h,h*2,5,padding=2), nn.BatchNorm1d(h*2), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(h*2,h*2,3,padding=1), nn.BatchNorm1d(h*2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Sequential(nn.Linear(h*2*nl,64),nn.SiLU(),nn.Dropout(0.4),nn.Linear(64,2))
    def forward(self,x):
        B=x.shape[0]; x=x.view(B*self.nl,2,-1)
        return self.fc(self.enc(x).view(B,self.nl,-1).flatten(1))

tp = sum(p.numel() for p in PyramidCNN().parameters())
print(f"\n[3] PyramidCNN params={tp:,}", flush=True)

# ============= 4. Train =============
BATCH=4096; EPOCHS=10; LR=1e-3
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_l = DataLoader(FastPyDS(Py_tr, tr_y), BATCH, shuffle=True, num_workers=0)
    es_l = DataLoader(FastPyDS(Py_es, es_y), BATCH*2, shuffle=False, num_workers=0)
    te_l = DataLoader(FastPyDS(Py_te, te_y), BATCH*2, shuffle=False, num_workers=0)
    m = PyramidCNN(NL)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); best=0; best_state=None; t0=time.time()
    for ep in range(1, EPOCHS+1):
        m.train(); tl=0; n=0
        for xb, yb in tr_l:
            opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step()
            tl+=loss.item()*len(xb); n+=len(xb)
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb, yb in es_l:
                ps.append(F.softmax(m(xb),1)[:,1].numpy()); ys.append(yb.numpy())
        auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
        if auc>best: best=auc; best_state={k:v.clone() for k,v in m.state_dict().items()}
        print(f"  ep={ep} loss={tl/n:.4f} es={auc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.load_state_dict(best_state); m.eval(); ps=[]
    with torch.no_grad():
        for xb,_ in te_l: ps.append(F.softmax(m(xb),1)[:,1].numpy())
    return np.concatenate(ps), best

print(f"\n[4] Train 3 seeds FAST pyramid (预构建)...", flush=True)
pte=[]
for sd in [42,56,70]:
    print(f"=== seed={sd} ===", flush=True)
    p, bes = run(sd); pte.append(p); print(f"  es_best={bes:.4f}", flush=True)
p = np.mean(pte, axis=0); auc = roc_auc_score(te_y, p)
print(f"\n{'='*60}")
print(f"  ★ FAST PYRAMID CNN TEST AUC = {auc:.4f}")
print(f"    对比 LGBM ret+negw: 0.5428  ({(auc-0.5428)*100:+.2f}pp)")
print(f"    对比旧慢版 Pyramid: 0.5415  ({(auc-0.5415)*100:+.2f}pp)")
print(f"{'='*60}")

print(f"\nTOTAL: {time.time()-T0:.0f}s")
