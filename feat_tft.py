"""
TinyLSTM+FC, 全量 2M 样本, ETH 1min close+stoch(80,2,1) K
- 无 attention, CPU 每 batch 预计 0.15s
- 全量不降采样
"""
import numpy as np, time, torch, torch.nn as nn, warnings, gc
warnings.filterwarnings('ignore')
import polars as pl
from sklearn.metrics import roc_auc_score

t0 = time.time()

# ====== 1. 数据 ======
raw = pl.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
C = raw['close'].to_numpy().astype(np.float32)
H = raw['high'].to_numpy().astype(np.float32)
L = raw['low'].to_numpy().astype(np.float32)
TS = raw['ts'].to_numpy().astype(np.int64)
N = len(C)

from pandas import Series
lo80  = Series(L).rolling(80).min().to_numpy()
hi80  = Series(H).rolling(80).max().to_numpy()
stk   = Series((C-lo80)/(hi80-lo80+1e-9)*100).rolling(2).mean().to_numpy().astype(np.float32)
HOR   = 15
ret   = np.full(N, np.nan, dtype=np.float32); ret[:N-HOR] = (C[HOR:]/C[:N-HOR]-1).astype(np.float32)
DIR   = (ret > 0).astype(np.int32)

SPLIT = 1704067200
cm, cs = C[TS<SPLIT].mean(), C[TS<SPLIT].std()
C_N = np.clip((C-cm)/(cs+1e-9), -5, 5).astype(np.float32) * 2 - 1   # [-1, 1]
K_N = stk / 100.0                                                     # [0, 1]

# 释放原始 OHLC/ret
del raw, H, L, lo80, hi80, ret, stk, C; gc.collect()
print(f"[1] 特征就绪 ({time.time()-t0:.0f}s)", flush=True)

# ====== 2. 索引 ======
EL = 60
# valid_t 起点: 确保序列覆盖 [t-59, t] 都在 stok 有效区域内
vt = np.arange(139, N-HOR)
tr_idx = vt[TS[vt+HOR] < SPLIT]
te_idx = vt[TS[vt+HOR] >= SPLIT]
del TS, vt; gc.collect()
print(f"[2] 全量 train={len(tr_idx):,}, test={len(te_idx):,}", flush=True)

# ====== 3. 向量化构建 ======
def build(idx, cn, kn, el, chunk=300_000):
    out = np.empty((len(idx), el, 2), dtype=np.float32)
    off = np.arange(el); s = idx - el + 1
    for i in range(0, len(idx), chunk):
        e = min(i+chunk, len(idx))
        w = s[i:e][:,None] + off[None,:]
        out[i:e,:,0] = cn[w]; out[i:e,:,1] = kn[w]
    return out

np.random.seed(42); perm = np.random.permutation(len(tr_idx))
nv = int(len(tr_idx)*0.03); vi, ti2 = tr_idx[perm[:nv]], tr_idx[perm[nv:]]

print("[3] 构建 train/val...", flush=True); e=time.time()
Xtr = build(ti2, C_N, K_N, EL); ytr = DIR[ti2+HOR].astype(np.float32)
Xva = build(vi,  C_N, K_N, EL); yva = DIR[vi+HOR].astype(np.float32)
print(f"    train={Xtr.shape} val={Xva.shape} ({time.time()-e:.1f}s)", flush=True)

# 构建 test (小, 1.4M, float32 ~336MB)
print("[4] 构建 test...", flush=True); e=time.time()
Xte = build(te_idx, C_N, K_N, EL); yte = DIR[te_idx+HOR].astype(np.float32)
print(f"    test={Xte.shape} ({time.time()-e:.1f}s)", flush=True)

# 释放所有原始 numpy
del C_N, K_N, DIR, N, tr_idx, te_idx, ti2, vi, perm, nv, SPLIT, EL, HOR; gc.collect()
print(f"[5] 原始数组释放 ({time.time()-t0:.0f}s)", flush=True)

# ====== 4. DataLoader ======
tdtr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
tdva = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
tdte = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
del Xtr, Xva, ytr, yva; gc.collect()

LD = 2048
tld = torch.utils.data.DataLoader(tdtr, batch_size=LD, shuffle=True,  num_workers=0)
vld = torch.utils.data.DataLoader(tdva, batch_size=LD, shuffle=False, num_workers=0)
eld = torch.utils.data.DataLoader(tdte, batch_size=LD, shuffle=False, num_workers=0)
del tdtr, tdva, tdte; gc.collect()
print(f"[6] DL ready nb={len(tld)} ({time.time()-t0:.0f}s)", flush=True)

# ====== 5. 轻量 LSTM ======
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True, bidirectional=True)
        self.cls  = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,1))
    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        m = out.mean(1)                              # mean pool
        l = torch.cat([hn[0], hn[1]], dim=-1)        # last hidden
        return self.cls(m + l).squeeze(-1)           # residual

m = Tiny(); pcount = sum(p.numel() for p in m.parameters())
print(f"[7] params={pcount:,}", flush=True)

def ev(ld):
    m.eval(); ps, ls = [], []
    with torch.no_grad():
        for x, y in ld:
            ps.append(torch.sigmoid(m(x)).numpy())
            ls.append(y.numpy())
    ps_f=np.concatenate(ps); ls_f=np.concatenate(ls); mask=np.isfinite(ps_f); return ps_f[mask], ls_f[mask]

opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-5)
crit = nn.BCEWithLogitsLoss()
best_auc = 0; noimp = 0

for ep in range(20):
    m.train(); tl=0; tt=0; e0=time.time()
    for x, y in tld:
        opt.zero_grad(); lo=crit(m(x), y)
        if torch.isnan(lo): print(f"  [WARN] nan loss skipped"); continue
        lo.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5); opt.step()
        tl += lo.item()*y.size(0); tt += y.size(0)
    tp, tl2 = ev(vld); va = ((tp>0.5).astype(int)==tl2).mean(); auc = roc_auc_score(tl2, tp)
    dt = time.time()-e0
    print(f"  E{ep+1:2d} loss={tl/max(tt,1):.4f} acc={va:.4f} auc={auc:.4f} | {dt:.0f}s", flush=True)
    if auc > best_auc:
        best_auc = auc; torch.save(m.state_dict(),'/tmp/tft_best.pt'); noimp = 0
    else:
        noimp += 1
        if noimp >= 4: print("  Early stop!"); break

m.load_state_dict(torch.load('/tmp/tft_best.pt'))
tp, tl = ev(eld)
auc = roc_auc_score(tl, tp)
k1 = max(1, int(len(tp)*0.01))
t1 = np.mean(tl[np.argsort(tp)[-k1:]])
base = np.mean(tl)

print(f"\n{'='*65}")
print(f"全量 TinyLSTM 结果 (2M train, 1.4M test, close+stoch_K)")
print(f"{'='*65}")
print(f"  TinyLSTM (2 feats, 全量) AUC={auc:.4f}, top1%={t1:.4f}, lift={t1/base:.2f}x")
print(f"  LGBM (2 feats, str=5)     AUC=0.5174, top1%=0.5341, lift=1.06x")
print(f"  ds_ETH_h15 (55 feats)     AUC=0.5369, top1%=0.5734, lift=1.14x")
print(f"  baseline up rate: {base:.4f}")
print(f"  total time: {time.time()-t0:.0f}s")
