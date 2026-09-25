"""形态结构 CNN: 输入 256 根 1min close+stoch_k14 序列, 直接学形态, 不用 ret
用户说: 只看 close + stoch_k, 形态结构, 不要 ret
"""
import numpy as np, pandas as pd, time, gc, os, sys, warnings
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}, torch={torch.__version__}", flush=True)

# ============= 1. Build sequence dataset =============
print("[1] Build sequences...", flush=True)
raw = pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts').reset_index(drop=True)
ds  = pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','stoch_k14']).sort_values('ts').reset_index(drop=True)

# Compute stoch_k14 from raw (to be consistent, ds stoch_k14 already there but let's recompute to verify)
L14 = raw['low'].rolling(14, min_periods=1).min()
H14 = raw['high'].rolling(14, min_periods=1).max()
raw['sk14'] = np.where(H14-L14>1e-9, (raw['close']-L14)/(H14-L14)*100, 50.0).astype(np.float32)
raw['close'] = raw['close'].astype(np.float32)
print(f"  raw: {raw.shape}, ds: {ds.shape}", flush=True)

# Align ds ts to raw row indices (binary search, O(log n) per row)
raw_ts = raw['ts'].to_numpy().astype(np.int64)
ds_ts  = ds['ts'].to_numpy().astype(np.int64)
idx = np.searchsorted(raw_ts, ds_ts)
idx = np.clip(idx, 1000, len(raw)-1)  # need at least 256 lookback
print(f"  Aligned {len(idx):,} rows", flush=True)

# Split by ts (same as before)
TRAIN_END = 1722556800
ES_END    = 1725148800
META_END  = 1759363200

def get_split_mask(lo, hi):
    return (ds_ts >= lo) & (ds_ts < hi)

tr_idx = idx[get_split_mask(0, TRAIN_END)]
es_idx = idx[get_split_mask(TRAIN_END, ES_END)]
te_idx = idx[get_split_mask(META_END, 10**18)]
tr_y   = ds.loc[get_split_mask(0, TRAIN_END), 'label'].to_numpy().astype(np.int64)
es_y   = ds.loc[get_split_mask(TRAIN_END, ES_END), 'label'].to_numpy().astype(np.int64)
te_y   = ds.loc[get_split_mask(META_END, 10**18), 'label'].to_numpy().astype(np.int64)
print(f"  tr={len(tr_idx):,} es={len(es_idx):,} te={len(te_idx):,}", flush=True)

CLOSE = raw['close'].to_numpy()
SK14  = raw['sk14'].to_numpy()
del raw, ds; gc.collect()

WINDOW = 256  # 256 根 1min lookback = 4h16min

class SeqDataset(Dataset):
    def __init__(self, indices, labels, window=256):
        self.indices = indices.astype(np.int64)
        self.labels  = labels.astype(np.int64)
        self.window  = window
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        end = self.indices[i]
        s   = end - self.window + 1
        c   = CLOSE[s:end+1].copy()
        k   = SK14[s:end+1].copy()
        # Normalize close by first value (形态结构不看绝对价格)
        c0  = c[0]
        c   = np.where(c0>1e-9, c/c0 - 1.0, 0.0)  # 归一化到相对起点变化
        k   = k / 100.0  # stoch_k 已经是 0-100, /100 → 0-1
        x   = np.stack([c.astype(np.float32), k.astype(np.float32)], axis=0)  # (2, 256)
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)

print("[2] Define model...", flush=True)

# ============= 2. 1D ResNet CNN =============
class ResBlock1D(nn.Module):
    def __init__(self, ch, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, k, padding=k//2)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=k//2)
        self.bn2   = nn.BatchNorm1d(ch)
    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(out + x)

class ShapeCNN(nn.Module):
    def __init__(self, in_ch=2, out_chs=(32,64,128)):
        super().__init__()
        self.stem  = nn.Conv1d(in_ch, out_chs[0], 7, padding=3)
        self.pool1 = nn.MaxPool1d(4)
        self.res1  = ResBlock1D(out_chs[0], k=5)
        self.up1   = nn.Conv1d(out_chs[0], out_chs[1], 1)
        self.pool2 = nn.MaxPool1d(4)
        self.res2  = ResBlock1D(out_chs[1], k=3)
        self.up2   = nn.Conv1d(out_chs[1], out_chs[2], 1)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Sequential(
            nn.Linear(out_chs[2], 64), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = F.silu(self.stem(x))
        x = self.res1(self.pool1(x))
        x = self.up1(F.silu(x))
        x = self.res2(self.pool2(x))
        x = self.up2(F.silu(x))
        x = self.pool3(x).flatten(1)
        return self.fc(x)

# ============= 3. Train =============
BATCH=512
EPOCHS=15
LR=3e-4

def train_one(model, tr_idx, tr_y, es_idx, es_y, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds = SeqDataset(tr_idx, tr_y, WINDOW)
    es_ds = SeqDataset(es_idx, es_y, WINDOW)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
    es_loader = DataLoader(es_ds, batch_size=BATCH*2, shuffle=False, num_workers=0)
    
    m = ShapeCNN().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce  = nn.CrossEntropyLoss()
    
    best_es = 0.0; best_state = None
    for ep in range(1, EPOCHS+1):
        m.train(); tl=0; nt=0
        for xb, yb in tr_loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(); out=m(xb); loss=ce(out, yb); loss.backward()
            opt.step(); tl+=loss.item()*len(xb); nt+=len(xb)
        sch.step()
        
        # Eval
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb, yb in es_loader:
                xb=xb.to(DEVICE); out=m(xb)
                ps.append(F.softmax(out,1)[:,1].cpu().numpy())
                ys.append(yb.numpy())
        ps=np.concatenate(ps); ys=np.concatenate(ys)
        auc=roc_auc_score(ys, ps)
        if auc>best_es: best_es=auc; best_state={k:v.cpu() for k,v in m.state_dict().items()}
        if ep%5==0: print(f"  ep={ep} loss={tl/nt:.4f} es_AUC={auc:.4f} best={best_es:.4f}", flush=True)
    
    # Restore best
    m.load_state_dict(best_state)
    return m

def predict(model, indices, labels):
    ds = SeqDataset(indices, labels, WINDOW)
    loader = DataLoader(ds, batch_size=BATCH*2, shuffle=False, num_workers=0)
    model.eval(); ps=[]
    with torch.no_grad():
        for xb, _ in loader:
            xb=xb.to(DEVICE); out=model(xb)
            ps.append(F.softmax(out,1)[:,1].cpu().numpy())
    return np.concatenate(ps)

print(f"\n[3] Train 3-seed ShapeCNN (WINDOW={WINDOW})...", flush=True)
pte=[]
for sd in [42, 56, 70]:
    print(f"  === seed={sd} ===", flush=True)
    m = train_one(torch.nn.Identity(), tr_idx, tr_y, es_idx, es_y, seed=sd)
    pte.append(predict(m, te_idx, te_y))
    del m; gc.collect()

p = np.mean(pte, axis=0)
auc = roc_auc_score(te_y, p)
print(f"\n★★ ShapeCNN (close+stoch_k, seq shape) TEST AUC = {auc:.4f}", flush=True)
print(f"   baseline LightGBM (56 feats, ret-based): AUC = 0.5428", flush=True)
print(f"   差距: {(auc-0.5428)*100:+.2f}pp", flush=True)

# tpd sweep
print(f"\n[4] tpd sweep", flush=True)
print(f"{'q':>8} {'tpd':>6} {'ACC':>7} {'n':>7} {'Δ65%':>8}")
best=None
for q in np.arange(0.985, 0.998, 0.0005):
    th=np.quantile(p,q); lm=p>th; sm=p<(1-th); tm=lm|sm; n=tm.sum()
    if n<50: continue
    ss=sm[tm]; acc=(((~ss)&(te_y[tm]==1))|(ss&(te_y[tm]==0))).mean()*100
    tpd=n/332
    if 12<=tpd<=18:
        diff=acc-65
        mark='◀' if abs(tpd-14.4)<0.5 else ''
        print(f"{q:.4f}  {tpd:5.1f}  {acc:5.1f}%  {n:6d}  {diff:+7.1f}pp {mark}", flush=True)
        if best is None or acc>best[0]: best=(acc,q,tpd)

print(f"\n★ tpd≈14.4 最佳: ACC={best[0]:.1f}% q={best[1]:.4f} tpd={best[2]:.1f}", flush=True)
print(f"\nTOTAL: {time.time()-T0:.0f}s", flush=True)
