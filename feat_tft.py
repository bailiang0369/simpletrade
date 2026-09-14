import numpy as np, pandas as pd, time, torch, torch.nn as nn, warnings, gc
warnings.filterwarnings('ignore')
import polars as pl
from sklearn.metrics import roc_auc_score

t0 = time.time()
raw = pl.read_parquet('data/datasets/raw_ETH.parquet')
close = raw['close'].to_numpy().astype(np.float32)
high = raw['high'].to_numpy().astype(np.float32)
low = raw['low'].to_numpy().astype(np.float32)
ts = raw['ts'].to_numpy().astype(np.int64)
n = len(close)

# stoch K
low_80 = pd.Series(low).rolling(80).min().to_numpy()
high_80 = pd.Series(high).rolling(80).max().to_numpy()
stoch_k = pd.Series((close-low_80)/(high_80-low_80+1e-9)*100).rolling(2).mean().to_numpy().astype(np.float32)

h = 15; ret_future = np.full(n, np.nan, dtype=np.float32)
ret_future[:n-h] = (close[h:]/close[:n-h]-1).astype(np.float32)
direction = (ret_future > 0).astype(np.int32)

split_ts = 1704067200; STRIDE=5
valid_t = [t for t in range(59, n-15, STRIDE) if not np.isnan(stoch_k[t]) and not np.isnan(ret_future[t+15])]
valid_t = np.array(valid_t)
tr_idx = valid_t[ts[valid_t+15] < split_ts]
te_idx = valid_t[ts[valid_t+15] >= split_ts]
print(f"train={len(tr_idx):,}, test={len(te_idx):,}")

del raw, high, low, low_80, high_80, ret_future; gc.collect()

# 标准化 (clamp 极值)
cm, cs = close[tr_idx].mean(), close[tr_idx].std()
cn = np.clip((close-cm)/(cs+1e-9), -5, 5).astype(np.float32)
kn = stoch_k / 100.0  # [0,1]
del close, stoch_k; gc.collect()

print("构建序列...", flush=True)
ENCODER = 60
X_tr = np.zeros((len(tr_idx), ENCODER, 2), dtype=np.float32)
X_te = np.zeros((len(te_idx), ENCODER, 2), dtype=np.float32)
for i, t in enumerate(tr_idx): s=t-ENCODER+1; X_tr[i,:,0]=cn[s:t+1]; X_tr[i,:,1]=kn[s:t+1]
for i, t in enumerate(te_idx): s=t-ENCODER+1; X_te[i,:,0]=cn[s:t+1]; X_te[i,:,1]=kn[s:t+1]
y_tr = direction[tr_idx+15].astype(np.float32)
y_te = direction[te_idx+15].astype(np.float32)
del cn, kn, direction, ts; gc.collect()
print(f"X_tr={X_tr.shape}, {X_tr.nbytes/1024**2:.1f}MB, build done {time.time()-t0:.0f}s")

# val split
np.random.seed(42); perm = np.random.permutation(len(X_tr))
n_val = int(len(X_tr)*0.05)
X_va, y_va = X_tr[perm[:n_val]], y_tr[perm[:n_val]]
X_tr2, y_tr2 = X_tr[perm[n_val:]], y_tr[perm[n_val:]]
del X_tr, y_tr; gc.collect()

td_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr2), torch.tensor(y_tr2))
td_va = torch.utils.data.TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
td_te = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
del X_tr2, X_va; gc.collect()

LD = 512
tld = torch.utils.data.DataLoader(td_tr, batch_size=LD, shuffle=True, num_workers=0)
vld = torch.utils.data.DataLoader(td_va, batch_size=LD, shuffle=False, num_workers=0)
eld = torch.utils.data.DataLoader(td_te, batch_size=LD, shuffle=False, num_workers=0)

class TFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=nn.Linear(2,32)
        self.lstm=nn.LSTM(32,32,batch_first=True,bidirectional=True)
        self.attn=nn.MultiheadAttention(64,4,batch_first=True)
        self.norm=nn.LayerNorm(64)
        self.pool=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
        self.cls=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(self,x):
        h=self.inp(x); out,_=self.lstm(h)
        a,_=self.attn(out,out,out); h=self.norm(out+a)
        w=torch.softmax(self.pool(h).squeeze(-1),1); p=torch.bmm(w.unsqueeze(1),h).squeeze(1)
        return self.cls(p).squeeze(-1)

m=TFT(); pcount=sum(p.numel() for p in m.parameters()); print(f"params={pcount:,}")

def ev(ld):
    m.eval(); ps,ls=[],[]
    with torch.no_grad():
        for x,y in ld: ps.append(torch.sigmoid(m(x)).numpy()); ls.append(y.numpy())
    return np.concatenate(ps),np.concatenate(ls)

opt=torch.optim.Adam(m.parameters(),lr=5e-4,weight_decay=1e-5)
crit=nn.BCEWithLogitsLoss()
best=0; noimp=0
for ep in range(20):
    m.train(); tl=0; tt=0; e0=time.time(); n_ok=0
    for x,y in tld:
        opt.zero_grad(); lo=crit(m(x),y)
        if torch.isnan(lo): continue
        lo.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        tl+=lo.item()*y.size(0); tt+=y.size(0); n_ok+=1
    vp,vl=ev(vld); va=((vp>0.5).astype(int)==vl).mean()
    print(f"  E{ep+1:2d} loss={tl/max(tt,1):.4f} val={va:.4f} | {time.time()-e0:.0f}s (n_batches={n_ok})", flush=True)
    if va>best: best=va; torch.save(m.state_dict(),'/tmp/tft_best.pt'); noimp=0
    else:
        noimp+=1
        if noimp>=4: print("  Early stop"); break

m.load_state_dict(torch.load('/tmp/tft_best.pt'))
tp,tl=ev(eld)
auc_tft = roc_auc_score(tl, tp)
k1=max(1,int(len(tp)*0.01)); t1_tft=np.mean(tl[np.argsort(tp)[-k1:]])
base_up = np.mean(tl)

print(f"\n{'='*60}")
print("最终对比 (Time split: 2020-2023 train, 2024 test)")
print(f"{'='*60}")
print(f"  {'LightGBM (close+stoch_K)':<28s} AUC=0.5174, top1%=0.5341, lift=1.06x")
print(f"  {'Simple TFT (close+stoch_K)':<28s} AUC={auc_tft:.4f}, top1%={t1_tft:.4f}, lift={t1_tft/base_up:.2f}x")
print(f"  {'ds_ETH_h15 (55 feats)':<28s} AUC=0.5369, top1%=0.5734, lift=1.14x")
print(f"\n基线上涨率: {base_up:.4f}")
print(f"总耗时: {time.time()-t0:.0f}s")
