"""ETH h15 1D CNN: 直接看 OHLCV 序列形态
和 LGB/CB 完全不同的视角, ensemble 才会有效
内存优化: DataLoader streaming, 不预存全部窗口
"""
import numpy as np, pandas as pd, time, gc, sys, os, warnings
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); PAR='data/datasets'; NPY='data/splits_npy'
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
log = lambda *a: print(' '.join(str(x) for x in a), flush=True)
log(f"Device: {DEVICE}")

# ========== Step 1: Build OHLCV windows ==========
log("\n[1] Load raw ETH & build sequences...")
import polars as pl
raw = pl.read_parquet(f'{PAR}/raw_ETH.parquet').sort('ts')
ts = raw['ts'].to_numpy().astype(np.int64)
ohlcv = np.stack([raw[c].to_numpy().astype(np.float32) for c in ['open','high','low','close','buy_vol']], axis=1)
close = raw['close'].to_numpy().astype(np.float32)
log(f"  raw: {ohlcv.shape}")

# 参数
WINDOW = 60  # 过去 60 min (1h)
H = 15       # 预测 15 min
BATCH = 2048

# 数据集类 (streaming, 不预存)
class OHLCVDataset(Dataset):
    def __init__(self, ts_arr, ohlcv_arr, close_arr, h, tlo, thi, window):
        self.ts = ts_arr
        self.ohlcv = ohlcv_arr
        self.close = close_arr
        self.h = h
        self.w = window
        # 有效索引: window 之后 到 len-h 之前
        self.valid = np.where((ts_arr >= tlo) & (ts_arr < thi))[0]
        self.valid = self.valid[(self.valid >= window) & (self.valid <= len(close_arr)-h-1)]
        log(f"    ts range [{tlo},{thi}): {len(self.valid):,} valid indices")
    def __len__(self): return len(self.valid)
    def __getitem__(self, idx):
        i = self.valid[idx]
        x = self.ohlcv[i-self.w:i].copy()  # (60, 5)
        # Normalize per window (important for CNN!)
        x = x / (np.abs(x[-1,3]) + 1e-8) - 1  # normalize by last close
        y = 1 if self.close[i+self.h] / self.close[i] - 1 > 0 else 0
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)

ds_tr = OHLCVDataset(ts, ohlcv, close, H, 0, TRAIN_END, WINDOW)
ds_es = OHLCVDataset(ts, ohlcv, close, H, TRAIN_END, ES_END, WINDOW)
ds_te = OHLCVDataset(ts, ohlcv, close, H, META_END, 10**18, WINDOW)
log(f"  tr={len(ds_tr):,} es={len(ds_es):,} te={len(ds_te):,}")

del raw, ohlcv, close; gc.collect()

# ========== Step 2: CNN Model ==========
class SimpleCNN1D(nn.Module):
    def __init__(self, in_ch=5, w=60):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        fc_in = 32 * (w // 4)
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = x.transpose(1,2)  # (B, 5, 60)
        x = self.act(self.conv1(x))  # (B, 16, 60)
        x = self.pool(x)             # (B, 16, 30)
        x = self.act(self.conv2(x))  # (B, 32, 30)
        x = self.pool(x)             # (B, 32, 15)
        x = x.flatten(1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x

# ========== Step 3: Train ==========
log(f"\n[2] Train CNN ({DEVICE})...")
def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mdl = SimpleCNN1D().to(DEVICE)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    dl_es = DataLoader(ds_es, batch_size=BATCH, shuffle=False, num_workers=0)
    
    best_auc, best_state, patience = 0, None, 0
    for epoch in range(30):
        mdl.train(); t0=time.time(); tl=0; nb=0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); out = mdl(xb); l = loss_fn(out, yb); l.backward(); opt.step()
            tl += l.item(); nb += 1
        
        # Eval
        mdl.eval(); preds, labels = [], []
        with torch.no_grad():
            for xb, yb in dl_es:
                preds.append(torch.sigmoid(mdl(xb.to(DEVICE))).cpu().numpy())
                labels.append(yb.numpy())
        es_auc = roc_auc_score(np.concatenate(labels), np.concatenate(preds))
        
        if es_auc > best_auc:
            best_auc = es_auc
            best_state = {k: v.clone() for k,v in mdl.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 5: break
        
        if epoch % 3 == 0:
            log(f"  seed={seed} ep={epoch}: loss={tl/nb:.4f} es_auc={es_auc:.4f} best={best_auc:.4f} ({time.time()-t0:.0f}s)")
    
    mdl.load_state_dict(best_state)
    return mdl

# Train one model first to check AUC
model = train_one(42)
dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)
model.eval()
preds, labels = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        preds.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        labels.append(yb.numpy())
y_te = np.concatenate(labels); p_cnn = np.concatenate(preds)
cnn_te_auc = roc_auc_score(y_te, p_cnn)
log(f"\n[3] CNN seed=42: test AUC={cnn_te_auc:.4f}")

# Train 5 seeds if first one is promising
if cnn_te_auc > 0.530:
    log("  CNN promising! Training 5 seeds...")
    all_preds = [p_cnn]
    for sd in [49, 56, 63, 70]:
        m = train_one(sd)
        m.eval(); ps=[]
        with torch.no_grad():
            for xb, yb in dl_te:
                ps.append(torch.sigmoid(m(xb.to(DEVICE))).cpu().numpy())
        all_preds.append(np.concatenate(ps))
        del m; gc.collect(); torch.cuda.empty_cache() if DEVICE=='cuda' else None
    p_cnn_mean = np.mean(all_preds, axis=0)
    cnn_te_auc = roc_auc_score(y_te, p_cnn_mean)
    log(f"  CNN 5-seed: test AUC={cnn_te_auc:.4f}")

# ========== Step 4: Compare with LGB ==========
log(f"\n[4] Compare with LGB...")
# Load LGB predictions (need to build fresh or use base ds)
# 简化: 用 ds_ETH_h15.test 的 y 对齐 (CNN test ds 也是同一时间窗口)
import pyarrow.parquet as pq
t = pq.read_table(f'{PAR}/ds_ETH_h15.parquet', columns=['ts','label'])
ts_ds = t.column('ts').to_numpy().astype(np.int64)
y_ds = t.column('label').to_numpy().astype(np.int32)
del t; gc.collect()
m = (ts_ds >= META_END)
y_lgb = y_ds[m][:len(p_cnn)]  # trim to match CNN output length

# LGB was 0.5440 — 我们需要 LGB 预测值才能算 ensemble
# 简化: 重新快速训练 LGB 拿到预测值
log("  Retrain LGB to get predictions...")
NPY='data/splits_npy'
def load_c(prefixes, split):
    Xs=[]; y=None
    for xp, yp in prefixes:
        X=np.load(f'{NPY}/ETH_h15{xp}_{split}_X.npy').astype(np.float32)
        if yp is not None: y=np.load(f'{NPY}/ETH_h15{yp}_{split}_y.npy').astype(np.int32)
        Xs.append(X)
    mr=min(X.shape[0] for X in Xs)
    return np.hstack([X[:mr] for X in Xs]), y[:mr] if y is not None else None

import lightgbm as lgb
Xtr,ytr=load_c([('',''),('_fund',None)],'train')
Xes,yes=load_c([('',''),('_fund',None)],'early_stop')
Xte_lgb,yte_lgb=load_c([('',''),('_fund',None)],'test')
ret_full=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw=np.where(np.abs(ret_full[:Xtr.shape[0]])>=np.quantile(np.abs(ret_full[:Xtr.shape[0]]),0.90),0.3,1.0).astype(np.float32)
del ret_full; gc.collect()
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
        'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_lgb_all=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
    m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_lgb_all.append(m.predict(Xte_lgb))
del Xtr,ytr,Xes,yes,Xte_lgb,yte_lgb,sw; gc.collect()
p_lgb_mean = np.mean(p_lgb_all, axis=0)

# Align CNN and LGB preds lengths (可能差几十个)
min_len = min(len(p_cnn), len(p_lgb_mean))
p_cnn_a = p_cnn[:min_len]
p_lgb_a = p_lgb_mean[:min_len]
y_test = y_lgb[:min_len]

lgb_auc = roc_auc_score(y_test, p_lgb_a)
cnn_auc = roc_auc_score(y_test, p_cnn_a)
ens_auc = roc_auc_score(y_test, (p_cnn_a + p_lgb_a) / 2)
corr = np.corrcoef(p_cnn_a, p_lgb_a)[0,1]

log(f"\n{'='*55}")
log(f" ETH h15: CNN vs LGB + Ensemble")
log(f"{'='*55}")
log(f" LGB base+fund+negw: te={lgb_auc:.4f}")
log(f" CNN 1D OHLCV:        te={cnn_auc:.4f}")
log(f" Ensemble (avg):      te={ens_auc:.4f}  (gain={ens_auc-max(lgb_auc,cnn_auc):+.4f})")
log(f" corr(CNN,LGB):       {corr:.3f}")
log(f"\nTOTAL: {time.time()-T0:.0f}s")
