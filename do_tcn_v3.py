"""
TCN V3: 预计算per-bar特征 + 快速序列加载

关键优化:
- 先算完所有 (N, 8) per-bar 特征 → 112MB, 全放内存
- Dataset只做切片 + per-seq zscore + transpose → 极快
- 8 channels: ETH lr, range, body, bsi, funding, BTC lr, BTC bsi, vol_ratio

比V2快 ~5-10x, 让我们能跑更多配置/更大数据
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config

t0 = time.time()

# ============================================================
# 1. 加载原始数据
# ============================================================
print("Loading raw data...", flush=True)
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

ts = eth['ts'].to_numpy()
O_e = eth['open'].cast(pl.Float32).to_numpy()
H_e = eth['high'].cast(pl.Float32).to_numpy()
L_e = eth['low'].cast(pl.Float32).to_numpy()
C_e = eth['close'].cast(pl.Float32).to_numpy()
BV_e = eth['buy_vol'].cast(pl.Float32).to_numpy()
SV_e = eth['sell_vol'].cast(pl.Float32).to_numpy()
F_e = eth['funding'].cast(pl.Float32).to_numpy()
del eth; gc.collect()

ts_b = btc['ts'].to_numpy()
C_b = btc['close'].cast(pl.Float32).to_numpy()
BV_b = btc['buy_vol'].cast(pl.Float32).to_numpy()
SV_b = btc['sell_vol'].cast(pl.Float32).to_numpy()
del btc; gc.collect()

idx_b = np.searchsorted(ts_b, ts, side='right') - 1
idx_b = np.clip(idx_b, 0, len(C_b) - 1)
C_b_a = C_b[idx_b]; BV_b_a = BV_b[idx_b]; SV_b_a = SV_b[idx_b]
del ts_b, C_b, BV_b, SV_b, idx_b; gc.collect()

N = len(ts)
print(f"Loaded {N} bars, mem so far: {sum(a.nbytes for a in [O_e,H_e,L_e,C_e,BV_e,SV_e,F_e,C_b_a,BV_b_a,SV_b_a,ts])/1e9:.2f}GB", flush=True)

# ============================================================
# 2. 预计算所有 per-bar 特征 (N, 8) float32
# ============================================================
print("Precomputing per-bar features...", flush=True)

# log returns
lr_e = np.zeros(N, np.float32)
lr_e[1:] = np.log(np.maximum(C_e[1:], 1e-8) / np.maximum(C_e[:-1], 1e-8))
lr_b = np.zeros(N, np.float32)
lr_b[1:] = np.log(np.maximum(C_b_a[1:], 1e-8) / np.maximum(C_b_a[:-1], 1e-8))

# ETH features
range_e = (H_e - L_e) / (np.maximum(C_e, 1e-8))
body_e = (C_e - O_e) / (np.maximum(O_e, 1e-8))
bsi_e = (BV_e - SV_e) / (np.maximum(BV_e + SV_e, 1e-8))
bsi_b = (BV_b_a - SV_b_a) / (np.maximum(BV_b_a + SV_b_a, 1e-8))
vol_ratio = BV_e / (SV_e + 1e-8)

# 组装 (N, 8)
# [lr_e, range_e, body_e, bsi_e, funding, lr_b, bsi_b, vol_ratio]
FEAT = np.stack([lr_e, range_e, body_e, bsi_e, F_e, lr_b, bsi_b, vol_ratio], axis=1).astype(np.float32)
del lr_e, lr_b, range_e, body_e, bsi_e, bsi_b, vol_ratio
del O_e, H_e, L_e, BV_e, SV_e, BV_b_a, SV_b_a
gc.collect()

print(f"FEAT shape={FEAT.shape}, mem={FEAT.nbytes/1e6:.0f}MB", flush=True)

# ============================================================
# 3. Label + 切分
# ============================================================
HORIZON = 15
LOOKBACK = 60

label = (C_e[HORIZON:] > C_e[:-HORIZON]).astype(np.int8)
del C_e; gc.collect()

VALID_START = LOOKBACK
VALID_END = N - HORIZON

tre = int(dtm.datetime.strptime(config.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts[VALID_START:VALID_END] < tre)[0] + VALID_START
es_idx = np.where((ts[VALID_START:VALID_END] >= tre) & (ts[VALID_START:VALID_END] < es_end))[0] + VALID_START
mv_idx = np.where((ts[VALID_START:VALID_END] >= es_end) & (ts[VALID_START:VALID_END] < meta_end))[0] + VALID_START
te_idx = np.where(ts[VALID_START:VALID_END] >= meta_end)[0] + VALID_START

np.random.seed(42)
tr_sub = np.random.choice(tr_idx, min(2_000_000, len(tr_idx)), replace=False)
tr_sub.sort()
print(f"Splits: tr_full={len(tr_idx)} tr_sub={len(tr_sub)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}", flush=True)
del tr_idx; gc.collect()

# ============================================================
# 4. 快速 Dataset
# ============================================================
class FastSeqDS(Dataset):
    __slots__ = ('idx', 'feat', 'label', 'L')
    def __init__(self, indices, feat_arr, label_arr, lookback=60):
        self.idx = indices
        self.feat = feat_arr
        self.label = label_arr
        self.L = lookback
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]
        sl = self.feat[i-self.L:i]  # (L, 8)
        m = sl.mean(0, keepdims=True)
        s = sl.std(0, keepdims=True) + 1e-6
        sl = np.clip((sl - m) / s, -5, 5)
        # (L, CH) -> (CH, L) for Conv1d
        return torch.from_numpy(sl.T), torch.tensor(float(self.label[i]), dtype=torch.float32)


# ============================================================
# 5. TCN 模型
# ============================================================
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s > 0 else x

class TBlock(nn.Module):
    def __init__(self, ic, oc, k, d, drop=0.3):
        super().__init__()
        p = (k-1)*d
        self.c1 = nn.Conv1d(ic, oc, k, dilation=d, padding=p)
        self.ch1 = Chomp1d(p)
        self.bn1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, k, dilation=d, padding=p)
        self.ch2 = Chomp1d(p)
        self.bn2 = nn.BatchNorm1d(oc)
        self.down = nn.Conv1d(ic, oc, 1) if ic != oc else None
        self.r = nn.ReLU()
        self.dp = nn.Dropout(drop)
    def forward(self, x):
        o = self.r(self.bn1(self.ch1(self.c1(x)))); o = self.dp(o)
        o = self.r(self.bn2(self.ch2(self.c2(o)))); o = self.dp(o)
        res = x if self.down is None else self.down(x)
        return self.r(o + res)

class TCN(nn.Module):
    def __init__(self, in_ch=8, channels=[64,64,128,128], k=3, drop=0.3):
        super().__init__()
        layers = []; ic = in_ch
        for i, oc in enumerate(channels):
            layers.append(TBlock(ic, oc, k, 2**i, drop)); ic = oc
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels[-1], 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.head(self.net(x))


# ============================================================
# 6. 训练
# ============================================================
def run_training(channels, dropout, seed=42, lr=3e-4, wd=1e-3, ls=0.15):
    torch.manual_seed(seed); np.random.seed(seed)
    
    model = TCN(8, channels, k=3, drop=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Seed {seed}, params={n_params:,}, ch={channels}, drop={dropout}", flush=True)
    
    tr_ds = FastSeqDS(tr_sub, FEAT, label)
    es_ds = FastSeqDS(es_idx, FEAT, label)
    
    # 预取ES数据 (小, 直接放内存加速eval)
    es_loader_eval = DataLoader(es_ds, batch_size=1024, shuffle=False)
    
    tr_loader = DataLoader(tr_ds, batch_size=512, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    
    best_auc = 0; best_state = None; patience = 0
    
    for ep in range(15):
        t_ep = time.time(); model.train(); total_loss = 0; nb = 0
        
        for xb, yb in tr_loader:
            yt = yb * (1-ls) + 0.5*ls
            lg = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(lg, yt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); nb += 1
        
        sched.step()
        
        # ES eval
        model.eval(); preds = []
        with torch.no_grad():
            for xb, yb in es_loader_eval:
                preds.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
        preds = np.concatenate(preds)
        auc = roc_auc_score(label[es_idx], preds)
        dt_ep = time.time() - t_ep
        
        print(f"    Ep{ep+1:2d}: loss={total_loss/nb:.4f} ES_AUC={auc:.4f} {dt_ep:.0f}s lr={sched.get_last_lr()[0]:.5f}", flush=True)
        
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                print(f"    Early stop @ ep{ep+1}"); break
    
    if best_state: model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# 7. 快速批量评估
# ============================================================
def get_preds(model, indices):
    ds = FastSeqDS(indices, FEAT, label)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
    return np.concatenate(preds)

def rolling_eval(preds, indices, qs=[98, 99, 99.5]):
    """preds: 模型score, indices: 全局位置, qs: 分位数"""
    ts_eval = ts[indices]
    y_true = label[indices]
    auc = roc_auc_score(y_true, preds)
    n_days = (ts_eval[-1] - ts_eval[0]) / 86400
    
    results = {}
    for q in qs:
        thrs = np.zeros(len(indices))
        # 向量化: 用rolling window
        # 简化: 用过去30天的分位数
        for i in range(len(indices)):
            t_now = ts_eval[i]
            t_30d = t_now - 30*86400
            hist_mask = (ts_eval[:i] >= t_30d) & (ts_eval[:i] < t_now)
            if hist_mask.sum() >= 100:
                thrs[i] = np.percentile(preds[hist_mask], q)
            else:
                thrs[i] = np.percentile(preds[:max(i,1)], q)
        
        sig = preds > thrs
        acc = y_true[sig].mean() if sig.sum() > 0 else 0
        tpd = sig.sum() / n_days
        results[q] = (acc, tpd, int(sig.sum()))
    return auc, results


# ============================================================
# 8. 主实验
# ============================================================
configs = [
    ([64, 64, 128, 128], 0.3),
    ([32, 64, 128, 128], 0.3),
    ([64, 128, 256, 256], 0.35),
    ([64, 64, 64, 64], 0.25),
]

te_preds_list = []
te_y = label[te_idx]
ts_te = ts[te_idx]
ens_auc = None

for ci, (ch, dp) in enumerate(configs):
    print(f"\n{'='*55}")
    print(f"Config {ci}: channels={ch}, dropout={dp}")
    print(f"{'='*55}", flush=True)
    
    model, best_auc = run_training(ch, dp, seed=42+ci*7)
    print(f"  Best ES AUC: {best_auc:.4f}", flush=True)
    
    # Meta-val + Test
    print("  MetaVal:", flush=True)
    mv_preds = get_preds(model, mv_idx)
    mv_auc, mv_res = rolling_eval(mv_preds, mv_idx)
    print(f"    MV AUC={mv_auc:.4f}")
    for q, (a, t, n) in mv_res.items():
        print(f"      q{q}: acc={a:.4f} tpd={t:.1f} n={n}")
    
    print("  Test:", flush=True)
    te_preds = get_preds(model, te_idx)
    te_preds_list.append(te_preds)
    
    te_auc, te_res = rolling_eval(te_preds, te_idx)
    print(f"    TE AUC={te_auc:.4f}")
    for q, (a, t, n) in te_res.items():
        print(f"      q{q}: acc={a:.4f} tpd={t:.1f} n={n}")
    
    del model; gc.collect()

# Ensemble
print(f"\n{'='*55}")
print("ENSEMBLE (avg of 4 configs)")
print(f"{'='*55}", flush=True)

ens_preds = np.mean(te_preds_list, axis=0)
ens_auc = roc_auc_score(te_y, ens_preds)
print(f"Ensemble TE AUC: {ens_auc:.4f}")

ens_mv = np.mean([get_preds(__import__('copy').deepcopy(te_preds_list[0]) * 0 + 0, mv_idx) for _ in range(1)], axis=0)  # skip

# 直接用ensemble preds做rolling eval
n_days = (ts_te[-1] - ts_te[0]) / 86400
for q in [98, 99, 99.5, 99.7]:
    thrs = np.zeros(len(te_idx))
    for i in range(len(te_idx)):
        t_now = ts_te[i]; t_30d = t_now - 30*86400
        hm = (ts_te[:i] >= t_30d) & (ts_te[:i] < t_now)
        if hm.sum() >= 100:
            thrs[i] = np.percentile(ens_preds[hm], q)
        else:
            thrs[i] = np.percentile(ens_preds[:max(i,1)], q)
    sig = ens_preds > thrs
    acc = te_y[sig].mean() if sig.sum() > 0 else 0
    tpd = sig.sum() / n_days
    print(f"  q={q}: acc={acc:.4f} tpd={tpd:.1f} n={sig.sum()}")

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)
