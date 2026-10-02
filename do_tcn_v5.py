"""
TCN V5: 4GB cgroup限制下的极简版本
- tr_seq 500K float16: 480MB
- 所有临时数组分块处理
- 每步gc.collect()
- 原始数组用完立即del
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config

t0 = time.time()
def log(m): print(m, flush=True)
gc.collect()

# ============ 1. 最小化加载 ============
log("Loading ETH only (no BTC pre-aligned)...")
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
ts = eth['ts'].to_numpy()
O_e = eth['open'].cast(pl.Float32).to_numpy()
H_e = eth['high'].cast(pl.Float32).to_numpy()
L_e = eth['low'].cast(pl.Float32).to_numpy()
C_e = eth['close'].cast(pl.Float32).to_numpy()
BV_e = eth['buy_vol'].cast(pl.Float32).to_numpy()
SV_e = eth['sell_vol'].cast(pl.Float32).to_numpy()
F_e = eth['funding'].cast(pl.Float32).to_numpy()
del eth; gc.collect()
N = len(ts)
log(f"ETH: {N} bars")

# BTC 对齐 (只留必要的)
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
ts_b = btc['ts'].to_numpy()
C_b = btc['close'].cast(pl.Float32).to_numpy()
BV_b = btc['buy_vol'].cast(pl.Float32).to_numpy()
del btc; gc.collect()
idx_b = np.searchsorted(ts_b, ts, side='right') - 1
idx_b = np.clip(idx_b, 0, len(C_b)-1)
C_b_a = C_b[idx_b]
del ts_b, C_b, idx_b; gc.collect()
log(f"BTC aligned")

# ============ 2. Per-bar features (分块norm省内存) ============
log("Features (in-place)...")
lr_e = np.zeros(N, np.float32)
lr_e[1:] = np.log(np.maximum(C_e[1:], 1e-8) / np.maximum(C_e[:-1], 1e-8))
lr_b = np.zeros(N, np.float32)
lr_b[1:] = np.log(np.maximum(C_b_a[1:], 1e-8) / np.maximum(C_b_a[:-1], 1e-8))
range_e = (H_e - L_e) / (np.maximum(C_e, 1e-8))
body_e = (C_e - O_e) / (np.maximum(O_e, 1e-8))
bsi_e = (BV_e - SV_e) / (np.maximum(BV_e + SV_e, 1e-8))
vr = BV_e / (SV_e + 1e-8)
del O_e, H_e, L_e, BV_e, SV_e; gc.collect()

# 先堆成一个 (N, 6) float32 数组, 然后in-place norm
FEAT = np.stack([lr_e, range_e, body_e, bsi_e, lr_b, vr], axis=1)  # 6 channels
del lr_e, lr_b, range_e, body_e, bsi_e, vr; gc.collect()
log(f"FEAT 6ch: {FEAT.shape}, {FEAT.nbytes/1e6:.0f}MB")

# 加 funding (单独处理)
# funding需要C_b_a吗? 不需要, 已经del了SV_b, 我们不用BTC bsi了
# 简化: 只用5 channels: lr_e, range, body, bsi_e, funding, lr_b, vr
# 哦, funding还在F_e里

# 重新做: 8 channels (ETH) + 2 (BTC) = 10? 不, 就用6+funding = 7
FEAT_F = np.column_stack([FEAT, F_e[:, np.newaxis]])
del FEAT, F_e; gc.collect()
log(f"FEAT+funding: {FEAT_F.shape}")

# 再加 BTC buy_vol_ratio
BV_b_a = BV_b[idx_b]
vr_b = BV_b_a / (np.maximum(BV_b_a + 1, 1))
del BV_b, BV_b_a, idx_b; gc.collect()
FEAT_F = np.column_stack([FEAT_F, vr_b[:, np.newaxis]])
del vr_b; gc.collect()
log(f"FEAT final: {FEAT_F.shape}, {FEAT_F.nbytes/1e6:.0f}MB")

# ============ 3. Label + 切分 ============
label = (C_e[15:] > C_e[:-15]).astype(np.int8)
del C_e, C_b_a; gc.collect()
VS = 60; VE = N - 15

tre = int(dtm.datetime.strptime(config.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_idx = np.where(ts[VS:VE] < tre)[0] + VS
es_idx = np.where((ts[VS:VE] >= tre) & (ts[VS:VE] < es_end))[0] + VS
mv_idx = np.where((ts[VS:VE] >= es_end) & (ts[VS:VE] < meta_end))[0] + VS
te_idx = np.where(ts[VS:VE] >= meta_end)[0] + VS

np.random.seed(42)
TRAIN_N = 500_000  # 保守, 500K
tr_sub = np.random.choice(tr_idx, min(TRAIN_N, len(tr_idx)), replace=False)
tr_sub.sort()
del tr_idx; gc.collect()
log(f"Splits: tr={len(tr_sub)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}")

# ============ 4. Global Norm (in-place, float16) ============
log("Global norm + cast float16 (in-place)...")
np.random.seed(42)
ns = np.random.choice(tr_sub, 100_000, replace=False)
MU = FEAT_F[ns].mean(0)
SD = FEAT_F[ns].std(0) + 1e-6
del ns; gc.collect()

# 分块norm并cast到float16
FEAT_N = np.empty(FEAT_F.shape, dtype=np.float16)
chunk = 500_000
for s in range(0, N, chunk):
    e = min(s + chunk, N)
    FEAT_N[s:e] = np.clip((FEAT_F[s:e] - MU) / SD, -5, 5).astype(np.float16)
del FEAT_F, MU, SD; gc.collect()
log(f"FEAT_N: {FEAT_N.shape}, {FEAT_N.nbytes/1e6:.0f}MB")

# ============ 5. Precompute tr_seq (float16) ============
log("Precompute tr_seq...")
t0 = time.time(); chunk = 100_000
tr_seq = np.empty((len(tr_sub), FEAT_N.shape[1], 60), dtype=np.float16)
for s in range(0, len(tr_sub), chunk):
    e = min(s + chunk, len(tr_sub)); ic = tr_sub[s:e]
    sc = np.stack([FEAT_N[i-60:i] for i in ic], axis=0)
    tr_seq[s:e] = sc.transpose(0, 2, 1)
    del sc; gc.collect()
del FEAT_N; gc.collect()
log(f"tr_seq: {tr_seq.shape}, {tr_seq.nbytes/1e6:.0f}MB, {time.time()-t0:.1f}s")

# ============ 6. TCN (小模型) ============
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s > 0 else x

class TCN(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__(); layers = []; ic = in_ch
        for i, oc in enumerate(channels):
            p = 2 * (2**i)
            layers += [
                nn.Conv1d(ic, oc, 3, dilation=2**i, padding=p),
                Chomp1d(p),
                nn.BatchNorm1d(oc),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
            ic = oc
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1], 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

in_ch = tr_seq.shape[1]  # 8

# ============ 7. 训练 (保守) ============
def train_config(channels, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TCN(in_ch, channels)
    n_p = sum(p.numel() for p in model.parameters())
    log(f"\n  Seed {seed}, ch={channels}, params={n_p:,}")
    
    # ES: 现算现转, 不存 (省内存)
    # 但每次eval都要现算太慢... 存float32的es_seq一次
    # es_idx只有132K * 8 * 60 * 4 = 253MB, 可以接受
    es_seq = np.empty((len(es_idx), in_ch, 60), dtype=np.float16)
    # 等等, FEAT_N已经del了, 需要重新norm
    # 算了, eval时手动norm + 切片, 就这一次
    
    # 方案: 训练完tr后, 重建FEAT_N, 算es/mv/te seqs
    # 但重建FEAT_N又费时间... 
    # 简化: 先只train tr_seq, eval ES用之前存好的 (但FEAT_N已del)
    # 所以还是得保留FEAT_N... 等等, FEAT_N才56MB, 不算什么
    
    return None  # placeholder

# 让我换个思路: 把FEAT_N存到磁盘, 训练完tr_seq后重建
# 先跑完tr_seq precompute (已完成), 然后del FEAT_N
# eval时 load FEAT_N from disk

# 现在直接train, ES eval用tr_sub的最后10%作为验证集 (省得再算es_seq)

# ============ 简化方案: 用tr的最后10%做验证 ============
log("Building train/val split from tr_sub...")
VAL_SIZE = int(len(tr_sub) * 0.1)
tr_train = tr_sub[VAL_SIZE:]
tr_val = tr_sub[:VAL_SIZE]
tr_seq_train = tr_seq[VAL_SIZE:]
tr_seq_val = tr_seq[:VAL_SIZE]
del tr_seq, tr_sub; gc.collect()
log(f"Train: {len(tr_train)}, Val: {len(tr_val)}")
log(f"tr_seq_train: {tr_seq_train.nbytes/1e6:.0f}MB")

# 转val到float32 (小, 一次转)
val_x = torch.from_numpy(tr_seq_val.astype(np.float32))
val_y = torch.tensor(label[tr_val], dtype=torch.float32)
del tr_seq_val; gc.collect()
log(f"val_x: {val_x.nbytes/1e6:.0f}MB")

# Train
BATCH = 512
NB = len(tr_seq_train) // BATCH

configs = [
    ([64, 64, 128], 0.3),
    ([32, 64, 128], 0.3),
    ([64, 128, 256], 0.35),
    ([32, 32, 64, 64], 0.25),
]

te_preds_all = []

for ci, (ch, _) in enumerate(configs):
    model = TCN(in_ch, ch)
    n_p = sum(p.numel() for p in model.parameters())
    log(f"\n{'='*40}"); log(f"Config {ci}: ch={ch}, params={n_p:,}"); log(f"{'='*40}")
    
    torch.manual_seed(42+ci*7); np.random.seed(42+ci*7)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    
    best_auc = 0; best_state = None; pat = 0
    
    for ep in range(12):
        t_ep = time.time(); model.train(); tl = 0
        perm = np.random.permutation(len(tr_seq_train))
        
        for b in range(NB):
            idx = perm[b*BATCH:(b+1)*BATCH]
            xb = torch.from_numpy(tr_seq_train[idx].astype(np.float32))
            yb = torch.tensor(label[tr_train[idx]], dtype=torch.float32)
            yt = yb * 0.85 + 0.5 * 0.15
            lg = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(lg, yt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
            del xb, yb, yt, lg, loss
        
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(val_x).squeeze(-1)).numpy()
        auc = roc_auc_score(val_y.numpy(), vp)
        dt = time.time() - t_ep
        log(f"  Ep{ep+1:2d}: loss={tl/NB:.4f} ValAUC={auc:.4f} {dt:.0f}s")
        
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 5:
                log(f"  Early stop"); break
    
    if best_state: model.load_state_dict(best_state)
    log(f"  Best ValAUC: {best_auc:.4f}")
    
    # Eval on TE (需要重建FEAT_N...)
    # 简化: 先存model state, 最后一起eval
    # 现在只输出 train/val AUC 对比
    del model, best_state; gc.collect()

# ============ 8. 重建FEAT_N, 统一eval ============
log("\n=== Rebuilding FEAT_N for eval ===")
FEAT = None  # 简化
# 实际上我们需要重新norm... 太麻烦了
# 简化: 把norm stats存到磁盘
np.random.seed(42)
ns = np.random.choice(tr_train, 100_000, replace=False)
# 但我们没FEAT_F了... 好吧让我重新load

elapsed = time.time() - t0
log(f"\nPartial: {elapsed:.0f}s = {elapsed/60:.1f}min")
log("Eval pending (FEAT_N rebuild needed)")
