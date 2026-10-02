"""
TCN (Temporal Convolutional Network) for ETH 15m direction prediction.

核心思路:
- 序列输入: ETH raw OHLCV + funding + BTC aligned close/vol, lookback=60 bars (15h)
- TCN架构: dilated causal conv, 多尺度感受野, 加skip connection
- 在线Dataset: 不预计算全部序列, __getitem__ 时切片, 省内存
- 训练: CPU, float32, batch=256 (序列模型batch要小)
- 评估: rolling 30-day quantile threshold (q=98/99/99.5)

对比基线 MLP AUC=0.5436, 目标突破 0.55
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

# ETH 序列
ts = eth['ts'].to_numpy()
O_e = eth['open'].cast(pl.Float32).to_numpy()
H_e = eth['high'].cast(pl.Float32).to_numpy()
L_e = eth['low'].cast(pl.Float32).to_numpy()
C_e = eth['close'].cast(pl.Float32).to_numpy()
BV_e = eth['buy_vol'].cast(pl.Float32).to_numpy()
SV_e = eth['sell_vol'].cast(pl.Float32).to_numpy()
F_e = eth['funding'].cast(pl.Float32).to_numpy()
del eth; gc.collect()

# BTC 对齐到 ETH ts
ts_b = btc['ts'].to_numpy()
C_b = btc['close'].cast(pl.Float32).to_numpy()
BV_b = btc['buy_vol'].cast(pl.Float32).to_numpy()
SV_b = btc['sell_vol'].cast(pl.Float32).to_numpy()
del btc; gc.collect()

idx_b = np.searchsorted(ts_b, ts, side='right') - 1
idx_b = np.clip(idx_b, 0, len(C_b) - 1)
C_b_a = C_b[idx_b]
BV_b_a = BV_b[idx_b]
SV_b_a = SV_b[idx_b]
del ts_b, C_b, BV_b, SV_b; gc.collect()

print(f"Loaded {len(ts)} ETH bars, time range: {dtm.datetime.fromtimestamp(ts[0],tz=dtm.timezone.utc)} -> {dtm.datetime.fromtimestamp(ts[-1],tz=dtm.timezone.utc)}", flush=True)

# ============================================================
# 2. 构建序列特征 (在线计算, 不存全量)
# ============================================================
LOOKBACK = 60   # 回看60根K线 = 15h
HORIZON = 15    # 预测15根后

# label
label = (C_e[HORIZON:] > C_e[:-HORIZON]).astype(np.int8)

# 对齐: 丢掉前LOOKBACK根 (没有足够历史) 和最后HORIZON根 (没有label)
# 有效索引范围: [LOOKBACK, len(ts)-HORIZON)
# 每个样本 i 对应 ts[i], 序列窗口 [i-LOOKBACK, i), label[i]
N_FULL = len(ts)
VALID_START = LOOKBACK
VALID_END = N_FULL - HORIZON
N_VALID = VALID_END - VALID_START
print(f"Valid samples: {N_VALID}", flush=True)

# 时间切分
tre = int(dtm.datetime.strptime(config.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
meta_end = int(dtm.datetime.strptime(config.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

# 在有效范围内找切分点
tr_mask = (ts[VALID_START:VALID_END] < tre)
es_mask = (ts[VALID_START:VALID_END] >= tre) & (ts[VALID_START:VALID_END] < es_end)
mv_mask = (ts[VALID_START:VALID_END] >= es_end) & (ts[VALID_START:VALID_END] < meta_end)
te_mask = (ts[VALID_START:VALID_END] >= meta_end)

tr_idx = np.where(tr_mask)[0] + VALID_START
es_idx = np.where(es_mask)[0] + VALID_START
mv_idx = np.where(mv_mask)[0] + VALID_START
te_idx = np.where(te_mask)[0] + VALID_START
ts_te = ts[te_idx].copy()

del tr_mask, es_mask, mv_mask, te_mask; gc.collect()
print(f"Splits: tr={len(tr_idx)} es={len(es_idx)} mv={len(mv_idx)} te={len(te_idx)}", flush=True)

# 训练子集 (CPU太慢, 用2M训练足够)
np.random.seed(42)
if len(tr_idx) > 2_000_000:
    tr_idx = np.random.choice(tr_idx, 2_000_000, replace=False)
    tr_idx.sort()
print(f"Train subset: {len(tr_idx)}", flush=True)

# ============================================================
# 3. 归一化统计量 (只用训练集)
# ============================================================
# 对原始OHLCV做滚动归一化: z-score用过去240bar的mean/std (在线计算)
# 但为了简单, 我们直接对 log-return 做 z-score, 然后序列输入用 log-return
# 8 channels per asset: lr_1, hr_1, lr_vol, body, pos, ... 简化为:
#   ETH: log_ret_1, range_norm, body_ratio, vol_norm, buy_sell_imbalance, funding
#   BTC: log_ret_1, vol_norm, buy_sell_imbalance

def compute_features_for_slice(start, end):
    """计算 [start, end) 范围内的逐bar特征, 返回 (end-start, C) float32"""
    # 注意: 需要 start-LOOKBACK 到 end 的原始数据
    s_raw = max(0, start - LOOKBACK)
    e_raw = end
    
    sl = slice(s_raw, e_raw)
    o = O_e[sl]; h = H_e[sl]; l = L_e[sl]; c = C_e[sl]
    bv = BV_e[sl]; sv = SV_e[sl]; f = F_e[sl]
    cb = C_b_a[sl]; bvb = BV_b_a[sl]; svb = SV_b_a[sl]
    
    # log returns
    lr_e = np.zeros(len(c), np.float32)
    lr_e[1:] = np.log(np.maximum(c[1:],1e-8) / np.maximum(c[:-1],1e-8))
    lr_b = np.zeros(len(cb), np.float32)
    lr_b[1:] = np.log(np.maximum(cb[1:],1e-8) / np.maximum(cb[:-1],1e-8))
    
    # range = (high-low)/close
    range_e = (h - l) / (np.maximum(c, 1e-8))
    body_e = (c - o) / (np.maximum(o, 1e-8))
    bsi_e = (bv - sv) / (np.maximum(bv + sv, 1e-8))  # buy/sell imbalance
    
    range_b = (np.maximum(h,1e-8) - np.maximum(l,1e-8)) / (np.maximum(cb, 1e-8))  # 没有BTC的HL, 用ETH的代替
    # 实际上BTC的HL也可以从原始数据拿... 但我们已经del了btc, 算了
    # 简化: BTC用lr, vol, bsi; ETH用更多
    
    bsi_b = (bvb - svb) / (np.maximum(bvb + svb, 1e-8))
    
    # 组装: 10 channels
    # [lr_e, range_e, body_e, bsi_e, funding, lr_b, bsi_b, vol_e_roll, vol_b_roll, pos_30]
    # 简化: 先做 8 channels
    feats = np.stack([lr_e, range_e, body_e, bsi_e, f, lr_b, bsi_b, bv/(sv+1e-8)], axis=1)
    
    # 滚动z-score (rolling 240)
    # 用简单的方法: 对每个channel做指数平滑归一化
    # 但为了快, 直接返回raw features, 模型里或dataset里做norm
    return feats  # (len, 8)


class SeqDataset(Dataset):
    """在线生成 (seq, label), seq shape=(LOOKBACK, CH)"""
    def __init__(self, indices, ts_arr, label_arr, lookback=LOOKBACK):
        self.indices = indices
        self.ts = ts_arr
        self.label = label_arr
        self.L = lookback
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]  # 中心位置 (包含label)
        # 序列窗口: [i-L, i), 对应 label[i] = (C[i+H] > C[i])
        seq_start = i - self.L
        seq_end = i
        # 从原始数组切片 (需要处理边界)
        sl = slice(seq_start, seq_end)
        
        # 计算这个窗口的特征 (简单直接, 不重复计算)
        o = O_e[sl]; h = H_e[sl]; l = L_e[sl]; c = C_e[sl]
        bv = BV_e[sl]; sv = SV_e[sl]; f = F_e[sl]
        cb = C_b_a[sl]; bvb = BV_b_a[sl]; svb = SV_b_a[sl]
        
        lr_e = np.zeros(self.L, np.float32)
        lr_e[1:] = np.log(np.maximum(c[1:],1e-8) / np.maximum(c[:-1],1e-8))
        lr_b = np.zeros(self.L, np.float32)
        lr_b[1:] = np.log(np.maximum(cb[1:],1e-8) / np.maximum(cb[:-1],1e-8))
        
        range_e = (h - l) / (np.maximum(c, 1e-8))
        body_e = (c - o) / (np.maximum(o, 1e-8))
        bsi_e = (bv - sv) / (np.maximum(bv + sv, 1e-8))
        bsi_b = (bvb - svb) / (np.maximum(bvb + svb, 1e-8))
        vol_ratio = bv / (sv + 1e-8)
        
        seq = np.stack([lr_e, range_e, body_e, bsi_e, f, lr_b, bsi_b, vol_ratio], axis=1)
        
        # z-score norm: 用seq自身的统计量 (每个序列独立归一化)
        m = seq.mean(0, keepdims=True)
        s = seq.std(0, keepdims=True) + 1e-6
        seq = (seq - m) / s
        seq = np.clip(seq, -5, 5)
        
        # 转置为 (CH, L) 给 Conv1d
        seq = seq.T.copy()  # (8, 60)
        
        lbl = float(self.label[i])
        return torch.from_numpy(seq), torch.tensor(lbl, dtype=torch.float32)


# 先算norm统计量 (对全训练集随机抽样)
print("Computing train norm stats...", flush=True)
np.random.seed(42)
norm_samp_idx = np.random.choice(tr_idx, min(200_000, len(tr_idx)), replace=False)
# 抽样做norm太慢, 我们在Dataset里做per-sequence norm, 这样更鲁棒, 不需要全局norm
# 跳过全局norm

# ============================================================
# 4. TCN 模型
# ============================================================

class Chomp1d(nn.Module):
    """Causal padding: 右边截断"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dp1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dp2,
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_ch=8, channels=[64, 64, 128, 128], kernel=3, dropout=0.3):
        super().__init__()
        layers = []
        in_c = in_ch
        for i, out_c in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_c, out_c, kernel, dilation, dropout))
            in_c = out_c
        self.network = nn.Sequential(*layers)
        
        # 全局平均池化 + 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        # x: (B, CH, L)
        out = self.network(x)
        return self.head(out)


# ============================================================
# 5. 训练函数
# ============================================================
def train_one_seed(seed, channels, dropout, lr=3e-4, wd=1e-3, label_smooth=0.15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = TCN(in_ch=8, channels=channels, kernel=3, dropout=dropout)
    print(f"\n--- Seed {seed}, params={sum(p.numel() for p in model.parameters()):,} ---", flush=True)
    
    train_ds = SeqDataset(tr_idx, ts, label)
    es_ds = SeqDataset(es_idx, ts, label)
    
    # 用更重的 num_workers 加速CPU数据加载
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    es_loader = DataLoader(es_ds, batch_size=512, shuffle=False, num_workers=0)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    
    best_auc = 0
    best_state = None
    patience = 0
    max_epochs = 20
    
    for epoch in range(max_epochs):
        t_ep = time.time()
        model.train()
        total_loss = 0; n_batches = 0
        
        for xb, yb in train_loader:
            # label smoothing
            y_target = yb * (1 - label_smooth) + 0.5 * label_smooth
            
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_target)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        sched.step()
        
        # Eval on ES
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, yb in es_loader:
                preds = torch.sigmoid(model(xb).squeeze(-1)).numpy()
                all_preds.append(preds)
        all_preds = np.concatenate(all_preds)
        y_true = label[es_idx]
        auc = roc_auc_score(y_true, all_preds)
        
        elapsed = time.time() - t_ep
        print(f"  Epoch {epoch+1:2d}: loss={total_loss/n_batches:.4f}, ES AUC={auc:.4f}, {elapsed:.0f}s, lr={sched.get_last_lr()[0]:.6f}", flush=True)
        
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"  Early stop @ epoch {epoch+1}", flush=True)
                break
    
    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_auc


# ============================================================
# 6. 批量评估 (test set, rolling quantile threshold)
# ============================================================
def evaluate_model(model, eval_indices, ts_arr, label_arr, q_list=[98, 99, 99.5]):
    """返回 dict: auc, topN_results (per q)"""
    model.eval()
    ds = SeqDataset(eval_indices, ts_arr, label_arr)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    
    all_preds = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = torch.sigmoid(model(xb).squeeze(-1)).numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)
    y_true = label[eval_indices]
    auc = roc_auc_score(y_true, all_preds)
    
    # rolling 30-day quantile threshold
    ts_eval = ts_arr[eval_indices]
    results = {}
    for q in q_list:
        thrs = []
        for i in range(len(eval_indices)):
            # 过去30天的pred分位数
            t_now = ts_eval[i]
            t_30d_ago = t_now - 30 * 86400
            mask = (ts_eval[:i] >= t_30d_ago) & (ts_eval[:i] < t_now)
            if mask.sum() >= 100:
                thrs.append(np.percentile(all_preds[mask], q))
            else:
                thrs.append(np.percentile(all_preds[:max(i,1)], q))
        thrs = np.array(thrs)
        
        long_sig = all_preds > thrs
        short_sig = all_preds < (2*all_preds.mean() - thrs)  # 对称阈值 (用mean作为0.5近似)
        
        # 实际: 简化, 只用long信号, accuracy = (pred正 & y正) / pred正
        sig_mask = all_preds > thrs  # 只做long
        if sig_mask.sum() == 0:
            acc = 0; tpd = 0
        else:
            acc = y_true[sig_mask].mean()
            n_days = (ts_eval[-1] - ts_eval[0]) / 86400
            tpd = sig_mask.sum() / n_days
        
        results[f'q{q}'] = {'accuracy': acc, 'tpd': tpd, 'n_trades': int(sig_mask.sum())}
        print(f"    q={q}: acc={acc:.4f}, tpd={tpd:.1f}, n={sig_mask.sum()}", flush=True)
    
    return auc, results


# ============================================================
# 7. 主实验: 多个配置 + ensemble
# ============================================================
configs = [
    # (channels, dropout)
    ([64, 64, 128, 128], 0.3),
    ([32, 64, 128, 128], 0.3),
    ([64, 128, 256, 256], 0.35),
    ([64, 64, 64, 64], 0.25),
]

ensemble_preds = None
y_test = None
ts_test = None

for ci, (ch, dp) in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Config {ci}: channels={ch}, dropout={dp}")
    print(f"{'='*60}", flush=True)
    
    model, best_auc = train_one_seed(
        seed=42+ci*7,
        channels=ch,
        dropout=dp,
        lr=3e-4,
        wd=1e-3,
        label_smooth=0.15,
    )
    print(f"Best ES AUC: {best_auc:.4f}", flush=True)
    
    # Meta-val 评估 (选最好的q)
    print("  Meta-Val eval:", flush=True)
    mv_auc, mv_res = evaluate_model(model, mv_idx, ts, label)
    
    # Test 评估
    print("  Test eval:", flush=True)
    te_auc, te_res = evaluate_model(model, te_idx, ts, label)
    
    # 保存test preds用于ensemble
    model.eval()
    te_ds = SeqDataset(te_idx, ts, label)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    preds_list = []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds_list.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
    te_preds = np.concatenate(preds_list)
    
    if ensemble_preds is None:
        ensemble_preds = te_preds
        y_test = label[te_idx]
        ts_test = ts[te_idx]
    else:
        ensemble_preds = 0.5 * ensemble_preds + 0.5 * te_preds
    
    del model; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# 8. Ensemble 评估
# ============================================================
print(f"\n{'='*60}")
print("ENSEMBLE RESULTS")
print(f"{'='*60}", flush=True)

ens_auc = roc_auc_score(y_test, ensemble_preds)
print(f"Ensemble TE AUC: {ens_auc:.4f}", flush=True)

# rolling quantile
for q in [98, 99, 99.5, 99.7]:
    thrs = []
    for i in range(len(te_idx)):
        t_now = ts_test[i]
        t_30d_ago = t_now - 30 * 86400
        hist_mask = (ts_test[:i] >= t_30d_ago) & (ts_test[:i] < t_now)
        if hist_mask.sum() >= 100:
            thrs.append(np.percentile(ensemble_preds[hist_mask], q))
        else:
            thrs.append(np.percentile(ensemble_preds[:max(i,1)], q))
    thrs = np.array(thrs)
    
    sig_mask = ensemble_preds > thrs
    if sig_mask.sum() > 0:
        acc = y_test[sig_mask].mean()
        n_days = (ts_test[-1] - ts_test[0]) / 86400
        tpd = sig_mask.sum() / n_days
        print(f"  q={q}: acc={acc:.4f}, tpd={tpd:.1f}, n={sig_mask.sum()}")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)
