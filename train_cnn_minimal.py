"""
Minimal CNN: close + CVD + 4 derived feats → 1D Dilated CNN → direction predict.

6 channels per window (SEG=64, 1min bars):
  [0] close_ret   = log(close / close_t0)           window内价格形态
  [1] cvd_diff    = cumsum(buy-sell) / total_vol      窗口内多空力量累积
  [2] buy_ratio   = log(buy_vol / sell_vol)           每个bar主动买卖比
  [3] vol_norm    = log(buy+sell) - win_mean          窗口内成交量relative
  [4] funding     = funding_rate (global z-score)     资金费率
  [5] btc_ret     = log(BTC_close / BTC_close_t0)     BTC cross-asset

严格时间切分（config.SPLITS）, H=15 方向预测.
"""
import os, sys, time, gc, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

import config

# ============ 超参 ============
SEG = 64            # 窗口长度（分钟）
STRIDE = 16         # 滑窗步长
H = config.HORIZON_MIN  # 预测 H 分钟后的方向（默认 15）
BATCH = 512
EPOCHS = 15
LR = 3e-3
WD = 1e-4
DEVICE = torch.device('cpu')  # 3 核 CPU

# ============ 1. 加载 raw 数据 ============
t0 = time.time()
print(f'[1] 加载 raw 数据 ...', flush=True)

eth = pd.read_parquet(os.path.join(config.DS_DIR, 'raw_ETH.parquet'))
btc = pd.read_parquet(os.path.join(config.DS_DIR, 'raw_BTC.parquet'))

# 时间对齐（ETH 少 1 行）
eth_idx = pd.Index(eth['ts'].values)
btc_reorder = btc['ts'].values.searchsorted(eth_idx.values, side='right') - 1
btc_reorder = np.clip(btc_reorder, 0, len(btc) - 1)

close = eth['close'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
fund = eth['funding'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float32)[btc_reorder]

N = len(close)
print(f'  ETH N={N:,}  BTC aligned  funding in [{fund.min():.4f}, {fund.max():.4f}]', flush=True)

# ============ 2. 构建窗口化特征 ============
print(f'\n[2] 构建 {SEG}-min 窗口 (stride={STRIDE}) ...', flush=True)

# 全局 z-score 统计（只用 train 段, 防止泄露）
_tr_end_ts = int(pd.Timestamp(config.TRAIN_END, tz='UTC').timestamp())
_tr_mask = ts < _tr_end_ts
fund_mean = fund[_tr_mask].mean()
fund_std = max(fund[_tr_mask].std(), 1e-6)

# 预计算 bar-level 特征
bar_br = np.log(np.maximum(buy_v / np.maximum(sell_v, 1.0), 1e-6))  # log(buy/sell)
bar_tv = np.log(np.maximum(buy_v + sell_v, 1.0))                    # log(total vol)
bar_fund = (fund - fund_mean) / fund_std                             # z-scored funding
bar_bs = buy_v - sell_v                                              # buy-sell raw

del buy_v, sell_v, fund; gc.collect()

# 构建窗口 —— 逐步填充避免大数组 OOM
# 最终 shape: (num_windows, 6, SEG)
win_list = []
target_list = []
win_ts_list = []

cnt = 0
for start in range(0, N - SEG - H, STRIDE):
    end = start + SEG
    # label: close[end+H] vs close[end]
    if end + H >= N: break
    target = 1 if close[end + H] > close[end] else 0

    # window-level normalize: 以 start 点为锚
    c0 = close[start]
    bc0 = btc_c[start]

    w = np.zeros((6, SEG), np.float32)

    # [0] close_ret = log(close / c0)
    w[0] = np.log(close[start:end] / max(c0, 1e-6))

    # [1] cvd_diff = cumsum(buy-sell) in window / window_total_vol
    bs_win = bar_bs[start:end]
    cum_bs = np.cumsum(bs_win)
    tot_vol = max(np.abs(cum_bs[-1]), 1.0)
    w[1] = cum_bs / tot_vol  # 归一化到 [-1, 1] 附近

    # [2] buy_ratio = log(buy/sell) per bar
    w[2] = bar_br[start:end]

    # [3] vol_norm = log(vol) - window_mean
    tv_win = bar_tv[start:end]
    w[3] = tv_win - tv_win.mean()

    # [4] funding (z-scored)
    w[4] = bar_fund[start:end]

    # [5] btc_ret = log(btc_close / btc0)
    w[5] = np.log(btc_c[start:end] / max(bc0, 1e-6))

    # NaN 安全检查
    if np.isnan(w).any():
        continue

    win_list.append(w)
    target_list.append(target)
    win_ts_list.append(ts[end])

    cnt += 1
    if cnt % 100_000 == 0:
        elapsed = time.time() - t0
        print(f'  ... {cnt:,} windows ({elapsed:.0f}s)', flush=True)

X_all = np.array(win_list, np.float32)
y_all = np.array(target_list, np.int64)
ts_all = np.array(win_ts_list, np.int64)
del win_list, target_list, win_ts_list, bar_br, bar_tv, bar_bs, close, btc_c; gc.collect()

print(f'  ✅ windows: {X_all.shape}  pos_rate={y_all.mean():.3f}  ({time.time()-t0:.0f}s)', flush=True)

# ============ 3. 严格时间切分 ============
print(f'\n[3] 时间切分 ...', flush=True)

def _ts_mask(start_str, end_str):
    a = int(pd.Timestamp(start_str, tz='UTC').timestamp())
    b = int(pd.Timestamp(end_str, tz='UTC').timestamp())
    return (ts_all >= a) & (ts_all < b)

tr_mask = _ts_mask(*config.SPLITS['train'])
es_mask = _ts_mask(*config.SPLITS['early_stop'])
te_mask = _ts_mask(*config.SPLITS['test'])

X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_es, y_es = X_all[es_mask], y_all[es_mask]
X_te, y_te = X_all[te_mask], y_all[te_mask]
del X_all, y_all, ts_all, tr_mask, es_mask, te_mask; gc.collect()

print(f'  TR={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}')
print(f'  train pos_rate={y_tr.mean():.3f}  test pos_rate={y_te.mean():.3f}', flush=True)

# ============ 4. 1D Dilated CNN ============
class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(ch_in, ch_out, 1, bias=False) if ch_in != ch_out else None

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        if self.skip:
            out = out + self.skip(x)
        else:
            out = out + x
        return out

class MinimalCNN(nn.Module):
    def __init__(self, n_feat=6, seg=64):
        super().__init__()
        # stem: 6ch -> 64ch
        self.stem = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # dilated blocks: 感受野逐步扩大
        self.b1 = DilatedBlock(64, 128, dilation=2)   # RF=5
        self.b2 = DilatedBlock(128, 128, dilation=4)  # RF=13
        self.b3 = DilatedBlock(128, 128, dilation=8)  # RF=29
        self.b4 = DilatedBlock(128, 128, dilation=16) # RF=61
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

model = MinimalCNN(n_feat=6, seg=SEG)
n_params = sum(p.numel() for p in model.parameters())
print(f'\n[4] 模型参数量: {n_params:,} ({n_params/1e6:.2f}M)', flush=True)

# ============ 5. 训练 ============
print(f'\n[5] 训练 {EPOCHS} epochs ...', flush=True)

def to_ds(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

ds_tr = to_ds(X_tr, y_tr)
ds_es = to_ds(X_es, y_es)
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
dl_es = DataLoader(ds_es, batch_size=BATCH, shuffle=False, num_workers=0)

del X_tr, y_tr, X_es, y_es; gc.collect()

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_es_loss = float('inf')
best_state = None
patience = 5
no_improve = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []
    for xb, yb in dl_tr:
        optim.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb.float())
        loss.backward()
        optim.step()
        tr_losses.append(loss.item())

    sched.step()
    tr_loss = np.mean(tr_losses)

    # ES eval
    model.eval()
    es_losses = []
    with torch.no_grad():
        for xb, yb in dl_es:
            logits = model(xb).squeeze(-1)
            es_losses.append(criterion(logits, yb.float()).item())
    es_loss = np.mean(es_losses)

    if es_loss < best_es_loss - 1e-5:
        best_es_loss = es_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    auc_str = ''
    if es_losses:  # 快速算一次 ES AUC
        model.eval()
        pv_es_all, y_es_all = [], []
        with torch.no_grad():
            for xb, yb in dl_es:
                pv_es_all.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
                y_es_all.append(yb.numpy())
        pv_es = np.concatenate(pv_es_all)
        y_es_np = np.concatenate(y_es_all)
        try:
            auc_str = f'  es_auc={roc_auc_score(y_es_np, pv_es):.4f}'
        except:
            pass

    print(f'  epoch {ep:2d}  tr_loss={tr_loss:.4f}  es_loss={es_loss:.4f}{auc_str}  lr={sched.get_last_lr()[0]:.2e}  '
          f'{"★" if no_improve == 0 else ""}  ({time.time()-t0:.0f}s)', flush=True)

    if no_improve >= patience:
        print(f'  ⏹ early stop at epoch {ep}', flush=True)
        break

# ============ 6. Test 评估 ============
print(f'\n[6] Test 评估 ...', flush=True)

if best_state:
    model.load_state_dict(best_state)
    print('  ✅ loaded best ES state')

model.eval()
ds_te = to_ds(X_te, y_te)
dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)

pv_list, y_list = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        pv_list.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
        y_list.append(yb.numpy())

pv_te = np.concatenate(pv_list)
y_te_np = np.concatenate(y_list)

auc = roc_auc_score(y_te_np, pv_te)
print(f'\n  📊 Test AUC = {auc:.4f}  (H={H})', flush=True)

# top-k accuracy
for pct in [0.005, 0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(pv_te) * pct))
    top_idx = pv_te.argsort()[-k:]
    acc = y_te_np[top_idx].mean() * 100
    # tpd = trades per day
    days_in_test = len(pd.date_range(config.META_VAL_END, periods=10000, freq='D', tz='UTC'))
    # 简化: test 区间 2025-09-30 → ~2026-09-21 ≈ 356天 (数据到 ts 最新)
    # 实际用 ts_all 反推
    n_days = 356  # 近似
    tpd = (k / n_days) if n_days > 0 else 0
    flag = '✅' if (pct == 0.01 and acc >= 65) or (pct == 0.005 and acc >= 65) else ''
    print(f'  top-{pct*100:.1f}%: acc={acc:.1f}%  tpd≈{tpd:.0f}  {flag}', flush=True)

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
