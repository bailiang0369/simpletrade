"""H=5m 精确整除约数阵列 (1m, 5m) ResNet Pattern CNN 专项训练与评估

测试说明:
针对 H = 5m (预测未来 5 分钟涨跌，即 t+5 根 1min K 线价格)，
特征窗口使用 5 的整除约数配比:
  - 1m  (5 根 1m K 线)
  - 5m  (1 根 5m K 线)

验证 ResNet Pattern CNN 单模型在 H=5m 约数特征阵列下的准确率！
"""

import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 构建 H=5m 约数特征引擎 (1m, 5m) ==========
def build_h5_divisor_cnn_features(df_self, df_other=None):
    p_self = pl.from_pandas(df_self) if isinstance(df_self, pd.DataFrame) else df_self

    exprs = []
    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    buy = pl.col('buy_vol')
    sell = pl.col('sell_vol')
    d_vol = buy - sell
    tot_vol = buy + sell

    # 1m 微观基础几何特征
    range_1m = high - low + 1e-9
    exprs.append(((high - np.maximum(close, pl.col('open'))) / range_1m).fill_null(0.0).alias('upper_wick'))
    exprs.append(((np.minimum(close, pl.col('open')) - low) / range_1m).fill_null(0.0).alias('lower_wick'))
    exprs.append(((close - pl.col('open')).abs() / range_1m).fill_null(0.0).alias('body_ratio'))
    exprs.append((close.log() - close.log().shift(1)).fill_null(0.0).alias('lr1_1m'))
    exprs.append((d_vol / (tot_vol + 1e-9)).fill_null(0.0).alias('cvd_1m'))

    # H=5m 精确整除约数: [5]
    divisors = [5]
    for tf in divisors:
        roll_max = high.rolling_max(window_size=tf)
        roll_min = low.rolling_min(window_size=tf)
        range_tf = roll_max - roll_min + 1e-9

        exprs.append(((close - roll_min) / range_tf).fill_null(0.5).alias(f'pos_range_{tf}m'))
        exprs.append(((close - roll_max) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_up_{tf}m'))
        exprs.append(((roll_min - close) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_dn_{tf}m'))
        exprs.append((close.log() - close.log().shift(tf)).fill_null(0.0).alias(f'lr_{tf}m'))
        exprs.append((d_vol.rolling_sum(window_size=tf) / (tot_vol.rolling_sum(window_size=tf) + 1e-9)).fill_null(0.0).alias(f'cvd_{tf}m'))

        stoch_k = ((close - roll_min) / range_tf * 100.0).fill_null(50.0)
        exprs.append((stoch_k.rolling_mean(window_size=max(2, tf // 5)).fill_null(50.0) / 100.0).alias(f'stoch_{tf}m'))

        lr1 = (close.log() - close.log().shift(1)).fill_null(0.0)
        exprs.append((lr1.rolling_std(window_size=tf).fill_null(0.0) * 100.0).alias(f'rvol_{tf}m'))

        ema_tf = close.ewm_mean(span=tf, adjust=False)
        exprs.append(((close - ema_tf) / (close + 1e-9)).fill_null(0.0).alias(f'bias_{tf}m'))

    feat_df = p_self.with_columns(exprs)
    feat_cols = [c for c in feat_df.columns if c not in ['ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']]
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats, feat_cols

# ========== 2. ResNet Pattern CNN ==========
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + res)

class PatternResNet(nn.Module):
    def __init__(self, in_channels, lookback=60):
        super(PatternResNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res1 = ResBlock1D(64)
        self.res2 = ResBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.prep(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.pool(h).squeeze(-1)
        logits = self.fc(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 1e-7, 1.0 - 1e-7)

class FastPatternDataset(Dataset):
    def __init__(self, window_view, labels, valid_indices, lookback=60):
        self.window_indices = valid_indices - (lookback - 1)
        self.window_view = window_view
        self.labels = labels[valid_indices]

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        w_idx = self.window_indices[idx]
        return torch.from_numpy(self.window_view[w_idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

def eval_r2_daily(p, y, ts_arr):
    n = len(p)
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    sec_arr = ts_arr.astype(np.int64)
    day = sec_arr // 86400
    days = np.unique(day)
    sm = np.zeros(n, bool)

    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
        sub = np.where(md)[0]
        sm[sub[np.argsort(-conf[sub])[:kd]]] = True

    sel = np.where(sm)[0]
    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq if (mts == u).sum() >= 5}
    min_a = min(acc_m.values()) if len(acc_m) > 0 else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    overall_acc = float((pred[sel] == y[sel]).mean()) if len(sel) > 0 else 0.0
    tpd = float(sel.size) / len(days)
    return overall_acc, min_a, bad_m, tpd, acc_m

# ========== 3. 运行 H=5m 约数 CNN 训练与评估 ==========
def run_h5_divisor_cnn(symbol='ETH'):
    horizon = 5
    print(f"\n" + "=" * 75, flush=True)
    print(f"  【H=5m 精确整除约数 (1m, 5m) ResNet Pattern CNN 评估】{symbol}", flush=True)
    print("=" * 75, flush=True)

    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    raw_self = raw_e if symbol == 'ETH' else raw_b
    raw_other = raw_b if symbol == 'ETH' else raw_e

    feats_cnn, feat_cols = build_h5_divisor_cnn_features(raw_self, raw_other)

    ts = raw_self['ts'].values.astype(np.int64)
    close = raw_self['close'].values

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-horizon] = (close[horizon:] / close[:-horizon] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    LOOKBACK = 60
    valid = ~np.isnan(ret_fut) & (np.arange(n) >= LOOKBACK)

    def ts_mask(ts_arr, s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts_arr >= a) & (ts_arr < b)

    tr_m = ts_mask(ts, '2020-01-01', '2024-06-30') & valid
    es_m = ts_mask(ts, '2024-06-30', '2024-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    tr_idx = np.where(tr_m)[0][::3]
    es_idx = np.where(es_m)[0]
    te_idx = np.where(te_m)[0]

    sw_view = sliding_window_view(feats_cnn, window_shape=LOOKBACK, axis=0).transpose(0, 2, 1)

    tr_ds = FastPatternDataset(sw_view, label, tr_idx, lookback=LOOKBACK)
    es_ds = FastPatternDataset(sw_view, label, es_idx, lookback=LOOKBACK)
    te_ds = FastPatternDataset(sw_view, label, te_idx, lookback=LOOKBACK)

    BATCH_SIZE = 2048
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    es_loader = DataLoader(es_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    in_ch = feats_cnn.shape[1]
    cnn_model = PatternResNet(in_channels=in_ch, lookback=LOOKBACK).to(DEVICE)
    opt = torch.optim.AdamW(cnn_model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_es_auc = 0.0
    best_path = os.path.join(config.MODEL_DIR, f"h5_div_cnn_{symbol}.pt")

    for ep in range(1, 4):
        cnn_model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            p = cnn_model(bx)
            loss = criterion(p, by)
            loss.backward()
            opt.step()

        cnn_model.eval()
        es_preds = []
        with torch.no_grad():
            for bx, _ in es_loader: es_preds.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
        es_auc = roc_auc_score(label[es_idx], np.concatenate(es_preds))
        if es_auc > best_es_auc:
            best_es_auc = es_auc
            torch.save(cnn_model.state_dict(), best_path)

    if os.path.exists(best_path): cnn_model.load_state_dict(torch.load(best_path))
    cnn_model.eval()
    te_preds = []
    with torch.no_grad():
        for bx, _ in te_loader: te_preds.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
    p_cnn_te = np.concatenate(te_preds)

    auc_cnn = roc_auc_score(label[te_idx], p_cnn_te)
    acc_cnn, min_cnn, bad_cnn, tpd_cnn, acc_m = eval_r2_daily(p_cnn_te, label[te_idx], ts[te_idx])
    m_mean = np.mean(list(acc_m.values()))

    print(f"\n【H=5m 精确约数 (1m, 5m) {symbol} 评估结果】")
    print(f"  • Test AUC: {auc_cnn:.4f}")
    print(f"  • 总体 Top 1% 准确率: {acc_cnn*100:.2f}%")
    print(f"  • 12 个月月均准确率: {m_mean*100:.2f}%")
    print(f"  • 最低月份准确率: {min_cnn*100:.2f}%")
    print(f"  • 坏月 (<55%) 数量: {bad_cnn} 个")
    print(f"  • 逐月明细:\n    {acc_m}\n")

def main():
    print("=================================================================", flush=True)
    print("  H=5m 精确整除约数阵列 (1m, 5m) ResNet Pattern CNN 测试", flush=True)
    print("=================================================================", flush=True)

    run_h5_divisor_cnn('ETH')
    run_h5_divisor_cnn('BTC')

if __name__ == "__main__":
    main()
