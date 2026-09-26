"""H=30m Strict Causal Evaluation across ResNet Pattern CNN, Pattern GBDT, JOINT Pool20, and 3-Family Stacking

100% 盲测无前视评估 (盘前前 90 天 P99 历史置信度阈值)
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

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_h30_divisor_cnn_features(df_self, df_other=None):
    p_self = pl.from_pandas(df_self) if isinstance(df_self, pd.DataFrame) else df_self

    exprs = []
    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    buy = pl.col('buy_vol')
    sell = pl.col('sell_vol')
    d_vol = buy - sell
    tot_vol = buy + sell

    range_1m = high - low + 1e-9
    exprs.append(((high - np.maximum(close, pl.col('open'))) / range_1m).fill_null(0.0).alias('upper_wick'))
    exprs.append(((np.minimum(close, pl.col('open')) - low) / range_1m).fill_null(0.0).alias('lower_wick'))
    exprs.append(((close - pl.col('open')).abs() / range_1m).fill_null(0.0).alias('body_ratio'))
    exprs.append((close.log() - close.log().shift(1)).fill_null(0.0).alias('lr1_1m'))
    exprs.append((d_vol / (tot_vol + 1e-9)).fill_null(0.0).alias('cvd_1m'))

    divisors = [2, 3, 5, 10, 15, 30]
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

def to_rank(p):
    return (np.argsort(np.argsort(p)) / (len(p) - 1)).astype(np.float32)

def run_h30_causal_all(symbol='ETH'):
    horizon = 30
    print(f"\n" + "=" * 75, flush=True)
    print(f"  【H=30m 100% 盲测无前视因果评估 (盘前前 90 天 P99 历史阈值)】{symbol}", flush=True)
    print("=" * 75, flush=True)

    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    raw_self = raw_e if symbol == 'ETH' else raw_b
    raw_other = raw_b if symbol == 'ETH' else raw_e

    feats, feat_names = build_h30_divisor_cnn_features(raw_self, raw_other)

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
    mv_m = ts_mask(ts, '2024-09-30', '2025-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    tr_idx = np.where(tr_m)[0][::3]
    es_idx = np.where(es_m)[0]
    mv_idx = np.where(mv_m)[0]
    te_idx = np.where(te_m)[0]

    Xtr, ytr, retr = feats[tr_idx], label[tr_idx], ret_fut[tr_idx]
    Xes, yes = feats[es_idx], label[es_idx]
    Xmv = feats[mv_idx]
    Xte, yte, ts_te = feats[te_idx], label[te_idx], ts[te_idx]

    # 1. ResNet Pattern CNN
    print("\n[1/3] 训练 ResNet Pattern CNN...", flush=True)
    sw_view = sliding_window_view(feats, window_shape=LOOKBACK, axis=0).transpose(0, 2, 1)

    tr_ds = FastPatternDataset(sw_view, label, tr_idx, lookback=LOOKBACK)
    es_ds = FastPatternDataset(sw_view, label, es_idx, lookback=LOOKBACK)
    mv_ds = FastPatternDataset(sw_view, label, mv_idx, lookback=LOOKBACK)
    te_ds = FastPatternDataset(sw_view, label, te_idx, lookback=LOOKBACK)

    BATCH_SIZE = 2048
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    es_loader = DataLoader(es_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    mv_loader = DataLoader(mv_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    in_ch = feats.shape[1]
    cnn_model = PatternResNet(in_channels=in_ch, lookback=LOOKBACK).to(DEVICE)
    opt = torch.optim.AdamW(cnn_model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_es_auc = 0.0
    best_path = os.path.join(config.MODEL_DIR, f"causal_h30_cnn_{symbol}.pt")

    for ep in range(1, 3):
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
    mv_cnn, te_cnn = [], []
    with torch.no_grad():
        for bx, _ in mv_loader: mv_cnn.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
        for bx, _ in te_loader: te_cnn.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
    p_cnn_mv = np.concatenate(mv_cnn)
    p_cnn_te = np.concatenate(te_cnn)

    # 2. Pattern GBDT
    print("[2/3] 训练 Pattern GBDT...", flush=True)
    sw = np.clip(np.abs(retr) * 50, 0.5, 5.0)
    dtr = lgb.Dataset(Xtr, ytr, weight=sw)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    params_lgb = dict(
        objective='binary', metric='auc', learning_rate=0.03, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
        min_child_samples=200, verbose=-1, seed=42
    )
    m_gbdt = lgb.train(params_lgb, dtr, num_boost_round=1000, valid_sets=[des],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
    p_gbdt_mv = m_gbdt.predict(Xmv)
    p_gbdt_te = m_gbdt.predict(Xte)

    # 3. JOINT Pool20
    print("[3/3] 加载 JOINT Pool20...", flush=True)
    from data_store import AssetContext
    ctx = AssetContext(symbol, horizon=30)
    ts_joint_mv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    ts_joint_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    ts_p_mv = ts[mv_idx]
    ts_p_te = ts[te_idx]

    idx_mv_match = np.clip(np.searchsorted(ts_joint_mv, ts_p_mv), 0, len(ts_joint_mv) - 1)
    idx_te_match = np.clip(np.searchsorted(ts_joint_te, ts_p_te), 0, len(ts_joint_te) - 1)

    p_lgb_mv = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_lgb_meta_val_P.npy")).mean(axis=0)[idx_mv_match]
    p_xgb_mv = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_xgb_meta_val_P.npy")).mean(axis=0)[idx_mv_match]
    p_cat_mv = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_cat_meta_val_P.npy")).mean(axis=0)[idx_mv_match]
    p_joint_mv = (p_lgb_mv + p_xgb_mv + p_cat_mv) / 3.0

    p_lgb_te = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_lgb_test_P.npy")).mean(axis=0)[idx_te_match]
    p_xgb_te = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_xgb_test_P.npy")).mean(axis=0)[idx_te_match]
    p_cat_te = np.load(os.path.join(config.DS_DIR, f"JOINT_{symbol}_cat_test_P.npy")).mean(axis=0)[idx_te_match]
    p_joint_te = (p_lgb_te + p_xgb_te + p_cat_te) / 3.0

    # 4. 100% 盲测无前视评估 (盘前前 90 天 P99 历史阈值)
    r_cnn_mv, r_cnn_te = to_rank(p_cnn_mv), to_rank(p_cnn_te)
    r_gbdt_mv, r_gbdt_te = to_rank(p_gbdt_mv), to_rank(p_gbdt_te)
    r_joint_mv, r_joint_te = to_rank(p_joint_mv), to_rank(p_joint_te)

    X_meta_tr = np.column_stack([r_cnn_mv, r_gbdt_mv, r_joint_mv])
    X_meta_te = np.column_stack([r_cnn_te, r_gbdt_te, r_joint_te])
    y_meta_tr = label[mv_idx]

    meta_learner = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    meta_learner.fit(X_meta_tr, y_meta_tr)

    p_stacking = meta_learner.predict_proba(X_meta_te)[:, 1]

    auc_cnn = roc_auc_score(yte, r_cnn_te)
    acc_cnn, min_cnn, bad_cnn, _, acc_m_cnn = eval_r2_causal_daily(r_cnn_te, yte, ts_te)

    auc_gbdt = roc_auc_score(yte, r_gbdt_te)
    acc_gbdt, min_gbdt, bad_gbdt, _, acc_m_gbdt = eval_r2_causal_daily(r_gbdt_te, yte, ts_te)

    auc_joint = roc_auc_score(yte, r_joint_te)
    acc_joint, min_joint, bad_joint, _, acc_m_joint = eval_r2_causal_daily(r_joint_te, yte, ts_te)

    auc_stacking = roc_auc_score(yte, p_stacking)
    acc_stacking, min_stacking, bad_stacking, tpd_stacking, acc_m_stacking = eval_r2_causal_daily(p_stacking, yte, ts_te)
    m_mean_stacking = np.mean(list(acc_m_stacking.values()))

    print("\n" + "=" * 75, flush=True)
    print(f"  【H=30m 100% 盲测无前视结果 ({symbol})】", flush=True)
    print("=" * 75, flush=True)
    print(f"1. ResNet Pattern CNN -> AUC: {auc_cnn:.4f} | 真实Top1%准确率: {acc_cnn*100:.2f}% | 最低月: {min_cnn*100:.1f}% | 坏月: {bad_cnn}个")
    print(f"2. Pattern GBDT       -> AUC: {auc_gbdt:.4f} | 真实Top1%准确率: {acc_gbdt*100:.2f}% | 最低月: {min_gbdt*100:.1f}% | 坏月: {bad_gbdt}个")
    print(f"3. JOINT Pool20       -> AUC: {auc_joint:.4f} | 真实Top1%准确率: {acc_joint*100:.2f}% | 最低月: {min_joint*100:.1f}% | 坏月: {bad_joint}个")
    print("-" * 75, flush=True)
    print(f"★ 3 大模型族 Stacking 元学习器 -> Test AUC: {auc_stacking:.4f}")
    print(f"  • 真实 Top 1% 准确率: {acc_stacking*100:.2f}%")
    print(f"  • 12 个月月均准确率: {m_mean_stacking*100:.2f}%")
    print(f"  • 最低月份准确率: {min_stacking*100:.2f}%")
    print(f"  • 坏月 (<55%) 数量: {bad_stacking} 个")
    print(f"  • 每日交易笔数: {tpd_stacking:.1f} 笔/天")
    print(f"\n  {symbol} H=30m Stacking 逐月明细:\n  {acc_m_stacking}\n")

def main():
    print("=================================================================", flush=True)
    print("  H=30m 100% 盲测无前视因果评估 (ResNet CNN vs GBDT vs JOINT vs Stacking)", flush=True)
    print("=================================================================", flush=True)

    run_h30_causal_all('ETH')
    run_h30_causal_all('BTC')

if __name__ == "__main__":
    main()
