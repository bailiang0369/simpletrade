"""Three True Model Families Stacking Meta-Learner Across Horizons (H=15m vs H=30m vs H=60m)

核心对比与评测逻辑:
在同等 3 大真正不同模型族 (PatternResNet CNN + Pattern GBDT + JOINT Pool20) 的 Stacking 架构下，
分别测试不同未来预测目标 H (15m, 30m, 60m)，配合适合该周期的实时多周期特征分辨率进行预测。
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
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
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

# ========== 1. 构建全实时 1min Tick 特征引擎 ==========
def build_realtime_multitf_features(df_self, df_other=None, higher_tf_minutes=[1, 2, 3, 5, 10, 15, 30, 60]):
    p_self = pl.from_pandas(df_self) if isinstance(df_self, pd.DataFrame) else df_self

    exprs = []
    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    buy = pl.col('buy_vol')
    sell = pl.col('sell_vol')
    d_vol = buy - sell
    tot_vol = buy + sell

    exprs.append((close.log() - close.log().shift(1)).fill_null(0.0).alias('lr1_1m'))
    exprs.append((d_vol / (tot_vol + 1e-9)).fill_null(0.0).alias('cvd_1m'))

    for tf in higher_tf_minutes:
        if tf == 1: continue
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

    if df_other is not None:
        p_other = pl.from_pandas(df_other) if isinstance(df_other, pd.DataFrame) else df_other
        ts_s = p_self['ts'].to_numpy()
        ts_o = p_other['ts'].to_numpy()
        close_o = p_other['close'].to_numpy()

        idx_o = np.searchsorted(ts_o, ts_s, side='right') - 1
        idx_o = np.clip(idx_o, 0, len(ts_o) - 1)
        close_o_aligned = close_o[idx_o]

        close_s = p_self['close'].to_numpy()
        ratio = close_s / (close_o_aligned + 1e-12)

        for tf in higher_tf_minutes:
            if tf == 1: continue
            r_ret = np.zeros(len(close_s), dtype=np.float32)
            r_ret[tf:] = np.log(np.maximum(ratio[tf:], 1e-12) / np.maximum(ratio[:-tf], 1e-12)).astype(np.float32)
            exprs.append(pl.Series(f'cross_rret_{tf}m', r_ret))

    feat_df = p_self.with_columns(exprs)
    feat_cols = [c for c in feat_df.columns if c not in ['ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']]
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats, feat_cols

# ========== 2. PatternResNet 网络架构 ==========
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

def to_rank(p):
    return (np.argsort(np.argsort(p)) / (len(p) - 1)).astype(np.float32)

def run_stacking_for_horizon(symbol='ETH', horizon=30):
    if horizon == 15:
        tf_list = [1, 2, 3, 5, 10, 15]
    elif horizon == 30:
        tf_list = [1, 3, 5, 15, 30, 60]
    else:
        tf_list = [1, 3, 5, 15, 30, 60, 120]

    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    if symbol == 'ETH':
        feats, feat_names = build_realtime_multitf_features(raw_e, raw_b, higher_tf_minutes=tf_list)
        ts = raw_e['ts'].values.astype(np.int64)
        close = raw_e['close'].values
    else:
        feats, feat_names = build_realtime_multitf_features(raw_b, raw_e, higher_tf_minutes=tf_list)
        ts = raw_b['ts'].values.astype(np.int64)
        close = raw_b['close'].values

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-horizon] = (close[horizon:] / close[:-horizon] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    LOOKBACK = 60
    valid = ~np.isnan(ret_fut) & (np.arange(n) >= max(LOOKBACK, max(tf_list)))

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

    # --- 1. ResNet Pattern CNN ---
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
    best_path = os.path.join(config.MODEL_DIR, f"cnn_family_{symbol}_h{horizon}.pt")

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

    if os.path.exists(best_path):
        cnn_model.load_state_dict(torch.load(best_path))
    cnn_model.eval()
    mv_cnn, te_cnn = [], []
    with torch.no_grad():
        for bx, _ in mv_loader: mv_cnn.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
        for bx, _ in te_loader: te_cnn.append(cnn_model(bx.to(DEVICE)).cpu().numpy())
    p_cnn_mv = np.concatenate(mv_cnn)
    p_cnn_te = np.concatenate(te_cnn)

    # --- 2. Pattern GBDT ---
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

    # --- 3. JOINT Pool20 跨资产融合族 ---
    from data_store import AssetContext
    ctx = AssetContext(symbol, horizon=30 if horizon not in [15, 30] else horizon)
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

    # --- 4. Stacking 元学习器 ---
    r_cnn_mv, r_cnn_te = to_rank(p_cnn_mv), to_rank(p_cnn_te)
    r_gbdt_mv, r_gbdt_te = to_rank(p_gbdt_mv), to_rank(p_gbdt_te)
    r_joint_mv, r_joint_te = to_rank(p_joint_mv), to_rank(p_joint_te)

    X_meta_tr = np.column_stack([r_cnn_mv, r_gbdt_mv, r_joint_mv])
    X_meta_te = np.column_stack([r_cnn_te, r_gbdt_te, r_joint_te])
    y_meta_tr = label[mv_idx]

    meta_learner = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    meta_learner.fit(X_meta_tr, y_meta_tr)

    p_stacking = meta_learner.predict_proba(X_meta_te)[:, 1]
    auc_stacking = roc_auc_score(yte, p_stacking)
    acc_stacking, min_stacking, bad_stacking, tpd_stacking, acc_m_stacking = eval_r2_daily(p_stacking, yte, ts_te)
    m_mean_stacking = np.mean(list(acc_m_stacking.values()))

    return {
        'symbol': symbol,
        'horizon': horizon,
        'auc': auc_stacking,
        'acc': acc_stacking,
        'm_mean': m_mean_stacking,
        'min_a': min_stacking,
        'bad_m': bad_stacking,
        'tpd': tpd_stacking,
        'acc_m': acc_m_stacking,
    }

def main():
    print("=================================================================", flush=True)
    print("  三大真正核心模型族 (CNN + Pattern GBDT + JOINT) 全预测周期 Stacking 对比", flush=True)
    print("=================================================================\n", flush=True)

    results = []
    for sym in ['ETH', 'BTC']:
        for h in [15, 30, 60]:
            print(f"正在训练与评估 {sym} 未来 H={h}m 涨跌 (3 大模型族 Stacking 元学习器)...", flush=True)
            res = run_stacking_for_horizon(sym, h)
            results.append(res)
            print(f"  --> {sym} H={h}m | Stacking AUC: {res['auc']:.4f} | Top1%准确率: {res['acc']*100:.2f}% | 月均: {res['m_mean']*100:.2f}% | 最低月: {res['min_a']*100:.1f}% | 坏月: {res['bad_m']}个\n", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  【三大模型族 Stacking 元学习器 在不同预测 Horizon (15m vs 30m vs 60m) 下的最终对比表】", flush=True)
    print("=" * 85, flush=True)
    for r in results:
        print(f"[{r['symbol']} 未来{r['horizon']:2d}m涨跌] -> Stacking AUC: {r['auc']:.4f} | 总体Top1%准确率: {r['acc']*100:.2f}% | 12个月月均: {r['m_mean']*100:.2f}% | 最低单月: {r['min_a']*100:.1f}% | 坏月(<55%): {r['bad_m']}个")

if __name__ == "__main__":
    main()
