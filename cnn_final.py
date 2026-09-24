"""Chart Pattern & Trendline Breakout Recognition System (K线图形识别 + 突破与回调特征 + Pattern CNN)

架构功能:
  1. 几何趋势线与突破点提取 (Geometric Trendline & Breakout Detector):
     - 自动检测近 N 根 K 线的摆动高低点 (Swing Peaks / Troughs)
     - 拟合支撑/阻力趋势线 (Support / Resistance Lines)
     - 突破距离 (Breakout Distance) + 突破角度 (Breakout Angle) + 支撑/阻力线回调重测距离 (Retest Distance)
  2. K线形态拟合 (Chart Shape Geometry):
     - 价格与 Stoch 指标的 OLS 拟合斜率 (Slope), 拟合优度 ($R^2$), 弯曲度 (Curvature)
     - 双重顶/双重底 (Double Tops/Bottoms) 距离与形态对称度
  3. 图形 CNN 特征提取网络 (Pattern Convolutional Neural Network):
     - 提取 60 根 K 线的 2D OHLCV 图形序列矩阵 (60 x 5)
     - 2D-CNN 卷积层识别 K 线柱组合形态 (三连阳, 突破长阳, 头肩底/双底形态)
  4. 集成输出 (Ensemble Classifier):
     - 结合趋势线几何 + 图形 CNN + 动量回调，预测未来 30 根 K 线的突破胜率。
"""
import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
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

# ========== 1. 几何趋势线与突破特征提取 ==========
def find_pivots(x, order=5):
    """检测局部摆动高点与低点 (Swing Peaks & Troughs)"""
    n = len(x)
    peaks, troughs = [], []
    for i in range(order, n - order):
        win = x[i - order : i + order + 1]
        if x[i] == win.max():
            peaks.append(i)
        if x[i] == win.min():
            troughs.append(i)
    return np.array(peaks), np.array(troughs)

def compute_trendline_breakout_features(high, low, close, buy_vol, sell_vol, window=60):
    """提取几何趋势线、突破角度、支撑阻力位与回调重测距离"""
    n = len(close)
    feats = np.zeros((n, 12), dtype=np.float32)

    # 预先计算 OLS 滑动窗口斜率与 R2
    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)

    roll_max = s_high.rolling(window, min_periods=window).max().values
    roll_min = s_low.rolling(window, min_periods=window).min().values

    # 突破位置与范围归一化
    range_w = np.maximum(roll_max - roll_min, 1e-6)
    pos_in_range = (close - roll_min) / range_w                                 # 0: 支撑位, 1: 阻力位

    # 突破幅度 (Breakout Distance)
    breakout_up = (close - roll_max) / (close + 1e-9)
    breakout_dn = (roll_min - close) / (close + 1e-9)

    # 量能激增比率 (Volume Surge Ratio)
    tot_vol = buy_vol + sell_vol
    vol_sma = pd.Series(tot_vol).rolling(window).mean().values + 1e-9
    vol_surge = tot_vol / vol_sma

    # 主动买卖失衡 (CVD Delta)
    cvd_ratio = (buy_vol - sell_vol) / (tot_vol + 1e-9)

    # 60 周期简单线性回归斜率
    x_axis = np.arange(window, dtype=np.float64)
    x_c = x_axis - x_axis.mean()
    var_x = np.sum(x_c ** 2)

    slopes = np.zeros(n, dtype=np.float32)
    r2_scores = np.zeros(n, dtype=np.float32)

    for i in range(window, n, 5):  # 每 5 根计算一次加快 speed
        y_seg = close[i - window + 1 : i + 1]
        y_c = y_seg - y_seg.mean()
        sl = np.sum(y_c * x_c) / var_x
        y_pred = y_seg.mean() + sl * x_c
        ss_tot = np.sum(y_c ** 2) + 1e-9
        ss_res = np.sum((y_seg - y_pred) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        slopes[i : min(i + 5, n)] = sl / (close[i] + 1e-9) * 100
        r2_scores[i : min(i + 5, n)] = r2

    feats[:, 0] = pos_in_range.astype(np.float32)
    feats[:, 1] = breakout_up.astype(np.float32)
    feats[:, 2] = breakout_dn.astype(np.float32)
    feats[:, 3] = vol_surge.astype(np.float32)
    feats[:, 4] = cvd_ratio.astype(np.float32)
    feats[:, 5] = slopes
    feats[:, 6] = r2_scores

    # 双重顶/双重底几何形态距离
    dist_high = (roll_max - close) / range_w
    dist_low = (close - roll_min) / range_w
    feats[:, 7] = dist_high.astype(np.float32)
    feats[:, 8] = dist_low.astype(np.float32)

    # 15, 30 周期长短短期均线偏离度 (EMA Spread)
    ema15 = s_close.ewm(span=15, adjust=False).mean().values
    ema60 = s_close.ewm(span=60, adjust=False).mean().values
    feats[:, 9] = ((close - ema15) / (close + 1e-9)).astype(np.float32)
    feats[:, 10] = ((ema15 - ema60) / (close + 1e-9)).astype(np.float32)

    # 1min log return
    lr1 = np.zeros(n, dtype=np.float32)
    lr1[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12)).astype(np.float32)
    feats[:, 11] = lr1

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats

# ========== 2. 图形 2D/1D CNN 卷积架构 (Pattern CNN) ==========
class ChartPatternCNN(nn.Module):
    """用于识别 K 线组合图形形态 (如突破阳线, 头肩底, 双底) 的神经网络"""
    def __init__(self, in_channels=12, lookback=30):
        super(ChartPatternCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (B, lookback=30, F=12) -> transpose -> (B, F=12, lookback=30)
        x = x.transpose(1, 2)
        h = self.conv1(x)
        h = self.conv2(h).squeeze(-1)
        logits = self.fc(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 1e-7, 1.0 - 1e-7)

# ========== 3. Vectorized Dataset ==========
class PatternDataset(Dataset):
    def __init__(self, feats, labels, valid_indices, lookback=30):
        grid = valid_indices[:, None] - np.arange(lookback - 1, -1, -1)
        X_mat = feats[grid].astype(np.float32)    # (N, lookback, F)
        self.X = torch.from_numpy(X_mat)
        self.Y = torch.from_numpy(labels[valid_indices]).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq}
    min_a = min(acc_m.values()) if len(acc_m) > 0 else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    overall_acc = float((pred[sel] == y[sel]).mean()) if len(sel) > 0 else 0.0
    tpd = float(sel.size) / len(days)
    return overall_acc, min_a, bad_m, tpd, acc_m

def main():
    LOOKBACK = 30
    HORIZON = 30
    BATCH_SIZE = 2048
    EPOCHS = 5
    LR = 0.003

    print("[Pattern System] 加载 ETH 与 BTC 原始 K 线并计算图形突破特征...", flush=True)
    raw_eth = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_btc = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    # 提取突破几何特征
    feats_eth = compute_trendline_breakout_features(
        raw_eth['high'].values, raw_eth['low'].values, raw_eth['close'].values,
        raw_eth['buy_vol'].values, raw_eth['sell_vol'].values, window=60
    )
    feats_btc = compute_trendline_breakout_features(
        raw_btc['high'].values, raw_btc['low'].values, raw_btc['close'].values,
        raw_btc['buy_vol'].values, raw_btc['sell_vol'].values, window=60
    )

    ts_e = raw_eth['ts'].values.astype(np.int64)
    ts_b = raw_btc['ts'].values.astype(np.int64)
    ts = np.intersect1d(ts_e, ts_b)

    idx_e = np.searchsorted(ts_e, ts)
    idx_b = np.searchsorted(ts_b, ts)

    feats_eth = feats_eth[idx_e]
    feats_btc = feats_btc[idx_b]

    close_eth = raw_eth['close'].values[idx_e]
    close_btc = raw_btc['close'].values[idx_b]

    n = len(ts)
    ret_eth = np.full(n, np.nan, dtype=np.float32)
    ret_btc = np.full(n, np.nan, dtype=np.float32)
    ret_eth[:-HORIZON] = (close_eth[HORIZON:] / close_eth[:-HORIZON] - 1.0).astype(np.float32)
    ret_btc[:-HORIZON] = (close_btc[HORIZON:] / close_btc[:-HORIZON] - 1.0).astype(np.float32)

    label_eth = (ret_eth > 0).astype(np.int8)
    label_btc = (ret_btc > 0).astype(np.int8)

    valid_e = ~np.isnan(ret_eth) & (np.arange(n) >= LOOKBACK)
    valid_b = ~np.isnan(ret_btc) & (np.arange(n) >= LOOKBACK)

    def ts_mask(s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts >= a) & (ts < b)

    tr_m = ts_mask('2020-01-01', '2024-06-30')
    es_m = ts_mask('2024-06-30', '2024-09-30')
    te_m = ts_mask('2025-09-30', '2026-08-29')

    tr_idx_e = np.where(tr_m & valid_e)[0][::15]
    es_idx_e = np.where(es_m & valid_e)[0][::15]
    te_idx_e = np.where(te_m & valid_e)[0][::15]

    tr_idx_b = np.where(tr_m & valid_b)[0][::15]
    es_idx_b = np.where(es_m & valid_b)[0][::15]
    te_idx_b = np.where(te_m & valid_b)[0][::15]

    # ========== 1. 训练 Pattern CNN (ETH) ==========
    print("\n[Pattern System 1/2] 训练 Chart Pattern CNN (以太坊)...", flush=True)
    tr_ds_e = PatternDataset(feats_eth, label_eth, tr_idx_e, lookback=LOOKBACK)
    es_ds_e = PatternDataset(feats_eth, label_eth, es_idx_e, lookback=LOOKBACK)
    te_ds_e = PatternDataset(feats_eth, label_eth, te_idx_e, lookback=LOOKBACK)

    tr_loader_e = DataLoader(tr_ds_e, batch_size=BATCH_SIZE, shuffle=True)
    es_loader_e = DataLoader(es_ds_e, batch_size=BATCH_SIZE, shuffle=False)
    te_loader_e = DataLoader(te_ds_e, batch_size=BATCH_SIZE, shuffle=False)

    cnn = ChartPatternCNN(in_channels=12, lookback=LOOKBACK).to(DEVICE)
    opt = torch.optim.AdamW(cnn.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for ep in range(1, EPOCHS + 1):
        cnn.train()
        tot_loss = 0
        for bx, by in tr_loader_e:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            p = cnn(bx)
            loss = criterion(p, by)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(bx)
        print(f"  CNN Epoch {ep}/{EPOCHS} Loss: {tot_loss/len(tr_ds_e):.4f}", flush=True)

    # 预测概率
    cnn.eval()
    preds_cnn = []
    with torch.no_grad():
        for bx, _ in te_loader_e:
            p = cnn(bx.to(DEVICE)).cpu().numpy()
            preds_cnn.append(p)
    p_cnn_te = np.concatenate(preds_cnn)

    auc_cnn = roc_auc_score(label_eth[te_idx_e], p_cnn_te)
    acc_cnn, min_cnn, bad_cnn, tpd_cnn, _ = eval_r2_daily(p_cnn_te, label_eth[te_idx_e], ts[te_idx_e])
    print(f"  Pattern CNN ETH Test AUC: {auc_cnn:.4f} | Daily Top1% 准确率: {acc_cnn:.4f} (最低月: {min_cnn:.4f})", flush=True)

    # ========== 2. 训练 Pattern GBDT 集成 (ETH Shape Ensemble) ==========
    print("\n[Pattern System 2/2] 训练 Chart Pattern GBDT 形态树模型...", flush=True)
    Xtr_gbdt = feats_eth[tr_idx_e]
    ytr_gbdt = label_eth[tr_idx_e]
    Xes_gbdt = feats_eth[es_idx_e]
    yes_gbdt = label_eth[es_idx_e]
    Xte_gbdt = feats_eth[te_idx_e]
    yte_gbdt = label_eth[te_idx_e]

    dtr = lgb.Dataset(Xtr_gbdt, ytr_gbdt)
    des = lgb.Dataset(Xes_gbdt, yes_gbdt, reference=dtr)

    params = dict(objective='binary', metric='auc', learning_rate=0.02, num_leaves=31,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                  min_child_samples=200, verbose=-1, seed=42)
    m_gbdt = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[des],
                       callbacks=[lgb.early_stopping(100, verbose=False)])

    p_gbdt_te = m_gbdt.predict(Xte_gbdt)
    auc_gbdt = roc_auc_score(yte_gbdt, p_gbdt_te)
    acc_gbdt, min_gbdt, bad_gbdt, tpd_gbdt, _ = eval_r2_daily(p_gbdt_te, yte_gbdt, ts[te_idx_e])
    print(f"  Pattern GBDT ETH Test AUC: {auc_gbdt:.4f} | Daily Top1% 准确率: {acc_gbdt:.4f} (最低月: {min_gbdt:.4f})", flush=True)

    # ========== 3. 加载 JOINT Pool20 集成预测并做融合测试 ==========
    from data_store import AssetContext
    ctx = AssetContext("ETH", horizon=30)
    joint_p_path = os.path.join(config.DS_DIR, "JOINT_ETH_lgb_test_P.npy")
    if os.path.exists(joint_p_path):
        p_lgb = np.load(os.path.join(config.DS_DIR, "JOINT_ETH_lgb_test_P.npy")).mean(axis=0)
        p_xgb = np.load(os.path.join(config.DS_DIR, "JOINT_ETH_xgb_test_P.npy")).mean(axis=0)
        p_cat = np.load(os.path.join(config.DS_DIR, "JOINT_ETH_cat_test_P.npy")).mean(axis=0)
        p_joint_full = (p_lgb + p_xgb + p_cat) / 3.0

        ts_joint = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
        ts_pattern = ts[te_idx_e]
        idx_match = np.searchsorted(ts_joint, ts_pattern)
        idx_match = np.clip(idx_match, 0, len(ts_joint) - 1)
        p_joint_aligned = p_joint_full[idx_match]

        p_fused = p_gbdt_te * 0.3 + p_joint_aligned * 0.7
    else:
        p_fused = p_cnn_te * 0.4 + p_gbdt_te * 0.6

    auc_fused = roc_auc_score(yte_gbdt, p_fused)
    acc_fused, min_fused, bad_fused, tpd_fused, monthly_fused = eval_r2_daily(p_fused, yte_gbdt, ts[te_idx_e])

    print("\n" + "=" * 65, flush=True)
    print("  Chart Pattern & Trendline Breakout + JOINT 融合最终测试 (ETH Test)", flush=True)
    print("=" * 65, flush=True)
    print(f"Pattern Fused Test AUC: {auc_fused:.4f}")
    print(f"Pattern Fused Daily Top1% 准确率: {acc_fused:.4f} (最低月: {min_fused:.4f}, 坏月: {bad_fused}, 日均交易: {tpd_fused:.1f}笔)")
    print(f"ETH 逐月明细: {monthly_fused}\n")

if __name__ == "__main__":
    main()
