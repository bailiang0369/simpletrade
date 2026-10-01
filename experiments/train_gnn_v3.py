"""GNN v3 极简增强版: Focal Loss (焦点损失) + 时段感知图注意力 (Hour-Gated GAT)

针对 GNN v1 的优化诊断:
  1. Focal Loss (焦点衰减损失): 传统的 BCELoss 关注 99% 普通样本(P=0.5 附近)，
     使用 Focal Loss (gamma=2.0) 强迫梯度集中于尾部高置信度 (Top 1%) 样本的正确分类；
  2. 极简特征集 (保留用户效果最好的 4 维指标):
     - Price Log Return
     - Stoch (60, 2, 1)
     - CVD Ratio
     - MACD Hist
     - + Hour Sin/Cos (时段感知嵌入)
  3. 过滤低效 UTC 时段 (06:00-08:00 震荡盘整期)，进一步提升 Daily Top 1% 信号胜率与月度稳定性。
"""
import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. Focal Loss 损失函数 ==========
class BinaryFocalLoss(nn.Module):
    """专注于高难/尾部置信度样本的 Binary Focal Loss"""
    def __init__(self, alpha=0.5, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        loss = -focal_weight * torch.log(pt)
        return loss.mean()

# ========== 2. 极简+时段特征构建 (Polars 提速) ==========
def build_gnn_v3_features(df_target, df_other):
    p_target = pl.from_pandas(df_target[['ts', 'close', 'buy_vol', 'sell_vol']])
    p_other = pl.from_pandas(df_other[['ts', 'close']])

    ts_t = p_target['ts'].to_numpy()
    ts_o = p_other['ts'].to_numpy()
    close_o = p_other['close'].to_numpy()

    idx_o = np.searchsorted(ts_o, ts_t, side='right') - 1
    idx_o = np.clip(idx_o, 0, len(ts_o) - 1)
    close_o_aligned = close_o[idx_o]

    exprs = []
    # 1. Price Log Return
    exprs.append((pl.col('close').log() - pl.col('close').log().shift(1)).fill_null(0.0).alias('lr_1'))

    # 2. Stoch (60, 2, 1)
    lw = pl.col('close').rolling_min(window_size=60)
    hw = pl.col('close').rolling_max(window_size=60)
    fast_k = (pl.col('close') - lw) / (hw - lw + 1e-9) * 100
    slow_k = (fast_k.rolling_mean(window_size=2).fill_null(50.0) / 100.0).alias('stoch_60')
    exprs.append(slow_k)

    # 3. CVD (30-bar)
    d = pl.col('buy_vol') - pl.col('sell_vol')
    tot = pl.col('buy_vol') + pl.col('sell_vol')
    cvd_30 = (d.rolling_sum(window_size=30) / (tot.rolling_sum(window_size=30) + 1e-9)).fill_null(0.0).alias('cvd_30')
    exprs.append(cvd_30)

    # 4. MACD Hist (12, 26, 9)
    e_fast = pl.col('close').ewm_mean(span=12, adjust=False)
    e_slow = pl.col('close').ewm_mean(span=26, adjust=False)
    macd = e_fast - e_slow
    signal = macd.ewm_mean(span=9, adjust=False)
    hist = (macd - signal).fill_null(0.0).alias('macd_hist')
    exprs.append(hist)

    feat_df = p_target.with_columns(exprs)
    feat_cols = ['lr_1', 'stoch_60', 'cvd_30', 'macd_hist']
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)

    # 标准化 MACD
    macd_std = np.std(feats[:, 3]) + 1e-9
    feats[:, 3] /= macd_std

    # 5. 时段编码
    hour = (ts_t % 86400) // 3600
    h_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)[:, None]
    h_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)[:, None]

    # 6. 跨币对数收益率差
    close_t = p_target['close'].to_numpy()
    ratio = np.log(np.maximum(close_t, 1e-12) / np.maximum(close_o_aligned, 1e-12))
    r_ret = np.zeros(len(close_t), dtype=np.float32)
    r_ret[15:] = (ratio[15:] - ratio[:-15]).astype(np.float32)
    r_mat = r_ret[:, None]

    all_feats = np.hstack([feats, h_sin, h_cos, r_mat]).astype(np.float32)
    return all_feats

def load_and_prepare_data_v3(horizon=30, lookback=30):
    raw_eth = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_btc = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    ts_eth = raw_eth['ts'].values.astype(np.int64)
    ts_btc = raw_btc['ts'].values.astype(np.int64)
    common_ts = np.intersect1d(ts_eth, ts_btc)

    idx_eth = np.searchsorted(ts_eth, common_ts)
    idx_btc = np.searchsorted(ts_btc, common_ts)

    feats_eth_full = build_gnn_v3_features(raw_eth, raw_btc)
    feats_btc_full = build_gnn_v3_features(raw_btc, raw_eth)

    feats_eth = feats_eth_full[idx_eth]
    feats_btc = feats_btc_full[idx_btc]

    close_eth = raw_eth['close'].values[idx_eth]
    close_btc = raw_btc['close'].values[idx_btc]
    ts = common_ts

    del raw_eth, raw_btc, feats_eth_full, feats_btc_full
    gc.collect()

    n = len(ts)
    ret_eth = np.full(n, np.nan, dtype=np.float32)
    ret_btc = np.full(n, np.nan, dtype=np.float32)
    ret_eth[:-horizon] = (close_eth[horizon:] / close_eth[:-horizon] - 1.0).astype(np.float32)
    ret_btc[:-horizon] = (close_btc[horizon:] / close_btc[:-horizon] - 1.0).astype(np.float32)

    label_eth = (ret_eth > 0).astype(np.int8)
    label_btc = (ret_btc > 0).astype(np.int8)

    valid = ~np.isnan(ret_eth) & ~np.isnan(ret_btc) & (np.arange(n) >= lookback)
    return ts, feats_eth, feats_btc, label_eth, label_btc, ret_eth, ret_btc, valid

# ========== 3. 时空图神经网络 (GNN v3) ==========
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = self.W(h)  # (B, N, out_features)

        Wh1 = Wh.repeat_interleave(N, dim=1)
        Wh2 = Wh.repeat(1, N, 1)
        comb = torch.cat([Wh1, Wh2], dim=-1).view(B, N, N, -1)

        e = self.leakyrelu(self.a(comb)).squeeze(-1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class SpatialTemporalGNNv3(nn.Module):
    def __init__(self, in_dim=7, lookback=30, hidden_dim=64):
        super(SpatialTemporalGNNv3, self).__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim

        self.time_conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.gat = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj):
        B, N, L, F_dim = x.shape
        x_flat = x.view(B * N, L, F_dim).transpose(1, 2)
        node_embed = self.time_conv(x_flat).squeeze(-1).view(B, N, self.hidden_dim)

        g1 = self.norm(self.gat(node_embed, adj) + node_embed)
        logits = self.fc(g1).squeeze(-1)
        return torch.sigmoid(logits)

# ========== 4. Dataset 与评估 ==========
class ZeroCopyGNNDataset(Dataset):
    def __init__(self, t_eth, t_btc, label_eth, label_btc, valid_indices, lookback=30):
        adj_idx = valid_indices - lookback
        sub_e = t_eth[adj_idx]
        sub_b = t_btc[adj_idx]
        self.X = torch.stack([sub_e, sub_b], dim=1).float()
        self.Y = torch.from_numpy(np.stack([label_eth[valid_indices], label_btc[valid_indices]], axis=1)).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def evaluate_gnn(model, loader, adj_tensor):
    model.eval()
    preds_eth, preds_btc = [], []
    ys_eth, ys_btc = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
            probs = model(bx, adj_tensor)
            preds_eth.append(probs[:, 0].cpu().numpy())
            preds_btc.append(probs[:, 1].cpu().numpy())
            ys_eth.append(by[:, 0].numpy())
            ys_btc.append(by[:, 1].numpy())

    p_eth = np.concatenate(preds_eth); y_eth = np.concatenate(ys_eth)
    p_btc = np.concatenate(preds_btc); y_btc = np.concatenate(ys_btc)

    auc_eth = roc_auc_score(y_eth, p_eth)
    auc_btc = roc_auc_score(y_btc, p_btc)
    return p_eth, p_btc, auc_eth, auc_btc

def eval_r2_daily(p, y, ts_arr):
    n = len(p)
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    sec_arr = ts_arr.astype(np.int64)
    day = sec_arr // 86400
    days = np.unique(day)
    sm = np.zeros(n, bool)

    # 时段软过滤: UTC 06:00-08:00 低流动性震荡时段小幅降权置信度
    hour = (sec_arr % 86400) // 3600
    chop_mask = (hour >= 6) & (hour <= 8)
    conf_adj = conf.copy()
    conf_adj[chop_mask] *= 0.9  # 降低低效时段的抢单权重

    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
        sub = np.where(md)[0]
        sm[sub[np.argsort(-conf_adj[sub])[:kd]]] = True

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
    BATCH_SIZE = 2048
    EPOCHS = 4
    LR = 0.003
    SEEDS = [42, 49, 56]

    ts, feats_eth, feats_btc, label_eth, label_btc, ret_eth, ret_btc, valid = load_and_prepare_data_v3(horizon=30, lookback=LOOKBACK)

    def ts_mask(s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts >= a) & (ts < b)

    tr_m = ts_mask('2020-01-01', '2024-06-30') & valid
    es_m = ts_mask('2024-06-30', '2024-09-30') & valid
    mv_m = ts_mask('2024-09-30', '2025-09-30') & valid
    te_m = ts_mask('2025-09-30', '2026-08-29') & valid

    tr_idx = np.where(tr_m)[0][::15]
    es_idx = np.where(es_m)[0][::15]
    mv_idx = np.where(mv_m)[0][::15]
    te_idx = np.where(te_m)[0][::15]

    print(f"[GNN v3 Focal 数据集] Train={len(tr_idx):,}, EarlyStop={len(es_idx):,}, MetaVal={len(mv_idx):,}, Test={len(te_idx):,}", flush=True)

    t_eth = torch.from_numpy(feats_eth).unfold(0, LOOKBACK, 1).transpose(1, 2)
    t_btc = torch.from_numpy(feats_btc).unfold(0, LOOKBACK, 1).transpose(1, 2)

    tr_ds = ZeroCopyGNNDataset(t_eth, t_btc, label_eth, label_btc, tr_idx, lookback=LOOKBACK)
    es_ds = ZeroCopyGNNDataset(t_eth, t_btc, label_eth, label_btc, es_idx, lookback=LOOKBACK)
    te_ds = ZeroCopyGNNDataset(t_eth, t_btc, label_eth, label_btc, te_idx, lookback=LOOKBACK)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    es_loader = DataLoader(es_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    adj = torch.ones((2, 2), dtype=torch.float32).to(DEVICE)
    in_dim = feats_eth.shape[1]

    all_preds_eth, all_preds_btc = [], []

    for seed in SEEDS:
        set_seed(seed)
        model = SpatialTemporalGNNv3(in_dim=in_dim, lookback=LOOKBACK, hidden_dim=64).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = BinaryFocalLoss(gamma=2.0)

        best_es_auc = 0.0
        best_path = os.path.join(config.MODEL_DIR, f"gnn_v3_focal_seed{seed}.pt")

        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for bx, by in tr_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                probs = model(bx, adj)
                loss = criterion(probs, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            scheduler.step()

            if ep == EPOCHS:
                _, _, es_auc_eth, es_auc_btc = evaluate_gnn(model, es_loader, adj)
                mean_es_auc = (es_auc_eth + es_auc_btc) / 2.0
                if mean_es_auc > best_es_auc:
                    best_es_auc = mean_es_auc
                    torch.save(model.state_dict(), best_path)

        # 加载最佳并预测 Test
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))
        p_e, p_b, te_auc_e, te_auc_b = evaluate_gnn(model, te_loader, adj)
        all_preds_eth.append(p_e)
        all_preds_btc.append(p_b)
        print(f"  [GNN v3 Focal Seed{seed:2d}] Done in {time.time()-t0:.0f}s | Test AUC: ETH={te_auc_e:.4f}, BTC={te_auc_b:.4f}", flush=True)

    # 集成 5-Seed 预测
    avg_p_eth = np.mean(all_preds_eth, axis=0)
    avg_p_btc = np.mean(all_preds_btc, axis=0)

    y_eth_te = label_eth[te_idx]
    y_btc_te = label_btc[te_idx]
    ts_te = ts[te_idx]

    auc_e = roc_auc_score(y_eth_te, avg_p_eth)
    auc_b = roc_auc_score(y_btc_te, avg_p_btc)

    acc_e, min_e, bad_e, tpd_e, monthly_e = eval_r2_daily(avg_p_eth, y_eth_te, ts_te)
    acc_b, min_b, bad_b, tpd_b, monthly_b = eval_r2_daily(avg_p_btc, y_btc_te, ts_te)

    print("\n" + "=" * 65, flush=True)
    print("  GNN v3 Focal Loss (5-Seed 集成) 优化增强版 Test 评估结果", flush=True)
    print("=" * 65, flush=True)
    print(f"ETH GNN v3 Test AUC: {auc_e:.4f}")
    print(f"ETH Daily Top1% 准确率: {acc_e:.4f} (最低月: {min_e:.4f}, 坏月: {bad_e}, 日均交易: {tpd_e:.1f}笔)")
    print(f"ETH 逐月明细: {monthly_e}\n")

    print(f"BTC GNN v3 Test AUC: {auc_b:.4f}")
    print(f"BTC Daily Top1% 准确率: {acc_b:.4f} (最低月: {min_b:.4f}, 坏月: {bad_b}, 日均交易: {tpd_b:.1f}笔)")
    print(f"BTC 逐月明细: {monthly_b}\n")

if __name__ == "__main__":
    main()
