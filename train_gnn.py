"""GNN (图神经网络) 加密资产趋势预测模型
极简精简特征集:
  1. Price Curve (对数收益率 lr / 滚动价格归一化)
  2. Stoch (60, 2, 1) (%K, %D 随机指标)
  3. CVD (Cumulative Volume Delta 主动买卖盘累计差比率)
  4. MACD Hist (MACD 柱状图: EMA12-EMA26 与 Signal9 差值)

图结构 (Cross-Asset Spatial-Temporal Graph):
  - 节点: Node 0 = ETH, Node 1 = BTC
  - 节点特征: 包含近 W 个 timestep 的 4 维极简特征
  - 边: 跨资产双向边 (ETH <-> BTC) + 自环 (Self-loop)
  - 图注意力网络 (Graph Attention Network, GAT): 学习 ETH 和 BTC 之间的动态贝塔传导与注意力权重
"""
import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 特征提取 ==========
def build_minimal_features(raw_df):
    close = raw_df['close'].values.astype(np.float64)
    buy = raw_df['buy_vol'].values.astype(np.float64)
    sell = raw_df['sell_vol'].values.astype(np.float64)
    n = len(close)

    # 1. Price Curve (1min 对数收益率)
    lr = np.zeros(n, dtype=np.float32)
    lr[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12)).astype(np.float32)

    # 2. Stoch (60, 2, 1)
    s_close = pd.Series(close)
    l60 = s_close.rolling(60).min()
    h60 = s_close.rolling(60).max()
    fast_k = (s_close - l60) / (h60 - l60 + 1e-9) * 100
    slow_k = fast_k.rolling(2).mean().fillna(50.0).values.astype(np.float32) / 100.0  # 归一化到 [0, 1]

    # 3. CVD (30-bar 滚动 Cumulative Volume Delta 比率)
    d = pd.Series(buy - sell)
    tot = pd.Series(buy + sell)
    cvd = (d.rolling(30).sum() / (tot.rolling(30).sum() + 1e-9)).fillna(0.0).values.astype(np.float32)

    # 4. MACD Hist
    ema12 = s_close.ewm(span=12, adjust=False).mean()
    ema26 = s_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line).fillna(0.0).values.astype(np.float32)
    # Z-score 归一化 MACD Hist
    macd_std = np.std(macd_hist) + 1e-9
    macd_hist = (macd_hist / macd_std).astype(np.float32)

    # 组合为 (n, 4) 矩阵
    feats = np.stack([lr, slow_k, cvd, macd_hist], axis=1).astype(np.float32)
    return feats

def load_and_prepare_data(horizon=30, lookback=30):
    print("[GNN] 提取 ETH 与 BTC 极简特征 (Price, Stoch, CVD, MACD Hist)...", flush=True)
    raw_eth = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_btc = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    # 对齐时间戳
    ts_eth = raw_eth['ts'].values.astype(np.int64)
    ts_btc = raw_btc['ts'].values.astype(np.int64)
    common_ts = np.intersect1d(ts_eth, ts_btc)

    idx_eth = np.searchsorted(ts_eth, common_ts)
    idx_btc = np.searchsorted(ts_btc, common_ts)

    feats_eth = build_minimal_features(raw_eth)[idx_eth]
    feats_btc = build_minimal_features(raw_btc)[idx_btc]

    close_eth = raw_eth['close'].values[idx_eth]
    close_btc = raw_btc['close'].values[idx_btc]
    ts = common_ts

    # 构建 30min 涨跌 Label (未来 horizon 根收盘大于当前)
    n = len(ts)
    ret_eth = np.full(n, np.nan, dtype=np.float32)
    ret_btc = np.full(n, np.nan, dtype=np.float32)
    ret_eth[:-horizon] = (close_eth[horizon:] / close_eth[:-horizon] - 1.0).astype(np.float32)
    ret_btc[:-horizon] = (close_btc[horizon:] / close_btc[:-horizon] - 1.0).astype(np.float32)

    label_eth = (ret_eth > 0).astype(np.int8)
    label_btc = (ret_btc > 0).astype(np.int8)

    # 有效行掩码
    valid = ~np.isnan(ret_eth) & ~np.isnan(ret_btc) & (np.arange(n) >= lookback)
    return ts, feats_eth, feats_btc, label_eth, label_btc, ret_eth, ret_btc, valid

# ========== 2. 图神经网络架构 (GAT Spatial-Temporal Model) ==========
class GraphAttentionLayer(nn.Module):
    """跨资产图注意力层 (GAT Layer)"""
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        # h shape: (batch_size, num_nodes, in_features)  num_nodes = 2 (0:ETH, 1:BTC)
        B, N, _ = h.shape
        Wh = self.W(h)  # (B, N, out_features)

        # 构造节点对拼接 [Wh_i || Wh_j]
        Wh1 = Wh.repeat_interleave(N, dim=1)  # (B, N*N, out_features)
        Wh2 = Wh.repeat(1, N, 1)              # (B, N*N, out_features)
        comb = torch.cat([Wh1, Wh2], dim=-1).view(B, N, N, 2 * self.out_features)

        e = self.leakyrelu(self.a(comb)).squeeze(-1)  # (B, N, N)

        # 邻接矩阵掩码注意力
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # (B, N, N)

        h_prime = torch.matmul(attention, Wh)     # (B, N, out_features)
        return F.elu(h_prime)

class SpatialTemporalGNN(nn.Module):
    """跨资产时空图神经网络"""
    def __init__(self, in_dim=4, lookback=30, hidden_dim=64, num_heads=2):
        super(SpatialTemporalGNN, self).__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim

        # 1. Temporal Feature Extractor (1D-CNN over lookback window for each node)
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 2. Spatial Graph Attention Layer (Inter-asset message passing between ETH and BTC)
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)

        # 3. Output Head for ETH and BTC prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj):
        # x shape: (B, N_nodes=2, lookback, in_dim=4)
        B, N, L, F_dim = x.shape

        # Reshape to process each node's time sequence through Conv1D
        x_flat = x.view(B * N, L, F_dim).transpose(1, 2)  # (B*N, F_dim, L)
        node_embed = self.time_conv(x_flat).squeeze(-1)   # (B*N, hidden_dim)
        node_embed = node_embed.view(B, N, self.hidden_dim) # (B, N, hidden_dim)

        # Graph Message Passing (ETH <-> BTC)
        g1 = self.gat1(node_embed, adj)
        g2 = self.gat2(g1, adj) + node_embed             # Residual connection

        # Predict probability for each node
        logits = self.fc(g2).squeeze(-1)                 # (B, N)
        probs = torch.sigmoid(logits)                    # (B, N)
        return probs

# ========== 3. Dataset与DataLoader ==========
class VectorizedGNNDataset(Dataset):
    def __init__(self, feats_eth, feats_btc, label_eth, label_btc, valid_indices, lookback=30):
        # 预先构建 3D 向量化张量 Tensor
        # feats shape: (N, 4) -> unfold -> (N - lookback + 1, 4, lookback) -> transpose -> (N - lookback + 1, lookback, 4)
        t_eth = torch.from_numpy(feats_eth).unfold(0, lookback, 1).transpose(1, 2)
        t_btc = torch.from_numpy(feats_btc).unfold(0, lookback, 1).transpose(1, 2)

        # 堆叠 ETH 与 BTC 为 (M, 2, lookback, 4)
        X_all = torch.stack([t_eth, t_btc], dim=1)  # shape (N - lookback + 1, 2, lookback, 4)

        # 提取 valid_indices 对应的行
        # 注意: index t 对应的 window 是 feats[t - lookback + 1 : t + 1], 其在 X_all 中的索引位置是 t - lookback
        adj_indices = valid_indices - lookback
        self.X = X_all[adj_indices].float()
        self.Y = torch.from_numpy(np.stack([label_eth[valid_indices], label_btc[valid_indices]], axis=1)).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ========== 4. 评估与训练逻辑 ==========
def evaluate_gnn(model, loader, adj_tensor):
    model.eval()
    preds_eth, preds_btc = [], []
    ys_eth, ys_btc = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
            probs = model(bx, adj_tensor)  # (B, 2)
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
    BATCH_SIZE = 8192
    EPOCHS = 5
    LR = 0.002

    ts, feats_eth, feats_btc, label_eth, label_btc, ret_eth, ret_btc, valid = load_and_prepare_data(horizon=30, lookback=LOOKBACK)

    # 严格时间切分
    def ts_mask(s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts >= a) & (ts < b)

    tr_m = ts_mask('2020-01-01', '2024-06-30') & valid
    es_m = ts_mask('2024-06-30', '2024-09-30') & valid
    mv_m = ts_mask('2024-09-30', '2025-09-30') & valid
    te_m = ts_mask('2025-09-30', '2026-08-29') & valid

    tr_idx = np.where(tr_m)[0][::5]  # 抽样 1/5 提速
    es_idx = np.where(es_m)[0]
    mv_idx = np.where(mv_m)[0]
    te_idx = np.where(te_m)[0]

    print(f"[GNN 数据集切分] Train={len(tr_idx):,}, EarlyStop={len(es_idx):,}, MetaVal={len(mv_idx):,}, Test={len(te_idx):,}", flush=True)

    # 创建 DataLoader
    print("[GNN] 向量化转换 Dataset Tensors...", flush=True)
    tr_dataset = VectorizedGNNDataset(feats_eth, feats_btc, label_eth, label_btc, tr_idx, lookback=LOOKBACK)
    es_dataset = VectorizedGNNDataset(feats_eth, feats_btc, label_eth, label_btc, es_idx, lookback=LOOKBACK)
    mv_dataset = VectorizedGNNDataset(feats_eth, feats_btc, label_eth, label_btc, mv_idx, lookback=LOOKBACK)
    te_dataset = VectorizedGNNDataset(feats_eth, feats_btc, label_eth, label_btc, te_idx, lookback=LOOKBACK)

    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    es_loader = DataLoader(es_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    mv_loader = DataLoader(mv_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 邻接矩阵 (2x2 全连接连通图 + 自环)
    adj = torch.ones((2, 2), dtype=torch.float32).to(DEVICE)

    model = SpatialTemporalGNN(in_dim=4, lookback=LOOKBACK, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCELoss()

    print(f"\n[GNN 训练开始] 设备: {DEVICE}, 参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    best_es_auc = 0.0
    best_model_path = os.path.join(config.MODEL_DIR, "best_gnn.pt")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
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

        train_loss = total_loss / len(tr_dataset)
        _, _, es_auc_eth, es_auc_btc = evaluate_gnn(model, es_loader, adj)
        mean_es_auc = (es_auc_eth + es_auc_btc) / 2.0

        if mean_es_auc > best_es_auc:
            best_es_auc = mean_es_auc
            torch.save(model.state_dict(), best_model_path)
            saved_str = " (Saved Best)"
        else:
            saved_str = ""

        print(f"  Epoch {ep:2d}/{EPOCHS} [{time.time()-t0:.0f}s] Loss: {train_loss:.4f} | EarlyStop AUC: ETH={es_auc_eth:.4f}, BTC={es_auc_btc:.4f}{saved_str}", flush=True)

    # ========== 5. 在 Test 上评估 ==========
    print("\n" + "=" * 65, flush=True)
    print("  GNN (图神经网络) 极简特征集 Test 段评估", flush=True)
    print("=" * 65, flush=True)

    model.load_state_dict(torch.load(best_model_path))
    p_eth_te, p_btc_te, te_auc_eth, te_auc_btc = evaluate_gnn(model, te_loader, adj)

    y_eth_te = label_eth[te_idx]
    y_btc_te = label_btc[te_idx]
    ts_te = ts[te_idx]

    acc_e, min_e, bad_e, tpd_e, monthly_e = eval_r2_daily(p_eth_te, y_eth_te, ts_te)
    acc_b, min_b, bad_b, tpd_b, monthly_b = eval_r2_daily(p_btc_te, y_btc_te, ts_te)

    print(f"ETH GNN Test AUC: {te_auc_eth:.4f}")
    print(f"ETH Daily Top1% 准确率: {acc_e:.4f} (最低月: {min_e:.4f}, 坏月: {bad_e}, 日均交易: {tpd_e:.1f}笔)")
    print(f"ETH 逐月明细: {monthly_e}\n")

    print(f"BTC GNN Test AUC: {te_auc_btc:.4f}")
    print(f"BTC Daily Top1% 准确率: {acc_b:.4f} (最低月: {min_b:.4f}, 坏月: {bad_b}, 日均交易: {tpd_b:.1f}笔)")
    print(f"BTC 逐月明细: {monthly_b}\n")

if __name__ == "__main__":
    main()
