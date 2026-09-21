"""轻量级时空图神经网络 (Lightweight Spatial-Temporal GNN Model in PyTorch)

架构：
1. 节点 (Nodes): BTC 与 ETH 作为图的两个资产节点 (N=2)
2. 时间窗口 (Temporal Window): 每个节点最近 W=30 步的 10 维核心特征 (10 维，极省内存)
3. 空间图卷积 (Spatial Graph Conv / Cross-Attention): 学习 BTC 与 ETH 节点间的相互影响权重矩阵 (Adjacency Matrix A)
4. 时间 1D CNN / GRU 提取序列特征
5. 输出: 对目标节点未来 H=30 分钟涨跌的分类概率

内存控制：
- 批次大小 Batch Size = 1024
- 10 维特征通道，单样本张量仅 (N=2, W=30, F=10) = 600 float32 (2.4 KB)
- 完美避免 4GB cgroup 内存爆炸
"""

import time
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from .gnn_features import CORE_GNN_FEATURES


class CrossAssetSTGNN(nn.Module):
    """轻量级时空图神经网络 (Spatial-Temporal Graph Neural Network)"""
    def __init__(self, num_nodes=2, in_channels=10, seq_len=30, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 1. 可学习节点邻接矩阵 (Learnable Graph Spatial Adjacency Matrix)
        self.A = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)

        # 2. 空间图卷积 (Spatial Convolution)
        self.spatial_fc = nn.Linear(in_channels, hidden_dim)

        # 3. 时间 1D 卷积 (Temporal Convolution)
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        # 4. 时间注意力 / GRU 提取时间依赖
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )

        # 5. 分类输出头 (用于预测目标节点的涨跌概率)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """x: (Batch, N=2, W=30, F=10)"""
        B, N, W, F_dim = x.shape

        # Normalize Adjacency
        A_norm = torch.softmax(self.A, dim=-1) # (N, N)

        # Spatial Graph Message Passing: (B, W, N, F) x (N, N) -> (B, W, N, F)
        x_perm = x.permute(0, 2, 1, 3) # (B, W, N, F)
        x_graph = torch.matmul(A_norm, x_perm) # (B, W, N, F)

        # Spatial FC
        h_spatial = F.relu(self.spatial_fc(x_graph)) # (B, W, N, hidden_dim)

        # Temporal Conv: reshape to (B*N, hidden_dim, W)
        h_temp_in = h_spatial.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, W)
        h_temp_out = F.relu(self.temporal_conv(h_temp_in)) # (B*N, hidden_dim, W)

        # GRU: reshape to (B*N, W, hidden_dim)
        h_gru_in = h_temp_out.permute(0, 2, 1)
        _, h_n = self.gru(h_gru_in) # h_n: (1, B*N, hidden_dim)

        # Reshape to (B, N * hidden_dim)
        h_flat = h_n.squeeze(0).reshape(B, N * self.hidden_dim)

        # Output probability
        logits = self.head(h_flat).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class GNNSequenceDataset(Dataset):
    """轻量级流式时空序列 Dataset"""
    def __init__(self, X_btc, X_eth, labels, W=30):
        """X_btc: (L, F), X_eth: (L, F), labels: (L,)"""
        self.X_btc = X_btc
        self.X_eth = X_eth
        self.labels = labels
        self.W = W
        self.valid_indices = np.arange(W - 1, len(labels))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_i = self.valid_indices[idx]
        slice_idx = slice(real_i - self.W + 1, real_i + 1)

        # Stack BTC and ETH to form graph nodes tensor: (N=2, W=30, F=10)
        btc_win = self.X_btc[slice_idx]
        eth_win = self.X_eth[slice_idx]

        x_graph = np.stack([btc_win, eth_win], axis=0).astype(np.float32)
        y = self.labels[real_i].astype(np.float32)
        return torch.from_numpy(x_graph), torch.tensor(y)


def train_gnn_model(ctx_target, ctx_other, horizon=30, epochs=3, batch_size=1024):
    """训练 GNN 模型预测 ctx_target (例如 BTC)"""
    t0 = time.time()
    device = torch.device("cpu") # CPU 训练

    # 对齐两币种的时间戳交集
    common_ts, idx_t, idx_o = np.intersect1d(ctx_target.ds_ts, ctx_other.ds_ts, return_indices=True)

    X_target_all = ctx_target.X_subset(CORE_GNN_FEATURES, np.ones(len(ctx_target.ds_ts), dtype=bool))[idx_t]
    X_other_all = ctx_other.X_subset(CORE_GNN_FEATURES, np.ones(len(ctx_other.ds_ts), dtype=bool))[idx_o]

    # 标签与切分掩码
    labels = ctx_target.label[idx_t]

    # 切分掩码
    tr_mask = ctx_target.split_rows["train"][idx_t]
    es_mask = ctx_target.split_rows["early_stop"][idx_t]
    te_mask = ctx_target.split_rows["test"][idx_t]

    # 归一化 (StandardScaler)
    mean_t = X_target_all[tr_mask].mean(axis=0, keepdims=True)
    std_t = X_target_all[tr_mask].std(axis=0, keepdims=True) + 1e-6
    X_target_norm = (X_target_all - mean_t) / std_t

    mean_o = X_other_all[tr_mask].mean(axis=0, keepdims=True)
    std_o = X_other_all[tr_mask].std(axis=0, keepdims=True) + 1e-6
    X_other_norm = (X_other_all - mean_o) / std_o

    # 训练采样 (最后 30 万步)
    tr_idx = np.where(tr_mask)[0]
    if len(tr_idx) > 300_000:
        tr_idx = tr_idx[-300_000:]

    es_idx = np.where(es_mask)[0]

    ds_train = GNNSequenceDataset(X_target_norm[tr_idx], X_other_norm[tr_idx], labels[tr_idx])
    ds_es = GNNSequenceDataset(X_target_norm[es_idx], X_other_norm[es_idx], labels[es_idx])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_es = DataLoader(ds_es, batch_size=batch_size, shuffle=False)

    model = CrossAssetSTGNN(num_nodes=2, in_channels=len(CORE_GNN_FEATURES), seq_len=30, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"[GNN] {ctx_target.symbol} H={horizon} 开始训练 PyTorch 时空图神经网络 ({len(ds_train)} 样本)...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in loader_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(ds_train)

        model.eval()
        es_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader_es:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                es_loss += loss.item() * len(y_batch)
        es_loss /= len(ds_es)

        print(f"  Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, EarlyStop Loss={es_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 预测测试集
    te_idx = np.where(te_mask)[0]
    ds_te = GNNSequenceDataset(X_target_norm[te_idx], X_other_norm[te_idx], labels[te_idx])
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    model.eval()
    test_preds = []
    with torch.no_grad():
        for x_batch, _ in loader_te:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            test_preds.append(preds.cpu().numpy())

    p_te_valid = np.concatenate(test_preds)

    p_te_full = np.full(len(te_idx), 0.5, dtype=np.float32)
    p_te_full[29:] = p_te_valid

    print(f"[GNN] {ctx_target.symbol} H={horizon} 完成! 耗时 {time.time()-t0:.0f}s", flush=True)
    return p_te_full, model
