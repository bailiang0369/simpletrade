"""高级时空图注意力网络 (Advanced Spatial-Temporal Graph Attention Network: ST-GAT)

针对 H=30 周期进行 PyTorch 模型实现与训练优化：
1. 多头图空间注意力 (Multi-Head Spatial Graph Attention)
2. 1D 扩张卷积 (Temporal Dilated Conv) + LayerNorm + 残差连接
3. Focal Loss (Focuses gradient on hard, high-confidence samples)
"""

import time
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_store import AssetContext
from evaluate import evaluate_topk
ADVANCED_GNN_FEATURES = [
    "lr_15", "lr_120", "lr_240",
    "rvol_30", "rvol_60",
    "z_30", "z_120",
    "pos_30", "pos_120",
    "cvd_30",
    "hour_sin", "hour_cos"
]


class FocalLoss(nn.Module):
    """Focal Loss: 降低简单样本权重，强化极不确定/极高置信度难样本学习"""
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SpatialGraphAttention(nn.Module):
    """多头图注意力 (Multi-Head Graph Spatial Attention)"""
    def __init__(self, in_features, hidden_dim, heads=2):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads

        self.q_fc = nn.Linear(in_features, hidden_dim)
        self.k_fc = nn.Linear(in_features, hidden_dim)
        self.v_fc = nn.Linear(in_features, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """x: (B, W, N, F)"""
        B, W, N, F_dim = x.shape
        # Reshape to (B*W, N, F)
        x_flat = x.reshape(B * W, N, F_dim)

        Q = self.q_fc(x_flat).reshape(B * W, N, self.heads, self.head_dim).permute(0, 2, 1, 3) # (B*W, heads, N, head_dim)
        K = self.k_fc(x_flat).reshape(B * W, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_fc(x_flat).reshape(B * W, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim) # (B*W, heads, N, N)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B * W, N, self.hidden_dim) # (B*W, N, hidden_dim)
        out = self.out_fc(out).reshape(B, W, N, self.hidden_dim)
        return out


class STGATModel(nn.Module):
    """ST-GAT: Spatial-Temporal Graph Attention Network"""
    def __init__(self, num_nodes=2, in_channels=12, seq_len=30, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 1. 空间图注意力 (Spatial Graph Attention)
        self.gat = SpatialGraphAttention(in_channels, hidden_dim, heads=2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(in_channels, hidden_dim)

        # 2. 时间 1D 扩张卷积 + 残差 (Temporal Dilated Conv)
        self.temp_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.temp_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 3. GRU 序列编码
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)

        # 4. 预测输出头 (Output Head)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """x: (Batch, N=2, W=30, F=12)"""
        B, N, W, F_dim = x.shape
        x_perm = x.permute(0, 2, 1, 3) # (B, W, N, F)

        # 空间注意力 + 残差连接
        h_gat = self.gat(x_perm) # (B, W, N, hidden_dim)
        h_proj = self.proj_in(x_perm)
        h_spatial = self.norm1(h_gat + h_proj)

        # 时间 1D 扩张卷积
        h_in = h_spatial.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, W)
        h_c1 = F.relu(self.temp_conv1(h_in))
        h_c2 = F.relu(self.temp_conv2(h_c1))
        h_temp = (h_c2 + h_in).permute(0, 2, 1) # (B*N, W, hidden_dim)

        # GRU
        _, h_n = self.gru(h_temp) # (2, B*N, hidden_dim)
        h_last = h_n[-1].reshape(B, N * self.hidden_dim)

        logits = self.head(h_last).squeeze(-1)
        return torch.sigmoid(logits)


class STGATDataset(Dataset):
    def __init__(self, X_target, X_other, labels, W=30):
        self.X_target = X_target
        self.X_other = X_other
        self.labels = labels
        self.W = W
        self.valid_indices = np.arange(W - 1, len(labels))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_i = self.valid_indices[idx]
        slice_idx = slice(real_i - self.W + 1, real_i + 1)
        t_win = self.X_target[slice_idx]
        o_win = self.X_other[slice_idx]
        x_graph = np.stack([t_win, o_win], axis=0).astype(np.float32)
        y = self.labels[real_i].astype(np.float32)
        return torch.from_numpy(x_graph), torch.tensor(y)


def train_and_eval_stgat(s_target="BTC", s_other="ETH", horizon=30, epochs=4, batch_size=1024):
    t0 = time.time()
    device = torch.device("cpu")

    ctx_target = AssetContext(s_target, horizon=horizon)
    ctx_other = AssetContext(s_other, horizon=horizon)

    common_ts, idx_t, idx_o = np.intersect1d(ctx_target.ds_ts, ctx_other.ds_ts, return_indices=True)

    X_target_all = ctx_target.X_subset(ADVANCED_GNN_FEATURES, np.ones(len(ctx_target.ds_ts), dtype=bool))[idx_t]
    X_other_all = ctx_other.X_subset(ADVANCED_GNN_FEATURES, np.ones(len(ctx_other.ds_ts), dtype=bool))[idx_o]
    labels = ctx_target.label[idx_t]

    tr_mask = ctx_target.split_rows["train"][idx_t]
    es_mask = ctx_target.split_rows["early_stop"][idx_t]
    te_mask = ctx_target.split_rows["test"][idx_t]

    mean_t = X_target_all[tr_mask].mean(axis=0, keepdims=True)
    std_t = X_target_all[tr_mask].std(axis=0, keepdims=True) + 1e-6
    X_target_norm = (X_target_all - mean_t) / std_t

    mean_o = X_other_all[tr_mask].mean(axis=0, keepdims=True)
    std_o = X_other_all[tr_mask].std(axis=0, keepdims=True) + 1e-6
    X_other_norm = (X_other_all - mean_o) / std_o

    tr_idx = np.where(tr_mask)[0][-300_000:]
    es_idx = np.where(es_mask)[0]

    ds_train = STGATDataset(X_target_norm[tr_idx], X_other_norm[tr_idx], labels[tr_idx])
    ds_es = STGATDataset(X_target_norm[es_idx], X_other_norm[es_idx], labels[es_idx])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_es = DataLoader(ds_es, batch_size=batch_size, shuffle=False)

    model = STGATModel(num_nodes=2, in_channels=len(ADVANCED_GNN_FEATURES), seq_len=30, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    print(f"[ST-GAT] 开始训练高级时空图注意力网络 {s_target} H={horizon} ({len(ds_train)} 样本)...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in loader_train:
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(ds_train)

        model.eval()
        es_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in loader_es:
                preds = model(x_b)
                loss = criterion(preds, y_b)
                es_loss += loss.item() * len(y_b)
        es_loss /= len(ds_es)
        print(f"  Epoch {epoch}/{epochs}: Train Focal Loss={train_loss:.4f}, EarlyStop Loss={es_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # Predict Test split
    te_idx = np.where(te_mask)[0]
    ds_te = STGATDataset(X_target_norm[te_idx], X_other_norm[te_idx], labels[te_idx])
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    model.eval()
    test_preds = []
    with torch.no_grad():
        for x_b, _ in loader_te:
            preds = model(x_b)
            test_preds.append(preds.numpy())

    p_valid = np.concatenate(test_preds)
    p_full = np.full(len(te_idx), 0.5, dtype=np.float32)
    p_full[29:] = p_valid

    # Evaluate Top 1.0%
    y_te = ctx_target.label[te_idx]
    times_te = ctx_target.times('test')[:len(te_idx)]
    retf_te = ctx_target.retf('test')[:len(te_idx)]

    r_te = evaluate_topk(p_full, y_te, retf_te, times_te, coverage=0.01)
    print(f"[ST-GAT] {s_target} H={horizon} 测算完毕: Top 1.0% 准确率 = {r_te['accuracy']*100:.2f}% (耗时 {time.time()-t0:.0f}s)", flush=True)
    return p_full, r_te


if __name__ == "__main__":
    train_and_eval_stgat("BTC", "ETH", horizon=30)
    train_and_eval_stgat("ETH", "BTC", horizon=30)
