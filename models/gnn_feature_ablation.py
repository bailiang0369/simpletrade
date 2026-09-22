"""系统化 GNN 特征消融实验脚本 (Systematic GNN Feature Ablation Script)

独立运行并评估 5 组不同特征子集的 GNN 表现，完全不修改原有逻辑与文件。
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
from models.gnn_feature_groups import FEATURE_GROUPS


class AblationSTGNN(nn.Module):
    """可变特征维度的 GNN 模型"""
    def __init__(self, num_nodes=2, in_channels=10, seq_len=30, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)
        self.spatial_fc = nn.Linear(in_channels, hidden_dim)
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, N, W, F_dim = x.shape
        A_norm = torch.softmax(self.A, dim=-1)
        x_perm = x.permute(0, 2, 1, 3)
        x_graph = torch.matmul(A_norm, x_perm)
        h_spatial = F.relu(self.spatial_fc(x_graph))
        h_temp_in = h_spatial.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, W)
        h_temp_out = F.relu(self.temporal_conv(h_temp_in))
        h_gru_in = h_temp_out.permute(0, 2, 1)
        _, h_n = self.gru(h_gru_in)
        h_flat = h_n.squeeze(0).reshape(B, N * self.hidden_dim)
        logits = self.head(h_flat).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class AblationSequenceDataset(Dataset):
    def __init__(self, X_btc, X_eth, labels, W=30):
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
        btc_win = self.X_btc[slice_idx]
        eth_win = self.X_eth[slice_idx]
        x_graph = np.stack([btc_win, eth_win], axis=0).astype(np.float32)
        y = self.labels[real_i].astype(np.float32)
        return torch.from_numpy(x_graph), torch.tensor(y)


def run_gnn_ablation_for_group(group_name, feature_list, ctx_target, ctx_other, horizon=30, epochs=3, batch_size=1024):
    t0 = time.time()
    device = torch.device("cpu")

    common_ts, idx_t, idx_o = np.intersect1d(ctx_target.ds_ts, ctx_other.ds_ts, return_indices=True)

    X_target_all = ctx_target.X_subset(feature_list, np.ones(len(ctx_target.ds_ts), dtype=bool))[idx_t]
    X_other_all = ctx_other.X_subset(feature_list, np.ones(len(ctx_other.ds_ts), dtype=bool))[idx_o]

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

    tr_idx = np.where(tr_mask)[0]
    if len(tr_idx) > 300_000:
        tr_idx = tr_idx[-300_000:]

    es_idx = np.where(es_mask)[0]

    ds_train = AblationSequenceDataset(X_target_norm[tr_idx], X_other_norm[tr_idx], labels[tr_idx])
    ds_es = AblationSequenceDataset(X_target_norm[es_idx], X_other_norm[es_idx], labels[es_idx])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_es = DataLoader(ds_es, batch_size=batch_size, shuffle=False)

    model = AblationSTGNN(num_nodes=2, in_channels=len(feature_list), seq_len=30, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCELoss()

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

    # Predict Test split
    te_idx = np.where(te_mask)[0]
    ds_te = AblationSequenceDataset(X_target_norm[te_idx], X_other_norm[te_idx], labels[te_idx])
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

    # Evaluate raw accuracy on test split
    r_raw = evaluate_topk(p_te_full, ctx_target.label[te_idx], ctx_target.retf('test')[:len(te_idx)], ctx_target.times('test')[:len(te_idx)], coverage=0.01)

    # Evaluate inverted accuracy on test split
    p_inv_full = 1.0 - p_te_full
    r_inv = evaluate_topk(p_inv_full, ctx_target.label[te_idx], ctx_target.retf('test')[:len(te_idx)], ctx_target.times('test')[:len(te_idx)], coverage=0.01)

    elapsed = time.time() - t0
    return {
        "group": group_name,
        "symbol": ctx_target.symbol,
        "n_feat": len(feature_list),
        "raw_acc": r_raw["accuracy"],
        "inv_acc": r_inv["accuracy"],
        "trades": r_raw["k"],
        "elapsed": elapsed
    }


def main():
    print("=================================================================")
    print("  GNN 特征子集系统化消融与评测 (Comprehensive Feature Ablation)")
    print("=================================================================\n")

    ctx_btc = AssetContext("BTC", horizon=30)
    ctx_eth = AssetContext("ETH", horizon=30)

    results = []
    for grp_name, feat_list in FEATURE_GROUPS.items():
        print(f"--- 评估 {grp_name} ({len(feat_list)} 维特征) ---", flush=True)
        res_btc = run_gnn_ablation_for_group(grp_name, feat_list, ctx_btc, ctx_eth, horizon=30)
        res_eth = run_gnn_ablation_for_group(grp_name, feat_list, ctx_eth, ctx_btc, horizon=30)
        results.append((res_btc, res_eth))
        print(f"  BTC: 原始胜率={res_btc['raw_acc']*100:5.2f}% | 镜像反转胜率={res_btc['inv_acc']*100:5.2f}% ({res_btc['elapsed']:.0f}s)")
        print(f"  ETH: 原始胜率={res_eth['raw_acc']*100:5.2f}% | 镜像反转胜率={res_eth['inv_acc']*100:5.2f}% ({res_eth['elapsed']:.0f}s)\n")

    print("=================================================================")
    print("  消融实验汇总表 (Ablation Summary Table)")
    print("=================================================================")
    print(f"{'Group Name':<22} {'Feat#':<6} {'BTC Raw':<10} {'BTC Inverted':<12} {'ETH Raw':<10} {'ETH Inverted':<12}")
    print("-" * 75)
    for res_btc, res_eth in results:
        print(f"{res_btc['group']:<22} {res_btc['n_feat']:<6} {res_btc['raw_acc']*100:>6.2f}%   {res_btc['inv_acc']*100:>8.2f}%     {res_eth['raw_acc']*100:>6.2f}%   {res_eth['inv_acc']*100:>8.2f}%")


if __name__ == "__main__":
    main()
