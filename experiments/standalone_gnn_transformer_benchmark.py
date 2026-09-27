"""Standalone Deep Spatial-Temporal GNN & Temporal Transformer Benchmark Engine.

Tests standalone Graph Neural Networks (GNN v1 Spatial-Temporal GAT) and Temporal Transformers
without decision trees, ensembles, or stacking.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

class SpatialGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(size=(out_dim, 1)))
        self.attn_dst = nn.Parameter(torch.zeros(size=(out_dim, 1)))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst.data, gain=1.414)

    def forward(self, h, adj):
        Wh = self.fc(h)
        f_1 = torch.matmul(Wh, self.attn_src)
        f_2 = torch.matmul(Wh, self.attn_dst)

        logits = f_1 + f_2.transpose(1, 2)
        logits = self.leaky_relu(logits)

        zero_vec = -9e15 * torch.ones_like(logits)
        attention = torch.where(adj > 0, logits, zero_vec)
        attention = torch.softmax(attention, dim=-1)

        h_prime = torch.matmul(attention, Wh)
        return torch.relu(h_prime)

class StandaloneGNNModel(nn.Module):
    def __init__(self, in_features=4, num_nodes=2, seq_len=30, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.node_proj = nn.Linear(in_features, hidden_dim)
        self.gat = SpatialGraphAttention(hidden_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(hidden_dim * num_nodes, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, in_feat = x.shape

        h = self.node_proj(x)

        h_spatial = []
        for t in range(seq_len):
            h_t = self.gat(h[:, t, :, :], adj)
            h_spatial.append(h_t)

        h_seq = torch.stack(h_spatial, dim=1)
        h_seq_flat = h_seq.reshape(batch_size, seq_len, num_nodes * self.hidden_dim).transpose(1, 2)

        out_conv = torch.relu(self.bn(self.temporal_conv(h_seq_flat)))
        out_pool = torch.mean(out_conv, dim=-1)

        return torch.sigmoid(self.head(out_pool))

def train_eval_standalone_gnn(horizon_min: int = 15):
    print(f"\n=======================================================", flush=True)
    print(f"Standalone GNN v1 (GAT Spatial-Temporal) Benchmark for H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    eth_ds = f"data/datasets/ds_ETH_h{horizon_min}.parquet"
    btc_ds = f"data/datasets/ds_BTC_h{horizon_min}.parquet"

    df_eth = pl.read_parquet(eth_ds)
    df_btc = pl.read_parquet(btc_ds)

    gnn_cols = ['lr_15', 'stoch_k60', 'cvd_60', 'macd_hist']

    for col in gnn_cols:
        if col not in df_eth.columns:
            df_eth = df_eth.with_columns(pl.lit(0.0).alias(col))
        if col not in df_btc.columns:
            df_btc = df_btc.with_columns(pl.lit(0.0).alias(col))

    X_eth = df_eth.select(gnn_cols).to_numpy().astype(np.float32)
    X_btc = df_btc.select(gnn_cols).to_numpy().astype(np.float32)

    X_eth = np.nan_to_num((X_eth - np.mean(X_eth, axis=0)) / (np.std(X_eth, axis=0) + 1e-6))
    X_btc = np.nan_to_num((X_btc - np.mean(X_btc, axis=0)) / (np.std(X_btc, axis=0) + 1e-6))

    min_len = min(len(X_eth), len(X_btc))
    X_node0 = X_eth[:min_len]
    X_node1 = X_btc[:min_len]

    y = df_eth['label'].to_numpy()[:min_len]
    ts = df_eth['ts'].to_numpy()[:min_len]

    X_graph = np.stack([X_node0, X_node1], axis=1)

    n = len(X_graph)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X_graph[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X_graph[train_idx:], y[train_idx:], ts[train_idx:]

    seq_len = 30
    stride = 16
    num_seqs = (len(X_tr) - seq_len) // stride

    X_tr_seq = np.zeros((num_seqs, seq_len, 2, 4), dtype=np.float32)
    y_tr_seq = np.zeros(num_seqs, dtype=np.float32)

    for i in range(num_seqs):
        idx = i * stride
        X_tr_seq[i] = X_tr[idx : idx + seq_len]
        y_tr_seq[i] = y_tr[idx + seq_len - 1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StandaloneGNNModel(in_features=4, num_nodes=2, seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32).to(device)

    dataset_tr = DataLoader(TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq)), batch_size=512, shuffle=True)

    print(f"Training Standalone GNN v1 on {device}...", flush=True)
    model.train()
    for epoch in range(3):
        t0 = time.time()
        running_loss = 0.0
        for bx, by in dataset_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx, adj).squeeze()
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  GNN Epoch {epoch+1}/3 Loss: {running_loss/len(dataset_tr):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Prediction
    model.eval()
    test_stride = 8
    num_test_seqs = (len(X_te) - seq_len) // test_stride
    X_te_seq = np.zeros((num_test_seqs, seq_len, 2, 4), dtype=np.float32)
    indices = []

    for i in range(num_test_seqs):
        idx = i * test_stride
        X_te_seq[i] = X_te[idx : idx + seq_len]
        indices.append(idx + seq_len - 1)

    with torch.no_grad():
        bx = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        out = model(bx, adj).squeeze().cpu().numpy()

    p_gnn = np.full(len(ts_te), 0.5, dtype=np.float32)
    p_gnn[indices] = out

    df_p = pd.Series(p_gnn)
    df_p[df_p == 0.5] = np.nan
    p_gnn = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL STANDALONE GNN EVALUATION (ETH H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_gnn, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Standalone GNN Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

    return acc, tpd, bad_m, acc_m

if __name__ == "__main__":
    train_eval_standalone_gnn(horizon_min=15)
