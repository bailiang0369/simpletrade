"""Raw K-Line Chart Geometry Tensor Neural Engine.

Transforms raw OHLC price series ending at minute t into scale-invariant relative chart percentage tensors:
P(tau) = (price[tau] - close[t]) / close[t] for tau in [t - seq_len + 1 .. t]
Channels: [Open_rel, High_rel, Low_rel, Close_rel, Buy_Vol_ratio, Sell_Vol_ratio]

This represents 100% pure scale-invariant chart pattern geometry (support/resistance levels, breakouts, double bottoms)
without indicator distortion or global scaling bias.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

class RawChartGeometryDataset(Dataset):
    """Generates 100% Scale-Invariant Relative K-Line Chart Geometry Tensors.
    For each window of length seq_len ending at bar t:
    Open_rel = (Open - Close_t) / Close_t
    High_rel = (High - Close_t) / Close_t
    Low_rel  = (Low  - Close_t) / Close_t
    Close_rel= (Close - Close_t) / Close_t
    """
    def __init__(self, raw_path, ds_path, seq_len=60, stride=8, is_train=True):
        df_raw = pl.read_parquet(raw_path).sort("ts")
        df_ds = pl.read_parquet(ds_path).sort("ts")

        df = df_raw.join(df_ds, on="ts", suffix="_ds")

        # Extract Raw OHLC
        self.open = df["open"].to_numpy().astype(np.float32)
        self.high = df["high"].to_numpy().astype(np.float32)
        self.low = df["low"].to_numpy().astype(np.float32)
        self.close = df["close"].to_numpy().astype(np.float32)

        # Volumes
        self.buy_vol = df["buy_vol"].to_numpy().astype(np.float32)
        self.sell_vol = df["sell_vol"].to_numpy().astype(np.float32)

        self.y = df["label"].to_numpy().astype(np.float32)
        self.ts = df["ts"].to_numpy().astype(np.int64)

        n = len(df)
        train_idx = int(n * 0.8)

        if is_train:
            self.start_idx = 0
            self.end_idx = train_idx
        else:
            self.start_idx = train_idx
            self.end_idx = n

        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (self.end_idx - self.start_idx - seq_len) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor_idx = self.start_idx + idx * self.stride + self.seq_len - 1
        start = anchor_idx - self.seq_len + 1
        end = anchor_idx + 1

        c_t = self.close[anchor_idx] + 1e-8

        # Relative Scale-Invariant Geometry (Price Channel Percentage relative to Close_t)
        open_rel = (self.open[start:end] - c_t) / c_t
        high_rel = (self.high[start:end] - c_t) / c_t
        low_rel  = (self.low[start:end] - c_t) / c_t
        close_rel= (self.close[start:end] - c_t) / c_t

        # Relative Volume Imbalance
        vol_sum = self.buy_vol[start:end] + self.sell_vol[start:end] + 1e-8
        buy_vol_rel = self.buy_vol[start:end] / vol_sum
        sell_vol_rel = self.sell_vol[start:end] / vol_sum

        # Shape: (seq_len, 6)
        tensor_x = np.column_stack([
            open_rel, high_rel, low_rel, close_rel, buy_vol_rel, sell_vol_rel
        ]).astype(np.float32)

        return torch.tensor(tensor_x, dtype=torch.float32), torch.tensor(self.y[anchor_idx], dtype=torch.float32)

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class RawChartResNet(nn.Module):
    def __init__(self, in_channels=6, seq_len=60, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.res1 = ResBlock1D(hidden_dim)
        self.res2 = ResBlock1D(hidden_dim)
        self.res3 = ResBlock1D(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -> transpose to (batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        h = torch.relu(self.in_proj(x))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.pool(h).squeeze(-1)
        return torch.sigmoid(self.head(h))

def train_eval_raw_chart_geometry(symbol: str = "ETH", horizon_min: int = 15, seq_len: int = 60):
    print(f"\n=======================================================", flush=True)
    print(f"Raw K-Line Chart Geometry Neural Engine for {symbol} H={horizon_min}m (Window={seq_len}m)", flush=True)
    print(f"=======================================================", flush=True)

    raw_path = f"data/datasets/raw_{symbol}.parquet"
    ds_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"

    ds_tr = RawChartGeometryDataset(raw_path, ds_path, seq_len=seq_len, stride=16, is_train=True)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RawChartResNet(in_channels=6, seq_len=seq_len, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"Training Raw Chart ResNet on device: {device}...", flush=True)
    model.train()
    for epoch in range(4):
        t0 = time.time()
        running_loss = 0.0
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/4 | Train BCE Loss: {running_loss/len(loader_tr):.5f} ({time.time()-t0:.1f}s)", flush=True)

    # Test Set Prediction
    print("Evaluating Test Set Predictions on Raw Chart Geometry Tensors...", flush=True)
    test_stride = 8
    ds_te = RawChartGeometryDataset(raw_path, ds_path, seq_len=seq_len, stride=test_stride, is_train=False)
    loader_te = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=2)

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for bx, by in loader_te:
            bx = bx.to(device)
            out = model(bx).squeeze(-1).cpu().numpy()
            test_preds.extend(out)
            test_targets.extend(by.numpy())

    preds_sub = np.array(test_preds)
    y_te_sub = np.array(test_targets)

    # Match test timestamps
    df_ds = pl.read_parquet(ds_path).sort("ts")
    n = len(df_ds)
    train_idx = int(n * 0.8)
    ts_te = df_ds["ts"].to_numpy()[train_idx:]
    y_te = df_ds["label"].to_numpy()[train_idx:]

    p_neural = np.full(len(ts_te), 0.5, dtype=np.float32)
    test_indices = np.arange(seq_len - 1, len(ts_te) - 1, test_stride)[:len(preds_sub)]
    p_neural[test_indices] = preds_sub

    df_p = pd.Series(p_neural)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL RAW CHART GEOMETRY EVALUATION ({symbol} H={horizon_min}m)", flush=True)
    print(f"=======================================================", flush=True)

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        print(f"Quantile P{q:4.1f}% | Daily Signals: {tpd:5.2f} | Raw Chart Geometry Neural Win Rate: {acc*100:6.2f}% | Worst Month: {min_a*100:5.2f}%", flush=True)

    return acc, tpd, bad_m, acc_m

if __name__ == "__main__":
    train_eval_raw_chart_geometry(symbol="ETH", horizon_min=15, seq_len=60)
