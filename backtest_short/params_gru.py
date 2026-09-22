#!/usr/bin/env python3
"""GRU 事件合约方向模型 (H=15): 6近原始通道 × 120窗口。

对齐: 用 data_store 的 ds_ts (与 split_rows/label 对齐), raw 通道按 ts 映射。
省内存: 训练用降采样(每5min), mini-batch 迭代, 不一次性装全部窗口。
"""
import os, sys, gc
sys.path.insert(0, "/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
from data_store import AssetContext

H = 15
WINDOW = 120
CHANNELS = 6
SEED = 42
BATCH = 1024
EPOCHS = 6
LR = 2e-3
SAMPLE = 10   # 训练降采样: 每10min取1样本


# ---------------- 特征通道 (按 ds_ts 对齐到 raw) ----------------
def map_to_ds(raw_df, ds_ts):
    """raw 特征通过 ts 精确映射到 ds 行。返回 (T_ds, C) float32。"""
    raw_ts = raw_df["ts"].to_numpy(np.int64)
    # 建 ts->index 映射
    ts2row = {int(t): i for i, t in enumerate(raw_ts)}
    idx = np.array([ts2row[int(t)] for t in ds_ts], dtype=np.int64)
    c = raw_df["close"].to_numpy(np.float64)[idx]
    hi = raw_df["high"].to_numpy(np.float64)[idx]
    lo = raw_df["low"].to_numpy(np.float64)[idx]
    buy = raw_df["buy_vol"].to_numpy(np.float64)[idx]
    sell = raw_df["sell_vol"].to_numpy(np.float64)[idx]

    n = len(c)
    lr1 = np.full(n, np.nan); lr1[1:] = np.log(c[1:] / c[:-1])
    imb = np.nan_to_num((buy - sell) / (buy + sell + 1e-9))
    # rvol: rolling std 15
    cval = np.nan_to_num(lr1)
    m = np.convolve(cval, np.ones(15) / 15, mode="full")[:n]
    m2 = np.convolve(cval ** 2, np.ones(15) / 15, mode="full")[:n]
    rvol = np.sqrt(np.clip(m2 - m ** 2, 0, None))
    rvol[0] = np.nan; rvol = np.nan_to_num(rvol, nan=0.0)
    # Stochastic K: rolling 120 high/low
    # 用 pandas rolling
    s = pd.Series(c)
    rmin = s.rolling(120, min_periods=1).min().to_numpy()
    rmax = s.rolling(120, min_periods=1).max().to_numpy()
    stoch = (c - rmin) / (rmax - rmin + 1e-9)

    X = np.stack([lr1, imb, rvol, stoch], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X, c


class SeqDataset(Dataset):
    def __init__(self, X, y, idx_list, W=WINDOW):
        self.X = X; self.y = y; self.W = W
        self.idx = np.array(idx_list, dtype=np.int64)
        self.idx = self.idx[self.idx >= W]  # 需要前 W 个点

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j - self.W:j].copy()  # (W, C)
        return torch.from_numpy(x).float(), torch.tensor(self.y[j], dtype=torch.float32)


class GRUModel(nn.Module):
    def __init__(self, channels, hidden=80, layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.gru = nn.GRU(channels, hidden, layers, batch_first=True, dropout=0.1 if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sym = "ETH"
    ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
    ds_ts = ctx.ds_ts
    # 跨资产
    other = "BTC"
    raw_eth = pd.read_parquet(f"{config.DS_DIR}/raw_ETH.parquet")
    raw_btc = pd.read_parquet(f"{config.DS_DIR}/raw_BTC.parquet")

    X_eth, c_eth = map_to_ds(raw_eth, ds_ts)
    X_btc, _ = map_to_ds(raw_btc, ds_ts)
    # 拼 6 通道: ETH lr1/imb/rvol/stoch + BTC lr1/imb
    X = np.concatenate([X_eth[:, [0, 1, 2, 3]], X_btc[:, [0, 1]]], axis=1).astype(np.float32)
    del X_eth, X_btc; gc.collect()
    print(f"X: {X.shape} (C={X.shape[1]})", flush=True)

    y = ctx.label.astype(np.float32)  # 方向
    T = len(y)

    # 训练: 降采样每SAMPLE min
    tr_all = np.where(ctx.split_rows["train"])[0]
    tr_all = tr_all[::SAMPLE]
    es_all = np.where(ctx.split_rows["early_stop"])[0]
    es_all = es_all[::SAMPLE]
    val_all = np.where(ctx.split_rows["meta_val"])[0]
    test_all = np.where(ctx.split_rows["test"])[0]

    # 训练集再分: 用 early_stop 做早停
    train_ds = SeqDataset(X, y, tr_all)
    es_ds = SeqDataset(X, y, es_all)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    es_dl = DataLoader(es_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = GRUModel(CHANNELS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    model.train()

    print(f"train windows: {len(train_ds)}, es windows: {len(es_ds)}", flush=True)
    import time
    t0 = time.time()
    best_auc = 0; best_state = None
    for ep in range(EPOCHS):
        tot = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        # 早停验证 AUC
        model.eval(); preds = []; trues = []
        with torch.no_grad():
            for xb, yb in es_dl:
                out = torch.sigmoid(model(xb))
                preds.append(out.numpy()); trues.append(yb.numpy())
        model.train()
        p = np.concatenate(preds); t = np.concatenate(trues)
        # AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(t, p)
        print(f"ep{ep}: loss={tot/len(train_ds):.4f} es_auc={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # 保存 GRU P
    torch.save(best_state, f"{config.DS_DIR}/SHORT_{sym}_h{H}_gru_state.pt")
    # 保存预测: 返回 (idx_with_p, p)
    def predict_windows(idx_list):
        ds = SeqDataset(X, y, idx_list)
        p = np.zeros(len(idx_list), dtype=np.float32)
        dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        out_i = []
        with torch.no_grad():
            for xb, _ in dl:
                out_i.append(torch.sigmoid(model(xb)).numpy())
        po = np.concatenate(out_i)
        valid = ds.idx  # absolute index (>=W)
        pos = {int(v): k for k, v in enumerate(idx_list)}
        for vi, absi in enumerate(valid):
            p[pos[absi]] = po[vi]
        return p

    print("predicting meta_val...", flush=True)
    np.save(f"{config.DS_DIR}/SHORT_{sym}_h{H}_gru_meta_val_P.npy", predict_windows(val_all).astype(np.float32))
    print("meta_val P saved", flush=True)
    print("predicting test...", flush=True)
    np.save(f"{config.DS_DIR}/SHORT_{sym}_h{H}_gru_test_P.npy", predict_windows(test_all).astype(np.float32))
    print("test P saved | best_auc=", round(best_auc, 4), flush=True)


if __name__ == "__main__":
    main()