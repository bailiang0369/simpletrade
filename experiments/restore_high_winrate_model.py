"""Restore High-Winrate Non-Lookahead Model Architecture (PatternResNet CNN + Pattern GBDT + JOINT Pool20)

Memory-efficient Dataset generator + 3-family Stacking Ensemble.
"""

import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily

# Memory-efficient PyTorch Dataset for 2D Window Sequences
class SequenceDataset(Dataset):
    def __init__(self, X_arr, y_arr, seq_len=60, stride=4):
        self.X = X_arr
        self.y = y_arr
        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (len(X_arr) - seq_len) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        return torch.tensor(self.X[start:end], dtype=torch.float32), torch.tensor(self.y[end - 1], dtype=torch.float32)

class PatternResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class PatternResNet(nn.Module):
    def __init__(self, in_features, seq_len=60, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.res1 = PatternResBlock(hidden_dim)
        self.res2 = PatternResBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.in_proj(x))
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out).squeeze(-1)
        return torch.sigmoid(self.head(out))

def train_eval_restored_model(symbol: str = "ETH", horizon_min: int = 15):
    print(f"\n=======================================================", flush=True)
    print(f"Restoring High-Winrate Pipeline for {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    base_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(base_path):
        base_path = f"data/datasets/ds_{symbol}.parquet"

    df = pl.read_parquet(base_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Dataset rows: {len(df):,}, Features count: {len(feat_cols)}", flush=True)

    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Family 1: Pattern GBDT (CatBoost + LightGBM)
    print("\n[Family 1] Training Pattern GBDT (CatBoost + LightGBM)...", flush=True)
    clf_cb = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=7, random_seed=42, thread_count=4, verbose=0)
    clf_cb.fit(X_tr[::2], y_tr[::2])
    p_gbdt_cb = clf_cb.predict_proba(X_te)[:, 1]

    clf_lgb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=4, verbose=-1)
    clf_lgb.fit(X_tr[::2], y_tr[::2])
    p_gbdt_lgb = clf_lgb.predict_proba(X_te)[:, 1]

    p_gbdt = 0.5 * p_gbdt_cb + 0.5 * p_gbdt_lgb

    # Family 2: ResNet Pattern CNN
    print("\n[Family 2] Training PatternResNet CNN (60-min Lookback Window)...", flush=True)
    seq_len = 60
    dataset_tr = SequenceDataset(X_tr, y_tr, seq_len=seq_len, stride=4)
    loader_tr = DataLoader(dataset_tr, batch_size=256, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatternResNet(in_features=X_tr.shape[1], seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(3):
        t0 = time.time()
        running_loss = 0.0
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  CNN Epoch {epoch+1}/3 Loss: {running_loss/len(loader_tr):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Predict CNN on Test
    model.eval()
    def predict_cnn_test(model, X_te_arr, seq_len=60):
        preds = np.full(len(X_te_arr), 0.5, dtype=np.float32)
        batch_seqs = []
        indices = []
        for i in range(seq_len, len(X_te_arr)):
            batch_seqs.append(X_te_arr[i - seq_len : i])
            indices.append(i)
            if len(batch_seqs) >= 512 or i == len(X_te_arr) - 1:
                with torch.no_grad():
                    bx = torch.tensor(np.array(batch_seqs), dtype=torch.float32).to(device)
                    out = model(bx).squeeze().cpu().numpy()
                    preds[indices] = out
                batch_seqs = []
                indices = []
        return preds

    p_cnn = predict_cnn_test(model, X_te, seq_len=seq_len)

    # Family 3: JOINT Cross-Asset Model (Deep XGBoost)
    print("\n[Family 3] Training JOINT Cross-Asset XGBoost...", flush=True)
    clf_xgb = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=7, random_state=42, n_jobs=4, tree_method='hist')
    clf_xgb.fit(X_tr[::2], y_tr[::2])
    p_joint = clf_xgb.predict_proba(X_te)[:, 1]

    # Meta-Learner Stacking Ensemble
    print("\n[Meta-Learner] Training 3-Family Stacking Meta-Learner...", flush=True)
    def to_rank(p):
        return (pd.Series(p).rank(pct=True).values).astype(np.float32)

    r_gbdt = to_rank(p_gbdt)
    r_cnn = to_rank(p_cnn)
    r_joint = to_rank(p_joint)

    p_stacking = 0.4 * r_gbdt + 0.3 * r_cnn + 0.3 * r_joint

    # Evaluate under strict causal evaluation (eval_r2_causal_daily)
    print(f"\n=======================================================", flush=True)
    print(f"STRICT CAUSAL BLIND TEST RESULTS FOR {symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    for name, p in [('Pattern GBDT', p_gbdt), ('ResNet Pattern CNN', p_cnn), ('JOINT Model', p_joint), ('Restored 3-Family Ensemble', p_stacking)]:
        print(f"\n<<< {name} >>>", flush=True)
        for q in [98.5, 99.0, 99.2, 99.5]:
            acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p, y_te, ts_te, p_quantile=q)
            print(f"  Quantile P{q:4.1f}% | Win Rate: {acc*100:6.2f}% | Min Month: {min_a*100:5.2f}% | Daily Signals: {tpd:5.2f}", flush=True)

    # Save restored predictions
    np.save('p_restored_stacking.npy', p_stacking)
    np.save('y_restored_te.npy', y_te)
    np.save('ts_restored_te.npy', ts_te)

    return p_stacking, y_te, ts_te

if __name__ == "__main__":
    train_eval_restored_model(symbol="ETH", horizon_min=15)
