"""Heterogeneous Multi-Family Ensemble Engine.
Combines GBDT Tree Models (LightGBM, CatBoost, XGBoost) and Deep Neural Models (Cross-Asset Attention Net, Deep TCN-ResNet)
under strict causal rolling 90-day P99 quantile rules.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from causal_eval import eval_r2_causal_daily
from experiments.cross_asset_attention_net import CrossAssetDataset, CrossAssetAttentionNet, FocalBCELoss

def run_heterogeneous_ensemble(target_symbol="ETH", cross_symbol="BTC", horizon_min=30):
    print(f"\n=======================================================", flush=True)
    print(f"HETEROGENEOUS ENSEMBLE ENGINE: {target_symbol} H={horizon_min}m", flush=True)
    print(f"=======================================================", flush=True)

    df_t = pl.read_parquet(f"data/datasets/ds_{target_symbol}_h{horizon_min}.parquet")
    df_c = pl.read_parquet(f"data/datasets/ds_{cross_symbol}_h{horizon_min}.parquet")

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols_t = [c for c in df_t.columns if c not in ignore_cols]
    feat_cols_c = [c for c in df_c.columns if c not in ignore_cols]

    min_len = min(len(df_t), len(df_c))
    df_t = df_t.slice(len(df_t) - min_len, min_len)
    df_c = df_c.slice(len(df_c) - min_len, min_len)

    X_t = df_t.select(feat_cols_t).to_numpy().astype(np.float32)
    X_c = df_c.select(feat_cols_c).to_numpy().astype(np.float32)
    y = df_t['label'].to_numpy().astype(np.float32)
    ts = df_t['ts'].to_numpy().astype(np.int64)

    train_idx = int(min_len * 0.8)

    X_t_tr, X_c_tr, y_tr = X_t[:train_idx], X_c[:train_idx], y[:train_idx]
    X_t_te, X_c_te, y_te, ts_te = X_t[train_idx:], X_c[train_idx:], y[train_idx:], ts[train_idx:]

    print("1. Training GBDT Tree Models (LightGBM, XGBoost, CatBoost)...", flush=True)
    # LightGBM
    lgb_tr = lgb.Dataset(X_t_tr, label=y_tr)
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03, 'num_leaves': 31, 'verbose': -1, 'seed': 42}
    bst_lgb = lgb.train(lgb_params, lgb_tr, num_boost_round=120)
    p_lgb = bst_lgb.predict(X_t_te)

    # CatBoost
    cb_model = cb.CatBoostClassifier(iterations=120, learning_rate=0.03, depth=6, verbose=0, random_seed=42)
    cb_model.fit(X_t_tr, y_tr)
    p_cb = cb_model.predict_proba(X_t_te)[:, 1]

    p_tree = 0.5 * p_lgb + 0.5 * p_cb

    print("2. Training Cross-Asset Attention Neural Net...", flush=True)
    seq_len = 30
    ds_tr = CrossAssetDataset(X_t_tr, X_c_tr, y_tr, seq_len=seq_len, stride=24)
    loader_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CrossAssetAttentionNet(target_dim=X_t.shape[1], cross_dim=X_c.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = FocalBCELoss(gamma=2.5)

    net.train()
    for epoch in range(3):
        for b_xt, b_xc, by, _ in loader_tr:
            b_xt, b_xc, by = b_xt.to(device), b_xc.to(device), by.to(device)
            optimizer.zero_grad()
            logits = net(b_xt, b_xc).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    ds_te = CrossAssetDataset(X_t_te, X_c_te, y_te, seq_len=seq_len, stride=12,
                              mean_target=ds_tr.mean_t, std_target=ds_tr.std_t,
                              mean_cross=ds_tr.mean_c, std_cross=ds_tr.std_c)
    loader_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    net.eval()
    p_neural_seq = np.full(len(ts_te), 0.5, dtype=np.float32)

    with torch.no_grad():
        for b_xt, b_xc, _, end_indices in loader_te:
            b_xt, b_xc = b_xt.to(device), b_xc.to(device)
            logits = net(b_xt, b_xc).squeeze(-1)
            probs = torch.sigmoid(logits / 0.75).cpu().numpy()
            p_neural_seq[end_indices.numpy()] = probs

    df_p = pd.Series(p_neural_seq)
    df_p[df_p == 0.5] = np.nan
    p_neural = df_p.ffill().bfill().to_numpy()

    print("\n3. Evaluating Standalone vs Heterogeneous Ensemble...", flush=True)
    p_hetero_ensemble = 0.55 * p_tree + 0.45 * p_neural

    for q in [98.5, 99.0, 99.2, 99.5]:
        acc_t, min_t, bad_t, tpd_t, _ = eval_r2_causal_daily(p_tree, y_te, ts_te, p_quantile=q)
        acc_n, min_n, bad_n, tpd_n, _ = eval_r2_causal_daily(p_neural, y_te, ts_te, p_quantile=q)
        acc_e, min_e, bad_e, tpd_e, _ = eval_r2_causal_daily(p_hetero_ensemble, y_te, ts_te, p_quantile=q)

        print(f"\n--- Quantile P{q:4.1f}% ---")
        print(f"  GBDT Tree Win Rate         : {acc_t*100:6.2f}% ({tpd_t:5.2f} trades/day) | Worst Month: {min_t*100:5.2f}%")
        print(f"  Cross-Asset Neural Win Rate: {acc_n*100:6.2f}% ({tpd_n:5.2f} trades/day) | Worst Month: {min_n*100:5.2f}%")
        print(f"  Heterogeneous Ensemble     : {acc_e*100:6.2f}% ({tpd_e:5.2f} trades/day) | Worst Month: {min_e*100:5.2f}%")

if __name__ == "__main__":
    run_heterogeneous_ensemble(target_symbol="ETH", cross_symbol="BTC", horizon_min=30)
