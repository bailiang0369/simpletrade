"""EXP-05: Focal Loss & Extreme Sample Reweighting Trainer.

Custom objective specifically penalizing false positives in the top 1% probability tail.
"""

import os, sys, time, warnings
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

# Custom Focal Loss for LightGBM/XGBoost
# Focal Loss: FL(p_t) = - (1 - p_t)^gamma * log(p_t)
def focal_loss_lgb(y_pred, dataset, gamma=2.0):
    y_true = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-15, 1 - 1e-15)

    p_t = np.where(y_true == 1, p, 1 - p)
    # Focal weight
    weight = (1 - p_t) ** gamma

    # First derivative (grad) and second derivative (hess)
    grad = np.where(y_true == 1, -(1 - p)**gamma * (1 - p + gamma * p * np.log(p)), p**gamma * (p + gamma * (1 - p) * np.log(1 - p)))
    hess = np.maximum(weight * p * (1 - p), 1e-6)
    return grad, hess

def run_exp05(horizon_str: str):
    base_path = f"data/datasets/ds_ETH_{horizon_str.lower()}.parquet"
    print(f"\n=======================================================", flush=True)
    print(f"Loading Base Dataset for EXP-05 Focal Loss ({base_path})", flush=True)
    print(f"=======================================================", flush=True)

    df = pl.read_parquet(base_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Sample reweighting: assign 3x weight to extreme volatility/moving bars
    ret_future_tr = df['ret_future'][:train_idx].to_numpy()
    high_vol_mask = np.abs(ret_future_tr) > np.percentile(np.abs(ret_future_tr), 90.0)
    sample_weights = np.where(high_vol_mask, 3.0, 1.0)

    print("\n--- Training LightGBM with Sample Reweighting on Extreme Volatility ---", flush=True)
    clf_lgb_weighted = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    clf_lgb_weighted.fit(X_tr[::2], y_tr[::2], sample_weight=sample_weights[::2])
    p_weighted = clf_lgb_weighted.predict_proba(X_te)[:, 1]

    print(f"\n--- Causal Evaluation for EXP-05 Sample-Weighted Model ({horizon_str}) ---", flush=True)
    for q in [98.5, 99.0, 99.2, 99.5]:
        acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p_weighted, y_te, ts_te, p_quantile=q)
        print(f"  Quantile P{q:4.1f}% | Win Rate: {acc*100:6.2f}% | Min Month: {min_a*100:5.2f}% | Trades/Day: {tpd:5.2f}", flush=True)

if __name__ == "__main__":
    run_exp05("H15")
