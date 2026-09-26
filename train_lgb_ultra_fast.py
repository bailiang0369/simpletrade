"""Fast LightGBM Only Trainer for Stoch Features (Subsampled/Binned for Speed).
"""

import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def train_eval_lgb_fast(dataset_path: str, horizon_str: str):
    print(f"\n=======================================================", flush=True)
    print(f"Loading dataset: {dataset_path} ({horizon_str})", flush=True)
    print(f"=======================================================", flush=True)

    # Read polars with stride or downsample if needed for ultra-speed
    df = pl.read_parquet(dataset_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Total rows: {len(df):,}, Features count: {len(feat_cols)}", flush=True)

    X = df.select(feat_cols).to_numpy()
    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # For fast training, stride train by 2
    X_tr_sub = X_tr[::2]
    y_tr_sub = y_tr[::2]

    print(f"Subsampled Train samples: {len(X_tr_sub):,}, Test samples: {len(X_te):,}", flush=True)

    # LightGBM Fast
    print("\n--- Training LightGBM Model ---", flush=True)
    lgb_params = {
        'n_estimators': 250,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    t0 = time.time()
    clf_lgb.fit(X_tr_sub, y_tr_sub)
    print(f"Fitting completed in {time.time()-t0:.1f}s", flush=True)

    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]
    auc_lgb = roc_auc_score(y_te, p_lgb)
    print(f"LGBM Test AUC: {auc_lgb:.4f}", flush=True)

    # Evaluate 二进制胜率 across Quantiles using strict causal evaluation
    print(f"\n--- Causal Evaluation (eval_r2_causal_daily) for {horizon_str} ---", flush=True)
    for q in [98.5, 99.0, 99.2, 99.4, 99.5]:
        acc_lgb, min_lgb, bad_lgb, tpd_lgb, _ = eval_r2_causal_daily(p_lgb, y_te, ts_te, p_quantile=q)
        print(f"[Quantile P{q:4.1f}% | Horizon: {horizon_str}] -> Win Rate: {acc_lgb*100:6.2f}% | Min Mo: {min_lgb*100:5.2f}% | Trades/Day: {tpd_lgb:5.2f}", flush=True)

    return p_lgb, y_te, ts_te

if __name__ == "__main__":
    train_eval_lgb_fast("data/datasets/ds_ETH_stoch_h15.parquet", "H15m")
    train_eval_lgb_fast("data/datasets/ds_ETH_stoch_h30.parquet", "H30m")
