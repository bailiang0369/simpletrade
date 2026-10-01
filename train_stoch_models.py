"""Train Targeted High-Confidence Models (LGBM, CatBoost, XGBoost & Stacking) for Event Options.

Optimized specifically for ETH 15m and 30m binary direction prediction using Stoch Oscillators & Price Action.
"""

import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def train_eval_stoch_models(dataset_path: str, horizon_str: str):
    print(f"\n=======================================================")
    print(f"Loading dataset: {dataset_path} ({horizon_str})")
    print(f"=======================================================")

    df = pl.read_parquet(dataset_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Total rows: {len(df):,}, Features count: {len(feat_cols)}")

    X = df.select(feat_cols).to_numpy()
    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    # Chronological 80/20 train/test split
    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    print(f"Train samples: {len(X_tr):,}, Test samples: {len(X_te):,}")

    # 1. LightGBM Deep Model tuned for Stoch & Price Action Reversals
    print("\n--- Training LightGBM Model ---")
    lgb_params = {
        'n_estimators': 600,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(X_tr, y_tr)
    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]
    auc_lgb = roc_auc_score(y_te, p_lgb)
    print(f"LGBM Test AUC: {auc_lgb:.4f}")

    # 2. CatBoost Deep Model
    print("\n--- Training CatBoost Model ---")
    cb_params = {
        'iterations': 600,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': 0
    }
    clf_cb = CatBoostClassifier(**cb_params)
    clf_cb.fit(X_tr, y_tr)
    p_cb = clf_cb.predict_proba(X_te)[:, 1]
    auc_cb = roc_auc_score(y_te, p_cb)
    print(f"CatBoost Test AUC: {auc_cb:.4f}")

    # 3. XGBoost Model
    print("\n--- Training XGBoost Model ---")
    xgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.02,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(X_tr, y_tr)
    p_xgb = clf_xgb.predict_proba(X_te)[:, 1]
    auc_xgb = roc_auc_score(y_te, p_xgb)
    print(f"XGBoost Test AUC: {auc_xgb:.4f}")

    # Ensemble Weighted Prediction
    p_ens = 0.4 * p_lgb + 0.3 * p_cb + 0.3 * p_xgb
    auc_ens = roc_auc_score(y_te, p_ens)
    print(f"\n>>> Ensemble Test AUC: {auc_ens:.4f} <<<")

    # Evaluate 二进制胜率 across Quantiles (P99, P99.2, P99.5) using strict causal evaluation
    print(f"\n--- Causal Evaluation (eval_r2_causal_daily) for {horizon_str} ---")
    for q in [99.0, 99.2, 99.4, 99.5]:
        acc_lgb, min_lgb, bad_lgb, tpd_lgb, _ = eval_r2_causal_daily(p_lgb, y_te, ts_te, p_quantile=q)
        acc_cb, min_cb, bad_cb, tpd_cb, _ = eval_r2_causal_daily(p_cb, y_te, ts_te, p_quantile=q)
        acc_xgb, min_xgb, bad_xgb, tpd_xgb, _ = eval_r2_causal_daily(p_xgb, y_te, ts_te, p_quantile=q)
        acc_ens, min_ens, bad_ens, tpd_ens, _ = eval_r2_causal_daily(p_ens, y_te, ts_te, p_quantile=q)

        print(f"\n[Quantile P{q:4.1f}% | Horizon: {horizon_str}]")
        print(f"  LGBM     -> Win Rate: {acc_lgb*100:6.2f}% | Min Mo: {min_lgb*100:5.2f}% | Trades/Day: {tpd_lgb:5.2f}")
        print(f"  CatBoost -> Win Rate: {acc_cb*100:6.2f}% | Min Mo: {min_cb*100:5.2f}% | Trades/Day: {tpd_cb:5.2f}")
        print(f"  XGBoost  -> Win Rate: {acc_xgb*100:6.2f}% | Min Mo: {min_xgb*100:5.2f}% | Trades/Day: {tpd_xgb:5.2f}")
        print(f"  ENSEMBLE -> Win Rate: {acc_ens*100:6.2f}% | Min Mo: {min_ens*100:5.2f}% | Trades/Day: {tpd_ens:5.2f}")

    return p_ens, y_te, ts_te

if __name__ == "__main__":
    p15, y15, ts15 = train_eval_stoch_models("data/datasets/ds_ETH_stoch_h15.parquet", "H15m")
    p30, y30, ts30 = train_eval_stoch_models("data/datasets/ds_ETH_stoch_h30.parquet", "H30m")
