"""Train Specialized Confluence Classifier & Hybrid Rule Engine.

Evaluates Win Rate under strict eval_r2_causal_daily.
"""

import os, sys, time, warnings
import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def train_eval_confluence(horizon_str: str):
    conf_path = f"data/datasets/ds_ETH_confluence_{horizon_str.lower()}.parquet"
    base_path = f"data/datasets/ds_ETH_{horizon_str.lower()}.parquet"

    print(f"\n=======================================================", flush=True)
    print(f"Loading Confluence Dataset ({conf_path})", flush=True)
    print(f"=======================================================", flush=True)

    df_conf = pl.read_parquet(conf_path)
    df_base = pl.read_parquet(base_path)

    df_joint = df_base.join(df_conf, on="ts", suffix="_conf")

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'ret_day_conf', 'label_conf', 'soft_label_conf', 'ret_future_conf']
    feat_cols = [c for c in df_joint.columns if c not in ignore_cols]

    print(f"Joint rows: {len(df_joint):,}, Joint features count: {len(feat_cols)}", flush=True)

    X = df_joint.select(feat_cols).to_numpy().astype(np.float32)
    y = df_joint['label'].to_numpy()
    ts = df_joint['ts'].to_numpy()

    n = len(df_joint)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Train CatBoost Classifier with Focal / High Depth on Confluence Setups
    print("\n--- Training CatBoost High-Precision Classifier ---", flush=True)
    cb_params = {
        'iterations': 700,
        'learning_rate': 0.02,
        'depth': 8,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'thread_count': 4,
        'verbose': 0
    }
    clf_cb = CatBoostClassifier(**cb_params)
    clf_cb.fit(X_tr[::2], y_tr[::2])
    p_cb = clf_cb.predict_proba(X_te)[:, 1]

    # Train LightGBM High-Precision
    print("\n--- Training LightGBM High-Precision Classifier ---", flush=True)
    lgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(X_tr[::2], y_tr[::2])
    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]

    # Weighted Ensemble
    p_ens = 0.5 * p_cb + 0.5 * p_lgb

    print(f"\n--- Causal Evaluation (eval_r2_causal_daily) for Confluence {horizon_str} ---", flush=True)
    for name, p in [('CatBoost High-Precision', p_cb), ('LightGBM High-Precision', p_lgb), ('Ensemble Confluence', p_ens)]:
        print(f"\n<<< {name} >>>", flush=True)
        for q in [98.5, 99.0, 99.2, 99.4, 99.5]:
            acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p, y_te, ts_te, p_quantile=q)
            print(f"  Quantile P{q:4.1f}% | Win Rate: {acc*100:6.2f}% | Min Month: {min_a*100:5.2f}% | Trades/Day: {tpd:5.2f}", flush=True)

    # Save ensemble predictions for report verification
    np.save(f'p_confluence_{horizon_str.lower()}.npy', p_ens)
    np.save(f'y_te_{horizon_str.lower()}.npy', y_te)
    np.save(f'ts_te_{horizon_str.lower()}.npy', ts_te)

if __name__ == "__main__":
    train_eval_confluence("H15")
    train_eval_confluence("H30")
