"""Combined Baseline + Multi-Timeframe Stoch Feature Trainer (Memory Efficient).
"""

import os, sys, time, gc, warnings
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def train_eval_combined_mem_efficient(horizon_str: str):
    base_path = f"data/datasets/ds_ETH_{horizon_str.lower()}.parquet"
    stoch_path = f"data/datasets/ds_ETH_stoch_{horizon_str.lower()}.parquet"

    print(f"\n=======================================================", flush=True)
    print(f"Loading Base ({base_path}) + Stoch ({stoch_path})", flush=True)
    print(f"=======================================================", flush=True)

    df_base = pl.read_parquet(base_path)
    df_stoch = pl.read_parquet(stoch_path)

    # Select key features to prevent OOM
    df_joint = df_base.join(df_stoch, on="ts", suffix="_stoch")

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'ret_day_stoch', 'label_stoch', 'soft_label_stoch', 'ret_future_stoch']
    feat_cols = [c for c in df_joint.columns if c not in ignore_cols]

    # Cast to Float32 to save 50% RAM
    df_joint = df_joint.select([pl.col(c).cast(pl.Float32) for c in feat_cols] + [pl.col('label').cast(pl.Int8), pl.col('ts')])

    print(f"Joint rows: {len(df_joint):,}, Joint features count: {len(feat_cols)}", flush=True)

    X = df_joint.select(feat_cols).to_numpy()
    y = df_joint['label'].to_numpy()
    ts = df_joint['ts'].to_numpy()

    del df_base, df_stoch, df_joint
    gc.collect()

    n = len(X)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Subsample train by 2
    X_tr_sub = X_tr[::2]
    y_tr_sub = y_tr[::2]

    print(f"Subsampled Train: {len(X_tr_sub):,}, Test: {len(X_te):,}", flush=True)

    print("\n--- Training LightGBM Combined Model ---", flush=True)
    lgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
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
    print(f"Combined LGBM Test AUC: {auc_lgb:.4f}", flush=True)

    print(f"\n--- Causal Evaluation (eval_r2_causal_daily) for Combined {horizon_str} ---", flush=True)
    for q in [98.5, 99.0, 99.2, 99.4, 99.5]:
        acc_lgb, min_lgb, bad_lgb, tpd_lgb, _ = eval_r2_causal_daily(p_lgb, y_te, ts_te, p_quantile=q)
        print(f"[Quantile P{q:4.1f}% | Horizon: {horizon_str}] -> Win Rate: {acc_lgb*100:6.2f}% | Min Mo: {min_lgb*100:5.2f}% | Trades/Day: {tpd_lgb:5.2f}", flush=True)

if __name__ == "__main__":
    train_eval_combined_mem_efficient("H15")
    train_eval_combined_mem_efficient("H30")
