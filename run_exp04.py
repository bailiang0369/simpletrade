"""EXP-04 Evaluation Script: Base + Concise Oscillator Joint Feature Model.
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

def run_exp04(horizon_str: str):
    exp04_path = f"data/datasets/ds_ETH_exp04_{horizon_str.lower()}.parquet"
    base_path = f"data/datasets/ds_ETH_{horizon_str.lower()}.parquet"

    print(f"\n=======================================================", flush=True)
    print(f"Loading EXP-04 Dataset ({exp04_path}) + Base ({base_path})", flush=True)
    print(f"=======================================================", flush=True)

    df_exp = pl.read_parquet(exp04_path)
    df_base = pl.read_parquet(base_path)

    df_joint = df_base.join(df_exp, on="ts", suffix="_exp")

    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'ret_day_exp', 'label_exp', 'soft_label_exp', 'ret_future_exp']
    feat_cols = [c for c in df_joint.columns if c not in ignore_cols]

    print(f"Joint rows: {len(df_joint):,}, Joint features count: {len(feat_cols)}", flush=True)

    X = df_joint.select(feat_cols).to_numpy().astype(np.float32)
    y = df_joint['label'].to_numpy()
    ts = df_joint['ts'].to_numpy()

    n = len(df_joint)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    # Train CatBoost
    print("\n--- Training CatBoost Classifier ---", flush=True)
    clf_cb = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=7, random_seed=42, thread_count=4, verbose=0)
    clf_cb.fit(X_tr[::2], y_tr[::2])
    p_cb = clf_cb.predict_proba(X_te)[:, 1]

    # Train LightGBM
    print("\n--- Training LightGBM Classifier ---", flush=True)
    clf_lgb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=4, verbose=-1)
    clf_lgb.fit(X_tr[::2], y_tr[::2])
    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]

    # Weighted Ensemble
    p_ens = 0.5 * p_cb + 0.5 * p_lgb

    print(f"\n--- Causal Evaluation (eval_r2_causal_daily) for EXP-04 {horizon_str} ---", flush=True)
    for name, p in [('CatBoost EXP-04', p_cb), ('LightGBM EXP-04', p_lgb), ('Ensemble EXP-04', p_ens)]:
        print(f"\n<<< {name} >>>", flush=True)
        for q in [98.5, 99.0, 99.2, 99.5]:
            acc, min_a, bad_m, tpd, acc_m = eval_r2_causal_daily(p, y_te, ts_te, p_quantile=q)
            print(f"  Quantile P{q:4.1f}% | Win Rate: {acc*100:6.2f}% | Min Month: {min_a*100:5.2f}% | Trades/Day: {tpd:5.2f}", flush=True)

if __name__ == "__main__":
    run_exp04("H15")
    run_exp04("H30")
