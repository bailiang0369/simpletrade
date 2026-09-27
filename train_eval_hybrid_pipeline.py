"""Train & Evaluate Hybrid High-Winrate Models (CatBoost + LightGBM) on Causal Hybrid Features.
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from causal_eval import eval_r2_causal_daily

def train_eval_hybrid(symbol: str = "ETH", horizon_min: int = 15):
    dataset_path = f"data/datasets/ds_{symbol}_hybrid_h{horizon_min}.parquet"
    print(f"\n=======================================================", flush=True)
    print(f"Loading Causal Hybrid Dataset for {symbol} H={horizon_min}m ({dataset_path})", flush=True)
    print(f"=======================================================", flush=True)

    df = pl.read_parquet(dataset_path)
    ignore_cols = ['ret_day', 'label', 'soft_label', 'ret_future', 'ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Rows: {len(df):,}, Hybrid Features count: {len(feat_cols)}", flush=True)

    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df['label'].to_numpy()
    ts = df['ts'].to_numpy()

    n = len(df)
    train_idx = int(n * 0.8)

    X_tr, y_tr = X[:train_idx], y[:train_idx]
    X_te, y_te, ts_te = X[train_idx:], y[train_idx:], ts[train_idx:]

    print("\n--- Training CatBoost Classifier ---", flush=True)
    clf_cb = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=7, random_seed=42, thread_count=4, verbose=0)
    clf_cb.fit(X_tr[::2], y_tr[::2])
    p_cb = clf_cb.predict_proba(X_te)[:, 1]

    print("\n--- Training LightGBM Classifier ---", flush=True)
    clf_lgb = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=63, random_state=42, n_jobs=4, verbose=-1)
    clf_lgb.fit(X_tr[::2], y_tr[::2])
    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]

    p_ens = 0.5 * p_cb + 0.5 * p_lgb

    # Strict Causal Evaluation
    p_quantile = 99.0
    pred = (p_ens >= 0.5).astype(np.int8)
    conf = np.maximum(p_ens, 1 - p_ens)
    sec_arr = ts_te.astype(np.int64)
    day = sec_arr // 86400
    days = np.unique(day)

    day_confs = {int(d): conf[day == d] for d in days}
    day_list = days.astype(int).tolist()

    sm = np.zeros(len(p_ens), bool)

    for i, d in enumerate(day_list):
        prior_days = day_list[max(0, i - 90):i]
        if len(prior_days) < 10:
            hist = day_confs[d]
        else:
            hist = np.concatenate([day_confs[d2] for d2 in prior_days])

        tau = float(np.percentile(hist, p_quantile))
        md = (day == d)
        sm[np.where(md)[0][conf[md] >= tau]] = True

    sel = np.where(sm)[0]
    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)

    acc_m = {}
    count_m = {}
    for u in uniq:
        mask_m = (mts == u)
        n_m = int(mask_m.sum())
        if n_m >= 5:
            month_str = str(u)[:7]
            acc_val = float((pred[sel] == y_te[sel])[mask_m].mean())
            acc_m[month_str] = acc_val
            count_m[month_str] = n_m

    overall_acc = float((pred[sel] == y_te[sel]).mean())
    tpd = float(sel.size) / len(days)

    bad_month_details = []
    for m, win_rate in acc_m.items():
        if win_rate < 0.55:
            bad_month_details.append({
                'month': m,
                'win_rate': win_rate,
                'signal_count': count_m[m]
            })

    out = {
        'symbol': symbol,
        'horizon_min': horizon_min,
        'overall_acc': overall_acc,
        'daily_trades': tpd,
        'bad_m_count': len(bad_month_details),
        'worst_month_acc': min(acc_m.values()) if len(acc_m) > 0 else 0.0,
        'acc_m': acc_m,
        'count_m': count_m,
        'bad_month_details': bad_month_details
    }

    print(f"\nRESULTS_{symbol}_HYBRID_H{horizon_min}: Win Rate={overall_acc*100:.2f}%, Daily Signals={tpd:.2f}, Bad Months={len(bad_month_details)}")
    with open(f"res_hybrid_{symbol}_h{horizon_min}.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    for sym in ["ETH", "BTC"]:
        for h in [15, 30]:
            train_eval_hybrid(sym, h)
