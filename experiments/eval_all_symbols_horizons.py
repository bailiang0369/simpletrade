"""Multi-Symbol & Multi-Horizon Complete Evaluation Runner (Fast Version).

Evaluates ETH and BTC across H15m and H30m under strict causal non-lookahead testing rules (eval_r2_causal_daily).
Tracks monthly accuracy, monthly total trade counts, and details of bad months (< 55%).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def eval_r2_causal_daily_with_counts(p, y, ts_arr, win_days=90, p_quantile=99.0, min_hist_days=10):
    n = len(p)
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    sec_arr = ts_arr.astype(np.int64)
    day = sec_arr // 86400
    days = np.unique(day)

    day_confs = {int(d): conf[day == d] for d in days}
    day_list = days.astype(int).tolist()

    sm = np.zeros(n, bool)

    for i, d in enumerate(day_list):
        prior_days = day_list[max(0, i - win_days):i]
        if len(prior_days) < min_hist_days:
            hist = day_confs[d]
        else:
            hist = np.concatenate([day_confs[d2] for d2 in prior_days])

        tau = float(np.percentile(hist, p_quantile))
        md = (day == d)
        sm[np.where(md)[0][conf[md] >= tau]] = True

    sel = np.where(sm)[0]
    if len(sel) == 0:
        return 0.0, 0.0, 0, 0.0, {}, {}

    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)

    acc_m = {}
    count_m = {}

    for u in uniq:
        mask_m = (mts == u)
        n_m = int(mask_m.sum())
        if n_m >= 5:
            month_str = str(u)[:7]
            acc_val = float((pred[sel] == y[sel])[mask_m].mean())
            acc_m[month_str] = acc_val
            count_m[month_str] = n_m

    min_a = min(acc_m.values()) if len(acc_m) > 0 else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    overall_acc = float((pred[sel] == y[sel]).mean())
    tpd = float(sel.size) / len(days)
    return overall_acc, min_a, bad_m, tpd, acc_m, count_m

def evaluate_symbol_horizon(symbol: str, horizon_min: int):
    base_path = f"data/datasets/ds_{symbol}_h{horizon_min}.parquet"
    if not os.path.exists(base_path):
        base_path = f"data/datasets/ds_{symbol}.parquet"
    if not os.path.exists(base_path):
        print(f"Dataset {base_path} not found. Skipping.", flush=True)
        return None

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

    print(f"\n--- Evaluating {symbol} H={horizon_min}m ---", flush=True)
    clf_cb = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=7, random_seed=42, thread_count=4, verbose=0)
    clf_cb.fit(X_tr[::2], y_tr[::2])
    p_cb = clf_cb.predict_proba(X_te)[:, 1]

    clf_lgb = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=63, random_state=42, n_jobs=4, verbose=-1)
    clf_lgb.fit(X_tr[::2], y_tr[::2])
    p_lgb = clf_lgb.predict_proba(X_te)[:, 1]

    p_ens = 0.5 * p_cb + 0.5 * p_lgb

    acc, min_a, bad_m, tpd, acc_m, count_m = eval_r2_causal_daily_with_counts(p_ens, y_te, ts_te, p_quantile=99.0)

    bad_month_details = []
    for m, win_rate in acc_m.items():
        if win_rate < 0.55:
            bad_month_details.append({
                'month': m,
                'win_rate': win_rate,
                'signal_count': count_m[m]
            })

    print(f"[{symbol} H={horizon_min}m] Overall Win Rate: {acc*100:.2f}%, Daily Trades: {tpd:.2f}, Bad Months: {bad_m}", flush=True)

    return {
        'symbol': symbol,
        'horizon_min': horizon_min,
        'overall_acc': acc,
        'daily_trades': tpd,
        'bad_m_count': bad_m,
        'worst_month_acc': min_a,
        'acc_m': acc_m,
        'count_m': count_m,
        'bad_month_details': bad_month_details
    }

if __name__ == "__main__":
    results = []
    for sym in ["ETH", "BTC"]:
        for h in [15, 30]:
            res = evaluate_symbol_horizon(sym, h)
            if res:
                results.append(res)

    # Save results to json/pickle
    import json
    with open("all_symbols_horizons_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation results to all_symbols_horizons_results.json", flush=True)
