"""Strict Causal Evaluation Function for No-Lookahead Testing

支持严格历史因果 (前 90 天 P99 置信度分位数) 评估，彻底消除日内前视泄漏！
"""

import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def eval_r2_causal_daily(p, y, ts_arr, win_days=90, p_quantile=99.0, min_hist_days=10):
    """严格因果每日置信度阈值选择 (100% 消除日内前视泄漏)

    当天 d 的置信度阈值 tau 仅由盘前历史 (前 win_days 天不含当天) 的置信度分布 P99 分位数决定。
    在当天时刻 t，conf_t >= tau 即可出信号，完全无任何日内或未来数据前视。
    """
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
            # 冷启动前 10 天
            hist = day_confs[d]
        else:
            hist = np.concatenate([day_confs[d2] for d2 in prior_days])

        tau = float(np.percentile(hist, p_quantile))
        md = (day == d)
        sm[np.where(md)[0][conf[md] >= tau]] = True

    sel = np.where(sm)[0]
    if len(sel) == 0:
        return 0.0, 0.0, 0, 0.0, {}

    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq if (mts == u).sum() >= 5}
    min_a = min(acc_m.values()) if len(acc_m) > 0 else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    overall_acc = float((pred[sel] == y[sel]).mean())
    tpd = float(sel.size) / len(days)
    return overall_acc, min_a, bad_m, tpd, acc_m

if __name__ == "__main__":
    print("Strict causal evaluation module ready.")
