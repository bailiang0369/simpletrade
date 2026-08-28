"""严格样本外评估: 覆盖率1%下的准确率、每日交易次数、跨资产差异、稳定性(按月)。
"""
import numpy as np
import pandas as pd

import config


def evaluate_topk(p, y, retf, times, coverage=None, return_sel=False):
    """p: 上涨概率; y: 真实标签; retf: 未来收益; times: 时间戳。
    按置信度 max(p,1-p) 取前 coverage 比例的样本, 计算准确率等指标。
    """
    coverage = coverage or config.COVERAGE
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int8)
    n = len(p)
    k = max(1, int(round(n * coverage)))
    sel = np.argsort(-conf)[:k]
    acc = (pred[sel] == y[sel]).mean()
    days = pd.Series(pd.to_datetime(times[sel])).dt.normalize()
    n_days = days.nunique()
    trades_per_day = k / max(1, len(np.unique(pd.to_datetime(times).normalize())))
    day_counts = days.value_counts()
    avg_ret = float(retf[sel].mean()) * 1e4   # bps
    up_mask = pred[sel] == 1
    avg_ret_up = float(retf[sel][up_mask].mean()) * 1e4 if up_mask.any() else 0.0
    avg_ret_dn = float(retf[sel][~up_mask].mean()) * 1e4 if (~up_mask).any() else 0.0
    # 按月稳定性
    m = pd.to_datetime(times[sel]).to_period("M")
    acc_by_month = (pd.Series((pred[sel] == y[sel]).astype(int), index=m)
                    .groupby(level=0).mean())
    n_by_month = m.value_counts().sort_index()
    r = {
        "coverage": coverage,
        "n": n,
        "k": int(k),
        "accuracy": float(acc),
        "n_days_selected": int(n_days),
        "trades_per_day": float(trades_per_day),
        "day_counts_min": int(day_counts.min()) if len(day_counts) else 0,
        "day_counts_max": int(day_counts.max()) if len(day_counts) else 0,
        "avg_ret_bps": float(avg_ret),
        "avg_ret_up_bps": float(avg_ret_up),
        "avg_ret_dn_bps": float(avg_ret_dn),
        "conf_mean": float(conf[sel].mean()),
        "acc_by_month": acc_by_month.round(4).to_dict(),
        "n_by_month": n_by_month.to_dict(),
    }
    if return_sel:
        r["_sel"] = sel
    return r


def evaluate_topk_daily(p, y, retf, times, coverage=None):
    """按天分别取置信度前coverage比例(每天约14.4单), 汇总准确率与每日交易数。"""
    coverage = coverage or config.COVERAGE
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int8)
    times = pd.to_datetime(times)
    day = times.normalize()
    uni = pd.unique(day)
    sel_mask = np.zeros(len(p), dtype=bool)
    for d in uni:
        m = day == d
        k = max(1, int(round(m.sum() * coverage)))
        sub = np.where(m)[0]
        top = sub[np.argsort(-conf[sub])[:k]]
        sel_mask[top] = True
    sel = np.where(sel_mask)[0]
    acc = (pred[sel] == y[sel]).mean()
    day_counts = pd.Series(day[sel]).value_counts()
    return {
        "coverage": coverage,
        "k": int(len(sel)),
        "accuracy": float(acc),
        "n_days_selected": int(len(uni)),
        "trades_per_day": float(len(sel) / len(uni)),
        "day_counts_min": int(day_counts.min()),
        "day_counts_max": int(day_counts.max()),
        "avg_ret_bps": float(retf[sel].mean()) * 1e4,
    }


def stability_report(acc_b, acc_e):
    """跨资产稳定性: 两资产top1%准确率差值。返回是否满足<=3pp。"""
    delta = abs(acc_b - acc_e)
    return delta, delta <= config.CROSS_ASSET_MAX_DELTA


def check_all(metrics_btc, metrics_eth, verbose=True):
    """综合验收: 覆盖率/准确率/每日交易次数/跨资产差值。"""
    ok = True
    checks = []
    for name, m in [("BTC", metrics_btc), ("ETH", metrics_eth)]:
        a_ok = m["accuracy"] >= config.TARGET_ACCURACY
        c_ok = abs(m["coverage"] - config.COVERAGE) < 1e-6
        t_ok = m["trades_per_day"] >= config.MIN_TRADES_PER_DAY
        checks.append((name, a_ok, c_ok, t_ok))
        ok &= (a_ok and c_ok and t_ok)
    delta, d_ok = stability_report(metrics_btc["accuracy"], metrics_eth["accuracy"])
    ok &= d_ok
    if verbose:
        for name, a_ok, c_ok, t_ok in checks:
            print(f"[check] {name}: acc>=65:{a_ok} coverage==1%:{c_ok} trades/day>=14:{t_ok}")
        print(f"[check] cross-asset delta={delta*100:.2f}pp <=3pp: {d_ok}")
        print(f"[check] ALL PASS: {ok}")
    return ok, delta
