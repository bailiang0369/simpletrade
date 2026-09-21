"""严格样本外评估: 覆盖率1%下的准确率、每日交易次数、跨资产差异、稳定性(按月)。
纯 numpy 实现, 不依赖 pandas, 适用于大规模数据。
"""
import numpy as np

import config


def _to_epoch_sec(times):
    """统一为 int64 epoch 秒数组(UTC), 用于后续按日/按月统计。"""
    arr = np.asarray(times)
    if arr.dtype.kind in "OUS":            # 对象/字符串/python datetime
        arr = arr.astype("datetime64[ns]")
    # datetime64 数组
    return arr.astype("datetime64[s]").astype(np.int64)


def _day_of_epoch(sec):
    """epoch 秒 -> 天序号(UTC)。"""
    return sec // 86400


def _month_of_epoch(sec):
    """epoch 秒 -> 月份 datetime64(去掉年内偏移, 用于 unique/分组)。"""
    return sec.astype("datetime64[s]").astype("datetime64[M]")


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
    sec = _to_epoch_sec(times)

    day_sel = _day_of_epoch(sec[sel])
    # 全覆盖时间范围内的整天数 (分母)
    day_all = _day_of_epoch(sec)
    n_all_days = int(np.unique(day_all).size)
    n_days = int(np.unique(day_sel).size)
    trades_per_day = k / max(1, n_all_days)
    _, day_counts = np.unique(day_sel, return_counts=True)

    avg_ret = float(retf[sel].mean()) * 1e4   # bps
    up_mask = pred[sel] == 1
    avg_ret_up = float(retf[sel][up_mask].mean()) * 1e4 if up_mask.any() else 0.0
    avg_ret_dn = float(retf[sel][~up_mask].mean()) * 1e4 if (~up_mask).any() else 0.0

    # 按月稳定性 (纯 numpy 分组)
    m_sel = _month_of_epoch(sec[sel])
    uniq_m, m_counts = np.unique(m_sel, return_counts=True)
    acc_by_month = {}
    for mu, mc in zip(uniq_m, m_counts):
        mm = (m_sel == mu)
        acc_by_month[str(mu)] = round(float((pred[sel][mm] == y[sel][mm]).mean()), 4)
    n_by_month = {str(mu): int(mc) for mu, mc in zip(uniq_m, m_counts)}

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
        "acc_by_month": acc_by_month,
        "n_by_month": n_by_month,
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
    sec = _to_epoch_sec(times)
    day = _day_of_epoch(sec)
    uni = np.unique(day)
    sel_mask = np.zeros(len(p), dtype=bool)
    for d in uni:
        m = day == d
        k = max(1, int(round(int(m.sum()) * coverage)))
        sub = np.where(m)[0]
        top = sub[np.argsort(-conf[sub])[:k]]
        sel_mask[top] = True
    sel = np.where(sel_mask)[0]
    acc = (pred[sel] == y[sel]).mean()
    _, day_counts = np.unique(day[sel], return_counts=True)
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