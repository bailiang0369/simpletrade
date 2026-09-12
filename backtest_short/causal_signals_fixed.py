#!/usr/bin/env python3
"""朴素因果信号: 无前视偏差.

用户口径: 前 WIN_DAYS 天(一个月=30天或更长时间)历史预测置信度 |p-0.5|*2 取 P99(top1%)
当阈值; 当日信号只和这个历史阈值比较, conf>=tau 就出信号. 出多少算多少,
不强制每天覆盖 1%. 只用历史不用当天 -> 无前视.

用法: 训练完模型后
  from backtest_short.causal_signals_fixed import causal_signals_fixed
  ts, pred, y = causal_signals_fixed("ETH", horizon=10)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
WIN_DAYS = 30        # 用户口径: 前一个月
COLD_MIN_DAYS = 30   # 冷启动: 历史不足一个月不发信号
PERCENTILE = 99.0    # 前 N 天历史 conf 的 P99 (top1% 分位点)


def load_P(symbol, horizon, split):
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
          for f in FAMILIES]
    P = np.stack(Ps, axis=0)  # (3,n) 单seed 或 (3,5,n) 5-seed
    if P.ndim == 3:
        P = P.reshape(-1, P.shape[-1])  # (15,n)
    return P.astype(np.float64)


def family_ranks(P):
    n = P.shape[1]; R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def ens_p(P, symbol, horizon):
    # P shape: (n_models, n_samples), 3family×1seed=3 或 3family×5seed=15
    return family_ranks(P).mean(axis=0)


def causal_signals_fixed(symbol, horizon=3, win_days=None, percentile=None):
    """朴素因果信号: 当日阈值 = 前 win_days 天历史 conf 的 P99.

    不强制每日 1% 覆盖率, 当日 conf>=tau 就出信号, 出多少算多少.
    返回 (ts, pred, y), 按时间排序, 无同分钟重复.
    """
    win_days = win_days or WIN_DAYS
    percentile = percentile or PERCENTILE
    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")
    split = "test"
    p = ens_p(load_P(symbol, horizon, split), symbol, horizon)
    y = ctx.y(split)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)
    o = np.argsort(sec); sec, conf, pred, y = sec[o], conf[o], pred[o], y[o]

    days = np.unique(sec // 86400)
    day_of = sec // 86400
    day_confs = {int(d): conf[day_of == d] for d in days}
    day_list = days.astype(int).tolist()

    # 逐天: 阈值 = 前 win_days 天(不含当天)全部历史 conf 的 P99
    sig_ts, sig_pred, sig_y = [], [], []
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - win_days):i]
        if len(prior) < COLD_MIN_DAYS:
            continue  # 冷启动: 历史不足一个月不发
        hist = np.concatenate([day_confs[d2] for d2 in prior])
        tau = float(np.percentile(hist, percentile))
        m = day_of == d
        rc = conf[m] >= tau
        sig_ts.append(sec[m][rc])
        sig_pred.append(pred[m][rc])
        sig_y.append(y[m][rc])

    ts = np.concatenate(sig_ts); pr = np.concatenate(sig_pred); gt = np.concatenate(sig_y)
    o2 = np.argsort(ts); ts, pr, gt = ts[o2], pr[o2], gt[o2]
    # 同分钟去重
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return ts[keep], pr[keep], gt[keep]


def quick_validate(horizon=3):
    """对比两套阈值机制的信号密度差异."""
    from sim_causal_3m import causal_signals as causal_old
    results = []
    for sym in ["ETH", "BTC"]:
        ts_old, _, _ = causal_old(sym, horizon)
        ts_new, _, _ = causal_signals_fixed(sym, horizon)
        results.append((sym, ts_old, ts_new))
    all_old = np.concatenate([r[1] for r in results])
    all_new = np.concatenate([r[2] for r in results])
    days = len(set(all_new // 86400))
    print(f"=== h{horizon}: 两套阈值对比 (ETH+BTC) ===")
    print(f"  因果全局P99(旧): {len(all_old)} 笔, {len(all_old)/max(days,1):.1f} 笔/天, "
          f"零信号天={sum(1 for d in set(all_old//86400) if (all_old//86400==d).sum()<5)}")
    print(f"  每日因果top1%(新): {len(all_new)} 笔, {len(all_new)/max(days,1):.1f} 笔/天")


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    quick_validate(h)
