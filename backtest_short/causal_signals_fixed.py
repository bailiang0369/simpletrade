#!/usr/bin/env python3
"""修正版因果信号: 保持每日 ≈1% 覆盖率的同时无前视偏差.

原 sim_causal_3m.py 用「前 90 天全局 P99」当阈值——对长周期完全不公平
(牛市/熊市/震荡期置信度分布大幅漂移, 全局 P99 与当日 tail 完全脱节).

修正方案: 前 90 天每一天「当日 top1% 阈值」的 P99 作为当日阈值.
- 因果性: 只用历史每日 top1% 阈值, 不用当日分布
- 密度: 保持每天 ≈ 1% 覆盖率的信号数
- 稳定性: 阈值是每日相对值的 P99, 不随宏观漂移失控

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
WIN_DAYS = 90
COLD_MIN_DAYS = 30
PERCENTILE = 99.0  # 前 N 天每日 top1% 阈值的 P99


def load_P(symbol, horizon, split):
    return np.concatenate([
        np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
        for f in FAMILIES
    ], axis=0).astype(np.float64)


def family_ranks(P):
    n = P.shape[1]; R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def ens_p(P, symbol, horizon):
    n = P.shape[1]
    R = family_ranks(P)
    wp = f"{config.MODEL_DIR}/pool_short_h{horizon}/ens_family_w.json"
    if os.path.exists(wp):
        w = np.asarray(json.load(open(wp))[symbol]["w"], np.float64)
    else:
        w = np.ones(3) / 3
    wf = np.repeat(w, len(SEEDS)); wf = wf / wf.sum()
    return (wf[:, None] * R).sum(axis=0)


def causal_signals_fixed(symbol, horizon=3):
    """修正版因果信号: 每日阈值 = 前 90 天每日 top1% 阈值的 P99.
    
    返回 (ts, pred, y), 按时间排序, 无同分钟重复.
    """
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
    
    # 第一步: 计算每天的 top1% 阈值 (不用当日完整分布, 只用 conf 列表排序取 P99)
    day_list = days.astype(int).tolist()
    daily_tau = {}  # day -> 当日 top1% 置信度阈值
    for d in day_list:
        m = day_of == d
        day_conf = conf[m]
        daily_tau[d] = float(np.percentile(day_conf, 99))
    
    # 第二步: 逐天因果决定当日阈值
    sig_ts, sig_pred, sig_y = [], [], []
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - WIN_DAYS):i]
        if len(prior) < COLD_MIN_DAYS:
            continue  # 冷启动跳过
        # 关键改动: 阈值 = 前90天每日 top1% 阈值的 P99, 而不是全局样本的 P99
        prior_taus = [daily_tau[d2] for d2 in prior]
        tau = float(np.percentile(prior_taus, PERCENTILE))
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
