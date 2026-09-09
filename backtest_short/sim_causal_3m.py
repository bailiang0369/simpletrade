#!/usr/bin/env python3
"""3分钟涨跌 因果(无前视)资金模拟。

与实盘 RollingThreshold 一致: 每日阈值 = 前 WIN_DAYS 天置信度 |p-0.5|*2 的 P99 分位,
当日 conf>=tau 才出信号; 冷启动(历史 < COLD_MIN_DAYS)不发。用历史不用当天 -> 无前视偏差。

对比回测的 daily_top1%(当天取分位, 有前视) 以量化虚高幅度。

用法:
  python -m backtest_short.sim_causal_3m flat      # 固定5U/信号
  python -m backtest_short.sim_causal_3m comp      # 复利 5000U/1%/封顶300U
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
WIN_DAYS = 90
P99 = 99.0
COLD_MIN_DAYS = 30
MAX_HIST_DAYS = 200          # 滑动窗口天数上限(足够覆盖90天, 并限内存)

CAPITAL0 = 5000.0
STAKE = 5.0
PAYOUT = 0.8


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
    R = family_ranks(P)
    wp = f"{config.MODEL_DIR}/pool_short_h{horizon}/ens_family_w.json"
    if os.path.exists(wp):
        w = np.asarray(json.load(open(wp))[symbol]["w"], np.float64)
    else:
        w = np.ones(3) / 3
    wf = np.repeat(w, len(SEEDS)); wf = wf / wf.sum()
    return (wf[:, None] * R).sum(axis=0)


def causal_signals(symbol, horizon=3):
    """逐符号因果信号: 返回 (ts, pred, y). 按时间排序。"""
    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")
    split = "test"
    p = ens_p(load_P(symbol, horizon, split), symbol, horizon)
    y = ctx.y(split)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)
    o = np.argsort(sec)
    sec, conf, pred, y = sec[o], conf[o], pred[o], y[o]
    days = np.unique(sec // 86400)
    day_of = sec // 86400
    # 每天置信度列表 (供历史窗口)
    day_confs = {int(d): conf[day_of == d] for d in days}
    day_list = days.astype(int).tolist()
    sig_ts, sig_pred, sig_y = [], [], []
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - WIN_DAYS):i]        # 前 WIN_DAYS 天(不含当天)
        if len(prior) < COLD_MIN_DAYS:
            continue                                    # 冷启动跳过
        hist = np.concatenate([day_confs[d2] for d2 in prior])
        tau = float(np.percentile(hist, P99))           # 前90天 99 分位
        m = day_of == d
        rc = conf[m] >= tau
        sig_ts.append(sec[m][rc]); sig_pred.append(pred[m][rc]); sig_y.append(y[m][rc])
    ts = np.concatenate(sig_ts); pr = np.concatenate(sig_pred); gt = np.concatenate(sig_y)
    o2 = np.argsort(ts)
    return ts[o2], pr[o2], gt[o2]


def flat_sim(ts, pred, ysig):
    n = len(ts)
    win = (pred == ysig).astype(np.int8)
    pnl = np.where(win == 1, STAKE * PAYOUT, -STAKE)
    eq = CAPITAL0 + np.concatenate([[0], np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq); dd = eq - peak
    stats = {"trades": int(n), "wins": int(win.sum()), "acc": float(win.mean()),
             "final": float(eq[-1]), "profit": float(eq[-1] - CAPITAL0),
             "roi": float((eq[-1] - CAPITAL0) / CAPITAL0), "max_dd_u": float(dd.min()),
             "max_dd_pct": float(dd.min() / np.maximum(peak.max(), 1e-9)),
             "days": float((ts[-1] - ts[0]) / 86400),
             "trades_per_day": float(n * 86400 / max(ts[-1] - ts[0], 1))}
    return eq, stats


def comp_sim(ts, pred, ysig, u0=5000.0, f=0.01, max_stake=300.0):
    eq = float(u0); win = (pred == ysig).astype(np.int8)
    cur = 0; mc = 0; n = 0; nw = 0
    for w in win:
        stake = min(eq, max_stake, round(max(3.0, eq * f), 2))
        eq = eq + stake * PAYOUT if w == 1 else eq - stake
        n += 1; nw += (w == 1)
        cur = cur + 1 if w == 0 else 0; mc = max(mc, cur)
    peak = 0.0
    # 重算回撤
    owin = (pred == ysig).astype(np.int8)
    eqc = float(u0); pkc = eqc; maxdd = 0.0
    for w in owin:
        stk = min(eqc, max_stake, round(max(3.0, eqc * f), 2))
        eqc = eqc + stk * PAYOUT if w == 1 else eqc - stk
        pkc = max(pkc, eqc)
        if pkc > 0:
            maxdd = min(maxdd, (eqc - pkc) / pkc)
    stats = {"trades": int(n), "acc": float(nw / max(n, 1)), "final": float(eq),
             "roi": float((eq - u0) / u0), "max_dd_pct": float(maxdd),
             "max_consec_loss": int(mc)}
    return np.asarray([u0]), stats


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    sigs = []
    for sym in ["ETH", "BTC"]:
        ts, pr, gt = causal_signals(sym, 3)
        sigs.append((ts, pr, gt))
        print(f"# 因果信号 {sym}: {len(ts)} 笔", flush=True)
    ts = np.concatenate([s[0] for s in sigs]); pr = np.concatenate([s[1] for s in sigs])
    gt = np.concatenate([s[2] for s in sigs]); o = np.argsort(ts)
    ts, pr, gt = ts[o], pr[o], gt[o]
    print(f"# 因果信号 ALL(合并): {len(ts)} 笔 ({len(ts)/max(ts[-1]-ts[0],1)*86400:.1f}笔/天)\n", flush=True)

    if mode in ("flat", "both"):
        eq, st = flat_sim(ts, pr, gt)
        print(f"[flat] 交易={st['trades']} 胜率={st['acc']:.4f} 终值={st['final']:.1f}U "
              f"ROI={st['roi']*100:+.2f}% 最大回撤={st['max_dd_u']:.1f}U ({st['max_dd_pct']*100:.2f}%) "
              f"笔/天={st['trades_per_day']:.1f}", flush=True)
        out = os.path.join(config.RESULT_DIR, "sim_causal_flat.json")
        x = (ts - ts[0]) / 86400.0
        n = len(x); ids = np.unique(np.linspace(0, n - 1, 300).astype(int))
        json.dump({"stats": st, "curve_x": x[ids].tolist(), "curve_eq": eq[ids].tolist()},
                  open(out, "w"))
        print(f"# 已存 {out}\n", flush=True)

    if mode in ("comp", "both"):
        for cap in [float("inf"), 300.0]:
            _, st = comp_sim(ts, pr, gt, 5000.0, 0.01, cap)
            lbl = "无封顶" if cap == float("inf") else f"封顶{int(cap)}U"
            print(f"[comp 1% {lbl}] 交易={st['trades']} 胜率={st['acc']:.4f} "
                  f"终值={st['final']:.0f}U ROI={st['roi']*100:+.1f}% "
                  f"最大回撤={st['max_dd_pct']*100:.2f}% 最大连亏={st['max_consec_loss']}", flush=True)


if __name__ == "__main__":
    main()