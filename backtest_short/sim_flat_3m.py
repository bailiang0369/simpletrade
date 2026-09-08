#!/usr/bin/env python3
"""3分钟涨跌预测资金曲线模拟 (固定5U/信号, 返奖率80, 本金5000U)。

信号定义与回测一致: 每日 top1% 置信度样本 (family权重集成, test段)。
规则: 每笔投入 stake=5U; 预测正确 返回 stake*(1+payout), 净赚 stake*payout=4U;
      预测错误 全亏 stake=5U。不做复利, 无手续费。

用法:
  python -m backtest_short.sim_flat_3m
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
CAPITAL0 = 5000.0
STAKE = 5.0
PAYOUT = 0.8          # 返奖率 80: 赢 +4U, 输 -5U
FRAC = 0.01           # 每日 top1%


def load_P(symbol, horizon, split):
    return np.concatenate([
        np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
        for f in FAMILIES
    ], axis=0).astype(np.float64)


def family_ranks(P):
    n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def daily_topk_sel(p, sec_arr, frac=FRAC):
    conf = np.maximum(p, 1 - p)
    day = sec_arr // 86400
    days = np.unique(day)
    n = len(p); sel = np.zeros(n, bool)
    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * frac)))
        sub = np.where(md)[0]
        sel[sub[np.argsort(-conf[sub])[:kd]]] = True
    return np.where(sel)[0]


def get_signals(symbol, horizon=3):
    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")
    split = "test"
    P = load_P(symbol, horizon, split)
    R = family_ranks(P)
    wp = f"{config.MODEL_DIR}/pool_short_h{horizon}/ens_family_w.json"
    if os.path.exists(wp):
        w = np.asarray(json.load(open(wp))[symbol]["w"], np.float64)
    else:
        w = np.ones(3) / 3
    wf = np.repeat(w, len(SEEDS)); wf = wf / wf.sum()
    p = (wf[:, None] * R).sum(axis=0)
    y = ctx.y(split)
    sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    idx = daily_topk_sel(p, sec_arr, FRAC)
    pred = (p[idx] >= 0.5).astype(np.int8)
    ysig = y[idx]
    ts = sec_arr[idx]
    conf = np.maximum(p[idx], 1 - p[idx])
    order = np.argsort(ts)
    return ts[order], pred[order], ysig[order], conf[order]


def simulate(ts, pred, ysig):
    """返回 (equity 曲线, 统计)。"""
    n = len(ts)
    win = (pred == ysig).astype(np.int8)
    pnl = np.where(win == 1, STAKE * PAYOUT, -STAKE).astype(np.float64)
    equity = CAPITAL0 + np.concatenate([[0], np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    # 最大连续亏损
    max_consec = 0; cur = 0
    for w in win:
        cur = cur + 1 if w == 0 else 0
        max_consec = max(max_consec, cur)
    stats = {
        "trades": int(n),
        "wins": int(win.sum()),
        "losses": int(n - win.sum()),
        "acc": float(win.mean()),
        "final_equity": float(equity[-1]),
        "profit": float(equity[-1] - CAPITAL0),
        "roi": float((equity[-1] - CAPITAL0) / CAPITAL0),
        "max_dd": float(max_dd),
        "max_dd_pct": float(max_dd / np.maximum(peak.max(), 1e-9)),
        "max_consec_loss": int(max_consec),
        "days": float((ts[-1] - ts[0]) / 86400),
        "trades_per_day": float(n * 86400 / max((ts[-1] - ts[0]), 1)),
    }
    return equity, stats


def downsample(ts, equity, n_out=400):
    ts0 = ts[0]
    x = (ts - ts0) / 86400.0  # 天数
    if len(x) <= n_out:
        return x, equity
    ids = np.unique(np.linspace(0, len(x) - 1, n_out).astype(int))
    return x[ids], equity[ids]


def main():
    results = {}
    comb_ts = []; comb_eq = None
    for sym in ["ETH", "BTC"]:
        ts, pred, ysig, conf = get_signals(sym, 3)
        equity, st = simulate(ts, pred, ysig)
        x, eq = downsample(ts, equity)
        results[sym] = {"stats": st, "curve_x": x.tolist(), "curve_eq": eq.tolist()}
        comb_ts.append(ts)
        print(f"# {sym} 3m: 交易={st['trades']} 胜率={st['acc']:.4f} "
              f"终值={st['final_equity']:.1f}U 利润={st['profit']:+.1f}U "
              f"ROI={st['roi']*100:+.2f}% 最大回撤={st['max_dd']:.1f}U "
              f"最大连亏={st['max_consec_loss']} 天数={st['days']:.0f}")
    # 合并 ETH+BTC
    ts_all = np.sort(np.concatenate(comb_ts))
    eq_c, eq_b, eq_e = results["ETH"]["stats"], results["BTC"]["stats"], None
    # 重建合并曲线: 按时间把两币信号交错
    sigs = []
    for sym in ["ETH", "BTC"]:
        ts, pred, ysig, conf = get_signals(sym, 3)
        sigs.append((ts, pred, ysig))
    ts_all, pred_all, ysig_all = [np.concatenate([s[i] for s in sigs]) for i in range(3)]
    o = np.argsort(ts_all)
    equity, st = simulate(ts_all[o], pred_all[o], ysig_all[o])
    x, eq = downsample(ts_all[o], equity)
    results["ALL"] = {"stats": st, "curve_x": x.tolist(), "curve_eq": eq.tolist()}
    print(f"# ALL(ETH+BTC) 3m: 交易={st['trades']} 胜率={st['acc']:.4f} "
          f"终值={st['final_equity']:.1f}U 利润={st['profit']:+.1f}U "
          f"ROI={st['roi']*100:+.2f}% 最大回撤={st['max_dd']:.1f}U "
          f"最大连亏={st['max_consec_loss']} 天数={st['days']:.0f}")
    out = os.path.join(config.RESULT_DIR, "sim_3m_flat.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"\n# 结果已存: {out}")


if __name__ == "__main__":
    main()
