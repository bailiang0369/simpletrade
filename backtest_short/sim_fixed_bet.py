#!/usr/bin/env python3
"""固定下注模拟: 5000U 总资金, 每个信号固定 5U, 返奖 80%(赢 +4U, 输 -5U)。

信号 = 测试段全部样本的模型预测(方向 p>=0.5 看涨, 否则看跌), 不做任何筛选/去重,
(family 权重取 tune_ensemble_short 保存的 ens_family_w.json)。
所有信号按时间顺序逐笔结算, 输出资金曲线与关键指标, 并存 JSON 供前端展示。

用法:
  python -m backtest_short.sim_fixed_bet
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
COMBOS = [(3, "ETH"), (3, "BTC"), (5, "ETH"), (5, "BTC")]
BET = 5.0          # 每信号下注 U
PAYOUT = 0.8       # 返奖率: 赢返还下注额的 80% 作为盈利
INIT_CAP = 5000.0


def family_ranks(P):
    n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def sim_combo(horizon, symbol, w):
    """返回 (times, equity_curve, trades 统计)。信号=测试段全部样本, 不做筛选/去重。"""
    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")
    split = "test"
    P = np.concatenate([
        np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
        for f in FAMILIES
    ], axis=0).astype(np.float64)
    R = family_ranks(P)
    wf = np.repeat(np.asarray(w, np.float64), len(SEEDS)); wf = wf / wf.sum()
    p = (wf[:, None] * R).sum(axis=0)
    y = ctx.y(split)
    sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    idx = np.arange(len(p))   # 全部样本都是信号, 不去重/不筛选

    order = idx[np.argsort(sec_arr[idx])]
    times = sec_arr[order]
    pred = (p[order] >= 0.5).astype(np.int8)
    win = (pred == y[order])

    eq = INIT_CAP
    curve = []
    peak = INIT_CAP
    max_dd = 0.0
    wins = 0
    for t, ok in zip(times, win):
        eq += BET * PAYOUT if ok else -BET
        wins += int(ok)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        curve.append([int(t), round(eq, 2)])
    n = len(order)
    pnl = sum(BET * PAYOUT if ok else -BET for ok in win)
    return {
        "horizon": horizon, "symbol": symbol,
        "w": [round(float(x), 3) for x in w],
        "n_trades": n,
        "wins": wins,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "pnl": round(pnl, 2),
        "final_equity": round(eq, 2),
        "total_return_pct": round((eq - INIT_CAP) / INIT_CAP * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_pnl_per_trade": round(pnl / n, 4) if n else 0.0,
        "curve": curve,
        "first_ts": int(times[0]) if n else None,
        "last_ts": int(times[-1]) if n else None,
    }


def main():
    wpath = lambda h: f"/workspace/models_saved/pool_short_h{h}/ens_family_w.json"
    ws = {h: json.load(open(wpath(h))) for h in (3, 5)}

    out = {"INIT_CAP": INIT_CAP, "BET": BET, "PAYOUT": PAYOUT, "combos": []}
    for horizon, symbol in COMBOS:
        w = ws[horizon].get(symbol, {}).get("w", [1/3]*3)
        r = sim_combo(horizon, symbol, w)
        out["combos"].append(r)
        print(f"h{horizon} {symbol}: 信号={r['n_trades']} 胜率={r['win_rate']:.2%} "
              f"盈亏={r['pnl']:+.1f}U 终值={r['final_equity']:.1f}U "
              f"({r['total_return_pct']:+.2f}%) 最大回撤={r['max_drawdown_pct']:.2f}% "
              f"单笔均值={r['avg_pnl_per_trade']:+.4f}U", flush=True)

    # 四组全部信号合并(时间序), 每个仍 5U
    all_trades = []
    for r in out["combos"]:
        times = np.asarray([c[0] for c in r["curve"]])
        eqs = np.asarray([c[1] for c in r["curve"]])
        # 从 curve 反推每笔胜负(equity 变化)
        d = np.diff(np.concatenate([[INIT_CAP], eqs]))
        for t, dd in zip(times, d):
            all_trades.append((int(t), dd > 0))
    all_trades.sort(key=lambda x: x[0])
    eq = INIT_CAP; peak = INIT_CAP; max_dd = 0.0; wins = 0
    curve = []
    for t, ok in all_trades:
        eq += BET * PAYOUT if ok else -BET
        wins += int(ok)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        curve.append([t, round(eq, 2)])
    n = len(all_trades)
    pnl = sum(BET * PAYOUT if ok else -BET for _, ok in all_trades)
    comb = {
        "horizon": "ALL", "symbol": "ALL4",
        "n_trades": n, "wins": wins,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "pnl": round(pnl, 2),
        "final_equity": round(eq, 2),
        "total_return_pct": round((eq - INIT_CAP) / INIT_CAP * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_pnl_per_trade": round(pnl / n, 4) if n else 0.0,
        "curve": curve,
        "first_ts": curve[0][0] if curve else None,
        "last_ts": curve[-1][0] if curve else None,
    }
    out["combos"].append(comb)
    print(f"ALL4 合并: 信号={n} 胜率={comb['win_rate']:.2%} 盈亏={comb['pnl']:+.1f}U "
          f"终值={comb['final_equity']:.1f}U ({comb['total_return_pct']:+.2f}%) "
          f"最大回撤={comb['max_drawdown_pct']:.2f}%", flush=True)

    with open("/workspace/backtest_short/sim_fixed_bet_out.json", "w") as f:
        json.dump(out, f)
    print("# done")


if __name__ == "__main__":
    main()
