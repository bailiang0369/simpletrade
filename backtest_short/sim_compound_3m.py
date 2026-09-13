#!/usr/bin/env python3
"""3分钟涨跌预测复利资金模拟。

规则:
  - 信号: 每日 top1% 置信度 (与回测/固定注额模拟一致), 合并 ETH+BTC test 段。
  - 复利: 每笔 stake = max(3.0, 当前权益*下注比例f), 保留2位小数 (小交易所要求单注>=3U)。
  - 返奖率 80: 赢 净赚 stake*0.8, 输 亏 stake。
  - 权益低于 3U 视为爆仓停止; stake 超过权益时全部投入最后一笔。
  - 无手续费。

用法:
  python -m backtest_short.sim_compound_3m            # 全扫描
  python -m backtest_short.sim_compound_3m 0.08 500   # 单点: f=0.08, u0=500
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext
from backtest_short.sim_flat_3m import get_signals

PAYOUT = 0.8
MIN_STAKE = 3.0

U0_GRID = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
F_GRID = [0.003, 0.004, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
DD_LIMIT = -0.20  # 最大回撤目标: 20%


def sim_compound(ts, pred, ysig, u0, f, max_stake=float("inf")):
    """复利模拟, 返回 (curve, stats)。

    max_stake: 单注上限(平台限制)。默认无穷大。
    """
    eq = float(u0)
    curve = [eq]
    win = (pred == ysig).astype(np.int8)
    n_wins = 0
    n = 0
    max_consec = 0; cur = 0
    for w in win:
        if eq < MIN_STAKE:
            break  # 爆仓/无法满足最小单注
        stake = min(eq, max_stake, round(max(MIN_STAKE, eq * f), 2))
        eq = eq + stake * PAYOUT if w == 1 else eq - stake
        curve.append(eq)
        n += 1
        n_wins += (w == 1)
        cur = cur + 1 if w == 0 else 0
        max_consec = max(max_consec, cur)
    curve = np.asarray(curve)
    peak = np.maximum.accumulate(curve)
    dd = curve - peak
    max_dd = dd.min()
    busted = eq < MIN_STAKE
    stats = {
        "trades": n,
        "acc": float(n_wins / max(n, 1)),
        "final": float(eq),
        "roi": float((eq - u0) / u0),
        "max_dd_u": float(max_dd),
        "max_dd_pct": float(max_dd / np.maximum(peak.max(), 1e-9)),
        "max_consec_loss": int(max_consec),
        "busted": bool(busted),
    }
    return curve, stats


def load_all_signals():
    ts_all, pred_all, ysig_all = [], [], []
    for sym in ["ETH", "BTC"]:
        ts, pred, ysig, conf = get_signals(sym, 3)
        ts_all.append(ts); pred_all.append(pred); ysig_all.append(ysig)
    ts = np.concatenate(ts_all); pred = np.concatenate(pred_all); ysig = np.concatenate(ysig_all)
    o = np.argsort(ts)
    return ts[o], pred[o], ysig[o]


def downsample(x, n_out=400):
    if len(x) <= n_out:
        return x
    ids = np.unique(np.linspace(0, len(x) - 1, n_out).astype(int))
    return x[ids]


def main():
    single = None
    max_stake = float("inf")
    if len(sys.argv) >= 3:
        single = (float(sys.argv[1]), float(sys.argv[2]))
        if len(sys.argv) >= 4:
            max_stake = float(sys.argv[3])
    ts, pred, ysig = load_all_signals()
    ts0 = ts[0]
    x_days = (ts - ts0) / 86400.0

    if single is not None:
        f, u0 = single
        curve, st = sim_compound(ts, pred, ysig, u0, f, max_stake)
        tag = f"f{int(f*1000)}_u{int(u0)}_cap{int(max_stake)}" if math.isfinite(max_stake) else f"f{int(f*1000)}_u{int(u0)}"
        print(f"# f={f:.2f} u0={u0:.0f} cap={max_stake}: 交易={st['trades']} 胜率={st['acc']:.4f} "
              f"终值={st['final']:.2f}U ROI={st['roi']*100:+.1f}% "
              f"最大回撤={st['max_dd_u']:.2f}U ({st['max_dd_pct']*100:.1f}%) "
              f"最大连亏={st['max_consec_loss']} 爆仓={st['busted']}")
        out = os.path.join(config.RESULT_DIR, f"sim_compound_{tag}.json")
        x = x_days[:len(curve)]
        json.dump({"f": f, "u0": u0, "stats": st,
                   "curve_x": downsample(x).tolist(), "curve_eq": downsample(curve).tolist()},
                  open(out, "w"))
        print(f"# 已存: {out}")
        return

    # 全扫描: 回撤矩阵 + 满足 <=20% 回撤的组合
    print(f"# 复利扫描 (合并ETH+BTC, 返奖{int(PAYOUT*100)}, 最小单注3U)")
    print(f"# {'U0':>6} | " + " | ".join(f"{'f'+str(int(f*100))+'%':>12}" for f in F_GRID))
    ok_list = []
    for u0 in U0_GRID:
        row = []
        for f in F_GRID:
            _, st = sim_compound(ts, pred, ysig, u0, f)
            if st["busted"]:
                row.append("爆仓")
                continue
            dd = st["max_dd_pct"] * 100
            cell = f"{dd:.1f}%/{st['roi']*100:.0f}%"
            if dd >= DD_LIMIT * 100:
                cell = f"[{dd:.1f}%/{st['roi']*100:.0f}%]"
                ok_list.append((u0, f, dd, st["roi"], st["final"], st["max_consec_loss"]))
            row.append(cell)
        print(f"# {u0:>6} | " + " | ".join(f"{c:>14}" for c in row))
    print(f"\n# 最大回撤<=20% 的组合 ({len(ok_list)}), 按ROI降序:")
    for u0, f, dd, roi, final, mc in sorted(ok_list, key=lambda t: -t[3]):
        print(f"  U0={u0:>5}U  f={f*100:>4.2f}%  回撤={dd:.1f}%  ROI={roi*100:+.0f}%  "
              f"终值={final:.3e}U  最大连亏={mc}")


if __name__ == "__main__":
    main()
