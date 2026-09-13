#!/usr/bin/env python3
"""四周期(3/5/10/15min)汇总:
- 当日前视 top1% (原回测法)
- 因果 前90天滚动99分位 + greedy_sparse(间距=horizon分钟)去重
输出: 胜率/交易数/固定5U ROI / 复利20%回撤档估算
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext
from backtest_short.sim_flat_3m import load_P, family_ranks, daily_topk_sel
from backtest_short.sim_causal_3m import flat_sim, comp_sim, causal_signals
from live.signals import greedy_sparse

HORIZONS = [3, 5, 10, 15]


def ens_p_symbol(sym, H):
    ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
    split = "test"
    P = load_P(sym, H, split); R = family_ranks(P)
    wp = f"{config.MODEL_DIR}/pool_short_h{H}/ens_family_w.json"
    w = np.asarray(json.load(open(wp))[sym]["w"], np.float64)
    wf = np.repeat(w, 5); wf = wf / wf.sum()
    p = (wf[:, None] * R).sum(0)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    return p, sec, ctx.y(split)


def lookahead_signals(H):
    ts, pr, gt = [], [], []
    for sym in ["ETH", "BTC"]:
        p, sec, y = ens_p_symbol(sym, H)
        idx = daily_topk_sel(p, sec)
        ts.append(sec[idx]); pr.append((p[idx] >= 0.5).astype(np.int8)); gt.append(y[idx])
    ts = np.concatenate(ts); pr = np.concatenate(pr); gt = np.concatenate(gt)
    o = np.argsort(ts)
    return ts[o], pr[o], gt[o]


def causal_signals_sparse(H):
    sig = []
    for sym in ["ETH", "BTC"]:
        ts, pr, gt = causal_signals(sym, H)          # 因果滚动阈值
        idx = greedy_sparse(ts.astype(np.int64), H)  # 间距=horizon分钟去重
        sig.append((ts[idx], pr[idx], gt[idx]))
    ts = np.concatenate([s[0] for s in sig]); pr = np.concatenate([s[1] for s in sig])
    gt = np.concatenate([s[2] for s in sig]); o = np.argsort(ts)
    return ts[o], pr[o], gt[o]


def main():
    print(f"{'周期':<4}{'当日前视胜率':>12}{'因果胜率':>10}{'因果虚高':>9}"
          f"{'固定5U ROI(因果)':>18}{'因果5U回撤':>12}")
    rows = {}
    for H in HORIZONS:
        lt, lp, lg = lookahead_signals(H)
        ct, cp, cg = causal_signals_sparse(H)
        _, sl = flat_sim(lt[lt >= lt.min()], lp, lg)
        _, sc = flat_sim(ct, cp, cg)
        gap = sl["acc"] - sc["acc"]
        print(f"{H:>2}分钟{sl['acc']*100:>10.2f}%{sc['acc']*100:>9.2f}%{gap*100:>8.2f}pp"
              f"{sc['roi']*100:>15.1f}%{sc['max_dd_pct']*100:>11.1f}%")
        rows[H] = {"look_acc": sl["acc"], "causal_acc": sc["acc"],
                   "causal_trades": int(sc["trades"]), "causal_roi": sc["roi"],
                   "causal_dd": sc["max_dd_pct"], "causal_bpd": sc["trades_per_day"]}
    json.dump(rows, open(f"{config.RESULT_DIR}/h_all_causal.json", "w"), indent=1)
    print("\n# 先定位因果胜率(去重后)最高的周期:")
    best = max(HORIZONS, key=lambda h: rows[h]["causal_acc"])
    print(f"  best horizon = {best}分钟 胜率={rows[best]['causal_acc']*100:.2f}%")

    # 20%回撤档 复利扫描(仅做最优周期)
    ct, cp, cg = causal_signals_sparse(best)
    print(f"\n# {best}分钟 复利回撤档扫描:")
    for f in [0.002, 0.003, 0.004, 0.005, 0.006]:
        _, sc = comp_sim(ct, cp, cg, 5000.0, f, float("inf"))
        print(f"   f={f:.3f}: 终值={sc['final']:.0f}U ROI={sc['roi']*100:+.0f}% "
              f"回撤={sc['max_dd_pct']*100:.1f}% 连亏={sc['max_consec_loss']}")


if __name__ == "__main__":
    main()