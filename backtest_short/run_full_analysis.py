#!/usr/bin/env python3
"""管线终步: 统一分析脚本. 所有结果落盘到 /workspace/results/"""
import os, sys, json
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext

HORIZONS = [3, 5, 10, 15]
FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]

def load_P(sym, h, split):
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{sym}_h{h}_{f}_{split}_P.npy")
          for f in FAMILIES]
    return np.stack(Ps, axis=0).astype(np.float64)

def ens_p(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)

def analyze_one_horizon(H):
    """单周期: 每日top1% vs 修正版因果阈值, 对比信号数/准确率."""
    from backtest_short.causal_signals_fixed import causal_signals_fixed
    results = {}
    all_entries = []
    for sym in ["ETH", "BTC"]:
        ctx = AssetContext(sym, horizon=H, ds_name=f"ds_{sym}_h{H}")
        split = "test"
        n = ctx.split_rows[split].sum()
        p = ens_p(load_P(sym, H, split))
        y = ctx.y(split)
        sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(p - 0.5) * 2
        pred = (p >= 0.5).astype(np.int8)

        # 每日 top 1%
        days = np.unique(sec // 86400)
        sel = np.zeros(len(sec), dtype=bool)
        for d in days:
            m = sec // 86400 == d
            k = max(1, int(np.ceil(int(m.sum()) * 0.01)))
            sub = np.where(m)[0]
            sel[sub[np.argsort(-conf[sub])[:k]]] = True
        top1_acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0
        top1_count = int(sel.sum())

        # 修正版因果
        ts, pr, gt = causal_signals_fixed(sym, H)
        causal_count = len(ts)
        causal_acc = float((pr == gt).mean()) if len(pr) else 0

        results[sym] = {"top1_n": top1_count, "top1_acc": top1_acc,
                       "causal_n": causal_count, "causal_acc": causal_acc,
                       "days": int(len(days))}
        all_entries.append((ts, pr, gt, sym))

    # 合并 ETH+BTC 的信号数和准确率
    total_top1 = sum(r["top1_n"] for r in results.values())
    total_causal = sum(r["causal_n"] for r in results.values())
    days = results["ETH"]["days"]
    return {
        "horizon": H,
        "top1_total": total_top1, "top1_per_day": total_top1 / max(days, 1),
        "top1_acc": np.mean([r["top1_acc"] for r in results.values()]),
        "causal_total": total_causal, "causal_per_day": total_causal / max(days, 1),
        "causal_acc": np.mean([r["causal_acc"] for r in results.values()]),
        "top1_vs_causal_ratio": total_causal / max(total_top1, 1),
        "results_by_symbol": results,
    }

def corr_analysis():
    """四周期相关性 - 用修正版因果信号."""
    from backtest_short.causal_signals_fixed import causal_signals_fixed
    from live.signals import greedy_sparse
    data = {}
    for H in HORIZONS:
        all_ts, all_dr = [], []
        for sym in ["ETH", "BTC"]:
            ts, pr, _ = causal_signals_fixed(sym, H)
            idx = greedy_sparse(ts.astype(np.int64), H)
            all_ts.append(ts[idx]); all_dr.append(pr[idx])
        ts = np.concatenate(all_ts); dr = np.concatenate(all_dr)
        o = np.argsort(ts); ts, dr = ts[o], dr[o]
        keep = np.concatenate([[True], np.diff(ts) > 0])
        ts, dr = ts[keep], dr[keep]
        t0, t1 = ts.min(), ts.max()
        grid = np.arange(t0, t1 + 60, 60)
        pos = np.searchsorted(grid, ts)
        series = np.zeros(len(grid), np.int8)
        series[pos] = np.where(dr >= 0.5, 1, -1)
        data[H] = (grid, series, ts, dr)

    n = len(HORIZONS)
    corr = np.zeros((n, n))
    agree = np.zeros((n, n))
    overlap = np.zeros((n, n))
    for i, H1 in enumerate(HORIZONS):
        for j, H2 in enumerate(HORIZONS):
            gi, si = data[H1][0], data[H1][1]
            gj, sj = data[H2][0], data[H2][1]
            lo = max(gi.min(), gj.min()); hi = min(gi.max(), gj.max())
            mi = (gi >= lo) & (gi <= hi); mj = (gj >= lo) & (gj <= hi)
            a, b = si[mi].astype(np.float64), sj[mj].astype(np.float64)
            if a.std() > 0 and b.std() > 0:
                corr[i, j] = float(np.corrcoef(a, b)[0, 1])
            both = (a != 0) & (b != 0)
            if both.sum() > 0:
                agree[i, j] = float((a[both] == b[both]).mean())
                overlap[i, j] = float(both.sum()) / max((a != 0).sum(), 1)
    return {"corr": corr.tolist(), "agree_pct": (agree * 100).tolist(),
            "overlap_pct": (overlap * 100).tolist(),
            "signal_counts": {H: len(data[H][2]) for H in HORIZONS}}

def main():
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    summary = {"horizon_analysis": [], "correlation": None}

    print("=" * 60)
    print("修正版因果信号管线 - 完整分析")
    print("=" * 60)

    # Part 1: 各周期信号数 + 准确率
    print("\n[Part 1] 各周期信号密度 + 准确率对比")
    print(f"{'horizon':>8} | {'每日top1%':>10} | {'因果(修正)':>10} | "
          f"{'top1%准确率':>12} | {'因果准确率':>12} | {'因果/top1%':>10}")
    print("-" * 75)
    for H in HORIZONS:
        r = analyze_one_horizon(H)
        summary["horizon_analysis"].append(r)
        print(f"{H:>5}min | {r['top1_per_day']:>10.1f} | {r['causal_per_day']:>10.1f} | "
              f"{r['top1_acc']*100:>11.2f}% | {r['causal_acc']*100:>11.2f}% | "
              f"{r['top1_vs_causal_ratio']*100:>9.0f}%")

    # Part 2: 相关性
    print("\n[Part 2] 四周期信号相关性 (修正版因果阈值)")
    corr = corr_analysis()
    summary["correlation"] = corr
    Hs = HORIZONS
    print("\nPearson 相关矩阵 (1min 网格, 含无信号分钟):")
    print(f"{'':>8}" + "".join(f"{h:>8}min" for h in Hs))
    for i in range(len(Hs)):
        print(f"{Hs[i]:>5}min" + "".join(f"{corr['corr'][i][j]:>8.3f}" for j in range(len(Hs))))
    print("\n方向一致率 (双方同分钟都有信号时):")
    print(f"{'':>8}" + "".join(f"{h:>8}min" for h in Hs))
    for i in range(len(Hs)):
        print(f"{Hs[i]:>5}min" + "".join(f"{corr['agree_pct'][i][j]:>8.1f}%" for j in range(len(Hs))))

    print("\n各周期信号数(修正因果): " + ", ".join(f"{H}min={corr['signal_counts'][H]}" for H in Hs))

    # 保存结果
    out = os.path.join(config.RESULT_DIR, "full_analysis_result.json")
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")

if __name__ == "__main__":
    main()
