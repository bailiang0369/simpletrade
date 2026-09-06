#!/usr/bin/env python3
"""无泄露阈值优化: 在 meta_val 上校准并扫描阈值, 用选出的阈值应用到 test。

流程(严格样本外):
  1. 对每个 (symbol, tag): 载入 15 模型 rank-mean 融合概率 p。
  2. 校准: meta_val 上拟合保序回归 (isotonic), 得到校准概率 q。
  3. meta_val 上扫描分位数 q ∈ {95..99.9}: 选中 q ≥ τ_q 的样本,
     评估 总acc/月度下限/bad/覆盖率。选择: bad=0 优先, 覆盖率∈[0.8,1.2]%, 再月度下限最高。
  4. test 应用: 固定 τ_q (校准后概率阈值), 报告。
对照: 同时报告 未校准固定阈值(R1) 与 每日滚动99分位(R2)。
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
QS = [95.0, 96.0, 97.0, 98.0, 99.0, 99.5, 99.9]


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, tag, split):
    Ps = []
    for f in FAMS:
        path = f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy" if tag == "JOINT" \
            else f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy"
        Ps.append(np.load(path))
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def eval_threshold(p, y, sec_arr, tau):
    pred = (p >= 0.5).astype(np.int8)
    sel = np.where(p >= tau)[0]
    mts_all = sec_arr.astype("datetime64[s]").astype("datetime64[M]")
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0, 0.0)
    ps, ys, ms = pred[sel], y[sel], mts_all[sel]
    acc = float((ps == ys).mean())
    uniq = np.unique(ms)
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in uniq}
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    cov = float(len(ps)) / len(sec_arr)
    return (acc, min_a, nbad, cov, acc_m)


def pick_best(results):
    """results: list of (q, acc, min_a, nbad, cov, acc_m). 优先 bad=0, 再覆盖率接近1%, 再 min_a。"""
    good = [r for r in results if r[2] == 0 and 0.008 <= r[3] <= 0.012]
    pool = good if good else [r for r in results if r[2] == 0]
    if not pool:
        pool = results
    pool.sort(key=lambda r: (r[1] if r[1] == r[1] else -1, r[3] != 1, -abs(r[3] - 0.01)), reverse=False)
    return pool[-1]


def main():
    from sklearn.isotonic import IsotonicRegression
    configs = [("ETH", ""), ("BTC", ""), ("ETH", "JOINT"), ("BTC", "JOINT")]
    for symbol, tag in configs:
        label = tag or "SOLO"
        ctx = AssetContext(symbol, horizon=30)
        mv = load_fused(symbol, tag, "meta_val")
        te = load_fused(symbol, tag, "test")
        y_mv = ctx.y("meta_val"); y_te = ctx.y("test")
        sec_mv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
        sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

        # 保序校准: meta_val 拟合
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(mv, y_mv.astype(np.float64))
        q_mv = ir.predict(mv); q_te = ir.predict(te)

        # meta_val 扫描
        results = []
        for qq in QS:
            tau = np.percentile(q_mv, qq)
            acc, min_a, nbad, cov, acc_m = eval_threshold(q_mv, y_mv, sec_mv, tau)
            results.append((qq, acc, min_a, nbad, cov, acc_m))
        best = pick_best(results)
        q_best, acc_b, min_b, bad_b, cov_b, _ = best
        print(f"\n===== {label} {symbol} (校准后 meta_val 扫描) =====")
        print("  q  ->  mv: acc/min_month/bad/cov")
        for r in results:
            print(f"  q={r[0]:4.1f}  mv(acc={r[1]:.4f}, min={r[2]:.4f}, bad={r[3]}, cov={r[4]:.4f})")
        print(f"  >> 选择 q={q_best}: mv acc={acc_b:.4f} min={min_b:.4f} bad={bad_b} cov={cov_b:.4f}")

        # test 应用
        tau_t = np.percentile(q_mv, q_best)
        acc, min_a, nbad, cov, acc_m = eval_threshold(q_te, y_te, sec_te, tau_t)
        print(f"  [test τ=q{q_best}] acc={acc:.4f} min_month={min_a:.4f} bad={nbad} cov={cov:.4f}")
        print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
        # 对照: 未校准固定 99 分位
        tau_raw = np.percentile(mv, 99.0)
        acc, min_a, nbad, cov, acc_m = eval_threshold(te, y_te, sec_te, tau_raw)
        print(f"  [test raw-99 ] acc={acc:.4f} min_month={min_a:.4f} bad={nbad} cov={cov:.4f}")
        del mv, te, q_mv, q_te, ir
        gc.collect()


if __name__ == "__main__":
    main()
