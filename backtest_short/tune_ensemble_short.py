#!/usr/bin/env python3
"""短周期集成权重调优: 用 meta_val 搜索最优集成权重, 把每日 top1% 准确率推向 62%。

常见的短周期瓶颈: 15 个模型(3 family × 5 seed)等权 rank-mean 集成的可区分度有限。
本脚本在 meta_val 上对 family 级权重与 seed 内权重做搜索, 使每日 top1% 选样准确率最大,
再用 test 验证是否稳健提升(避免对 meta_val 过拟合)。

用法:
  python -m backtest_short.tune_ensemble_short 5 ETH
  python -m backtest_short.tune_ensemble_short 3 ETH
  python -m backtest_short.tune_ensemble_short 5 BTC
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]


def load_P(ctx, symbol, horizon, split):
    """返回 (15, n) 的原始预测 logits/prob 矩阵, 按 family 分块。"""
    P = np.concatenate([
        np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
        for f in FAMILIES
    ], axis=0).astype(np.float64)  # (15, n)
    return P


def global_ranks(P):
    n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def daily_topk_acc(p, y, sec_arr, frac=0.01):
    """复刻 evaluate() 的 [daily] 逻辑: 每日取 top1% 最高置信度样本, 计算方向准确率。"""
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    day = sec_arr // 86400
    days = np.unique(day)
    n = len(p); sel = np.zeros(n, bool)
    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * frac)))
        sub = np.where(md)[0]
        sel[sub[np.argsort(-conf[sub])[:kd]]] = True
    s = np.where(sel)[0]
    return float((pred[s] == y[s]).mean()), s


def combine(R, w, n_models_per_fam=5):
    """权重 w 长度为 family 数或模型数; family 级权重广播到每个 seed。"""
    if len(w) == len(FAMILIES):
        w_full = np.repeat(np.asarray(w, np.float64), n_models_per_fam)
    else:
        w_full = np.asarray(w, np.float64)
    w_full = w_full / w_full.sum()
    return (w_full[:, None] * R).sum(axis=0)


def report(name, p, y, sec_arr):
    acc, _ = daily_topk_acc(p, y, sec_arr)
    print(f"  {name:<24} daily_top1%_acc = {acc:.4f}", flush=True)


def main():
    horizon = int(sys.argv[1])
    symbol = sys.argv[2].upper()
    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")

    R_mv = global_ranks(load_P(ctx, symbol, horizon, "meta_val"))
    R_te = global_ranks(load_P(ctx, symbol, horizon, "test"))
    ym, ym_t = ctx.y("meta_val"), ctx.y("test")
    sm = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    st = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    print(f"\n===== SHORT h{horizon} {symbol} 集成权重调优 =====", flush=True)

    # 基线: 各 family 独立 与 全等权
    nb = len(FAMILIES)
    for i, f in enumerate(FAMILIES):
        r_i = R_mv[i*nb:(i+1)*nb].mean(axis=0)
        r_t = R_te[i*nb:(i+1)*nb].mean(axis=0)
        report(f"family={f} (meta_val)", r_i, ym, sm)
        report(f"family={f} (test)    ", r_t, ym_t, st)
    eq = R_mv.mean(axis=0); eq_t = R_te.mean(axis=0)
    report("equal-rank-mean (meta_val)", eq, ym, sm)
    report("equal-rank-mean (test)    ", eq_t, ym_t, st)

    # ---- family 级权重网格搜索 (simplex, step=0.05) ----
    best = {"acc": -1.0, "w": None}
    step = 0.05
    for wa in np.arange(0, 1.0001, step):
        for wb in np.arange(0, 1.0001 - wa, step):
            wc = 1.0 - wa - wb
            w = [wa, wb, wc]
            p = combine(R_mv, w)
            acc, _ = daily_topk_acc(p, ym, sm)
            if acc > best["acc"]:
                best = {"acc": acc, "w": w}
    bw = best["w"]
    p = combine(R_mv, bw); report(f"best-fam-wt {np.round(bw,2)} (mv)", p, ym, sm)
    p_t = combine(R_te, bw)
    report(f"best-fam-wt (test)        ", p_t, ym_t, st)
    # test 逐月明细 + global(x1%) for chosen config
    acc, sel = daily_topk_acc(p_t, ym_t, st)
    mts = st[sel].astype("datetime64[s]").astype("datetime64[M]")
    acc_m = {}
    uniq = np.unique(mts)
    pred = (p_t >= 0.5).astype(np.int8)
    for u in uniq:
        acc_m[str(u)[:7]] = float((pred[sel] == ym_t[sel])[mts == u].mean())
    print(f"  [best-fam-wt test] daily acc={acc:.4f} cov={len(sel)/len(ym_t):.4f}", flush=True)
    print("    逐月:", {k: round(v, 3) for k, v in acc_m.items()}, flush=True)
    k = max(1, int(round(len(p_t) * 0.01)))
    selt = np.argsort(-np.maximum(p_t, 1 - p_t))[:k]
    greedy = float((pred[selt] == ym_t[selt]).mean())
    print(f"  [best-fam-wt test] global_top1 acc={greedy:.4f}", flush=True)

    # ---- 完整 15 权重用 scipy 优化(meta_val), 在 test 上检查稳健性 ----
    try:
        from scipy.optimize import minimize

        def neg_acc(w):
            wc = np.asarray(w, np.float64) + 1e-3
            wc = wc / wc.sum()
            p = combine(R_mv, wc)
            acc, _ = daily_topk_acc(p, ym, sm)
            return -acc

        w0 = np.ones(R_mv.shape[0]) / R_mv.shape[0]
        res = minimize(neg_acc, w0, method="L-BFGS-B",
                       bounds=[(0, None)] * R_mv.shape[0], options={"maxiter": 800})
        wfull = np.asarray(res.x) + 1e-3
        wfull = wfull / wfull.sum()
        p = combine(R_mv, wfull)
        report(f"opt15 (mv, acc={-res.fun:.4f})", p, ym, sm)
        p_t = combine(R_te, wfull)
        report("opt15 (test)", p_t, ym_t, st)
    except ImportError:
        print("  (scipy 不可用, 跳过 15 权重优化)", flush=True)

    # 持久化选中的 family 权重 (供实盘/模拟盘推理复用)
    import json
    w_path = f"/workspace/models_saved/pool_short_h{horizon}/ens_family_w.json"
    d = {}
    if os.path.exists(w_path):
        d = json.load(open(w_path))
    d[symbol] = {"w": bw, "daily_test_acc": best["acc"] if False else None}
    with open(w_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"    保存权重 -> {w_path}  {symbol}: {np.round(bw,2)}", flush=True)

    print(f"\n===== SHORT h{horizon} {symbol} 调优完成 =====", flush=True)


if __name__ == "__main__":
    main()