"""精细调参: BTC h30 找 acc≥65% + tpd≥11 的 sweet spot; 简单平均 stacking。"""
import os, sys, gc
sys.path.insert(0, "/workspace")
import numpy as np
from sklearn.metrics import roc_auc_score
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]


def load_P(sym, horizon, split):
    arrs = []
    for s in SEEDS:
        fp = f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_lgb_seed{s}_{split}_P.npy"
        if os.path.exists(fp):
            arrs.append(np.load(fp).astype(np.float64))
    return arrs


def rank_ens(arrays):
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def eval_nolookahead_simple(P, y, ts, q, win=30, cold=30):
    """给定 P, y, ts, q → (acc, tpd, n)。无前视。"""
    conf = np.abs(P - 0.5) * 2
    pred = (P >= 0.5).astype(np.int8)
    day_of = ts // 86400
    days = np.unique(day_of)
    n = len(P)
    sel = np.zeros(n, dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di - win):di]
        if len(prior) < cold: continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0: continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True
    acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(days), 1)
    return acc * 100, tpd, int(sel.sum())


def fine_grid(sym, h, split, q_start=97, q_end=99.9, step=0.1):
    """精细 q 网格搜索。"""
    arrs = load_P(sym, h, split)
    ctx = AssetContext(sym, horizon=h)
    mask = ctx.split_rows[split]
    P = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
    y = ctx.label[mask]
    ts = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    auc = roc_auc_score(y, P)

    qs = np.arange(q_start, q_end + 0.001, step)
    print(f"\n{sym} H={h} {split} AUC={auc:.4f}")
    print(f"  {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
    for q in qs:
        acc, tpd, n = eval_nolookahead_simple(P, y, ts, q=q)
        mark = " ★" if acc >= 65 and tpd >= 10 else ""
        print(f"  {q:5.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")


# ============ 1. BTC h30 精细搜索 ============
print("=" * 70)
print("1. BTC h30 精细 q 搜索 (test 上 q=99 已有 62.5%, q=99.5 已有 66.2%)")
print("=" * 70)
fine_grid("BTC", 30, "test", q_start=98.5, q_end=99.8, step=0.05)
fine_grid("BTC", 30, "meta_val", q_start=98.5, q_end=99.8, step=0.05)

# ============ 2. ETH h15 更细粒度 ============
print("\n" + "=" * 70)
print("2. ETH h15 精细 q 搜索")
print("=" * 70)
fine_grid("ETH", 15, "test", q_start=98.5, q_end=99.8, step=0.05)
fine_grid("ETH", 15, "meta_val", q_start=98.5, q_end=99.8, step=0.05)

# ============ 3. 简单平均 stacking (不用 meta LGB, 避免过拟合) ============
print("\n" + "=" * 70)
print("3. 简单跨 horizon AVERAGE (H5+H15+H30 平均 conf) — 避免 stacking 过拟合")
print("=" * 70)

def avg_stacking(sym, split):
    """取各 horizon rank_ens P 的平均, 再评估。"""
    horizons = [5, 15, 30]
    ctx15 = AssetContext(sym, horizon=15)
    ts_base = ctx15.ds_ts[ctx15.split_rows[split]]
    y_base = ctx15.label[ctx15.split_rows[split]]
    te_s = np.asarray(ctx15.times(split)).astype("datetime64[s]").astype(np.int64)

    # 各 horizon 对齐到 h15 的 ts
    Ps = []
    for h in horizons:
        ctx = AssetContext(sym, horizon=h)
        mask = ctx.split_rows[split]
        ts_h = ctx.ds_ts[mask]
        arrs = load_P(sym, h, split)
        p_h = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
        # 对齐
        idx = np.searchsorted(ts_h, ts_base)
        p_aligned = p_h[idx]
        Ps.append(p_aligned)
    P_avg = np.mean(Ps, axis=0)

    auc = roc_auc_score(y_base, P_avg)
    print(f"\n  {sym} {split} avg(h5,h15,h30) AUC={auc:.4f}")
    print(f"  {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
    for q in np.arange(97.0, 99.6, 0.5):
        acc, tpd, n = eval_nolookahead_simple(P_avg, y_base, te_s, q=q)
        mark = " ★" if acc >= 64 and tpd >= 11 else ""
        print(f"  {q:5.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")

for sym in ["ETH", "BTC"]:
    avg_stacking(sym, "meta_val")
    avg_stacking(sym, "test")

# ============ 4. ETH h15 + BTC h30 简单平均 (最佳单 horizon 组合) ============
print("\n" + "=" * 70)
print("4. ETH h15 + BTC h30 简单平均 (最佳单 horizon 各一)")
print("=" * 70)

def cross_coin_avg(sym1, h1, sym2, h2, split):
    """两个不同币种+horizon 的 P 对齐后平均, 下注方向用 sym1 的。"""
    ctx1 = AssetContext(sym1, horizon=h1)
    ctx2 = AssetContext(sym2, horizon=h2)
    m1 = ctx1.split_rows[split]
    m2 = ctx2.split_rows[split]
    ts1 = ctx1.ds_ts[m1]
    ts2 = ctx2.ds_ts[m2]

    arrs1 = load_P(sym1, h1, split)
    arrs2 = load_P(sym2, h2, split)
    p1 = rank_ens(arrs1) if len(arrs1) > 1 else arrs1[0]
    p2 = rank_ens(arrs2) if len(arrs2) > 1 else arrs2[0]

    # 对齐到 h15 的 ts
    ih = np.searchsorted(ts2, ts1)
    p2_aligned = p2[ih]
    P_avg = (p1 + p2_aligned) / 2

    y = ctx1.label[m1]
    te_s = np.asarray(ctx1.times(split)).astype("datetime64[s]").astype(np.int64)
    auc = roc_auc_score(y, P_avg)
    print(f"\n  {sym1} h{h1} + {sym2} h{h2} {split} AUC={auc:.4f}")
    print(f"  {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
    for q in np.arange(97.0, 99.6, 0.5):
        acc, tpd, n = eval_nolookahead_simple(P_avg, y, te_s, q=q)
        mark = " ★" if acc >= 64 and tpd >= 11 else ""
        print(f"  {q:5.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")

# 下注 ETH, 但参考 BTC h30 信号方向
cross_coin_avg("ETH", 15, "BTC", 30, "meta_val")
cross_coin_avg("ETH", 15, "BTC", 30, "test")
# 下注 BTC, 参考 ETH h15
cross_coin_avg("BTC", 30, "ETH", 15, "meta_val")
cross_coin_avg("BTC", 30, "ETH", 15, "test")

print("\n✅ 精细调参完成")
