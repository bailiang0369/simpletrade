"""综合评估: 无前视 + 跨 horizon stacking + 两币 gate + 结果汇总。"""
import os, sys, gc, time
sys.path.insert(0, "/workspace")
import numpy as np
import lightgbm as lgb
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


def eval_nolookahead_arrays(sym, horizon, split, ens_arrays, q=99.0, win_days=30, cold=30):
    """核心无前视评估: 给定 ensemble arrays, 返回 (acc, tpd, n, auc)。"""
    ctx = AssetContext(sym, horizon=horizon)
    mask = ctx.split_rows[split]
    y = ctx.label[mask]
    times = ctx.times(split)
    times_s = np.asarray(times).astype("datetime64[s]").astype(np.int64)

    p = rank_ens(ens_arrays) if len(ens_arrays) > 1 else ens_arrays[0].copy()
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)

    day_of = times_s // 86400
    days = np.unique(day_of)
    n = len(p)
    sel = np.zeros(n, dtype=bool)

    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di - win_days):di]
        if len(prior) < cold:
            continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0:
                continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True

    acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(days), 1)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return acc * 100, tpd, int(sel.sum()), auc


def eval_curve(sym, horizon, split, ens_arrays, qlist):
    return [(q, *eval_nolookahead_arrays(sym, horizon, split, ens_arrays, q=q)) for q in qlist]


# ================================================================
# 方向 A: 单 horizon 无前视评估
# ================================================================
def run_single_horizon_eval():
    print("=" * 70)
    print("方向 A: 单 HORIZON 无前视评估 (10-seed rank ens)")
    print("=" * 70)
    QLIST = [97.0, 98.0, 98.5, 99.0, 99.2, 99.5, 99.8]
    results = {}
    for sym in ["ETH", "BTC"]:
        for h in [5, 15, 30]:
            key = f"{sym}_h{h}"
            results[key] = {}
            print(f"\n--- {sym} H={h} ---")
            for split in ["meta_val", "test"]:
                arrs = load_P(sym, h, split)
                if not arrs:
                    print(f"  {split}: 无模型")
                    continue
                curve = eval_curve(sym, h, split, arrs, QLIST)
                results[key][split] = curve
                auc = curve[0][4]
                print(f"  {split}  AUC={auc:.4f}")
                print(f"  {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
                for q, acc, tpd, n, _ in curve:
                    mark = " ★" if acc >= 64 and tpd >= 11 else ""
                    print(f"  {q:6.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")
    return results


# ================================================================
# 方向 B: 跨 horizon stacking
# ================================================================
def run_cross_horizon_stack():
    print("\n" + "=" * 70)
    print("方向 B: 跨 HORIZON STACKING — H5+H15+H30 → meta LGB")
    print("=" * 70)

    horizons = [5, 15, 30]
    out = {}

    for sym in ["ETH", "BTC"]:
        print(f"\n### {sym}")
        out[sym] = {}

        def build_X(split):
            feats = []
            for h in horizons:
                arrs = load_P(sym, h, split)
                if not arrs:
                    return None, None, None
                p = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
                conf = np.abs(p - 0.5) * 2
                feats.append(p)
                feats.append(conf)
            X = np.stack(feats, axis=1)
            ctx15 = AssetContext(sym, horizon=15)
            mask = ctx15.split_rows[split]
            y = ctx15.label[mask].astype(np.float64)
            w = np.clip(np.abs(ctx15.retf(split)) * 50, 0.5, 5.0)
            return X, y, w

        Xmv, ymv, wmv = build_X("meta_val")
        if Xmv is None:
            print("  meta_val 数据缺失")
            continue

        n = len(ymv)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[int(n * 0.7):] = True

        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 100, "verbose": -1,
        }
        tr_ds = lgb.Dataset(Xmv[~val_mask], label=ymv[~val_mask], weight=wmv[~val_mask])
        es_ds = lgb.Dataset(Xmv[val_mask], label=ymv[val_mask], reference=tr_ds)
        m = lgb.train(params, tr_ds, num_boost_round=1000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        print(f"  meta LGB: iter={m.best_iteration} auc={m.best_score['valid_0']['auc']:.4f}")

        for split in ["meta_val", "test"]:
            Xs, ys, ws = build_X(split)
            if Xs is None:
                continue
            P_meta = m.predict(Xs)
            out[sym][split] = P_meta
            ctx = AssetContext(sym, horizon=15)
            mask = ctx.split_rows[split]
            y = ctx.label[mask]
            try:
                auc = roc_auc_score(y, P_meta)
            except Exception:
                auc = float("nan")
            print(f"\n  {split} stacking AUC={auc:.4f}")

            # 无前视
            p_mock = P_meta  # 已经是单 array
            acc, tpd, n_sel, _ = eval_nolookahead_arrays(sym, 15, split, [p_mock], q=99.0)
            print(f"    q=99.0 → acc={acc:.2f}% tpd={tpd:.1f} n={n_sel}")

            QLIST = [97.0, 98.0, 98.5, 99.0, 99.2, 99.5]
            print(f"    {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
            for q in QLIST:
                a, t, n_s, _ = eval_nolookahead_arrays(sym, 15, split, [P_meta], q=q)
                mark = " ★" if a >= 64 and t >= 11 else ""
                print(f"    {q:6.1f} {a:7.2f}% {t:7.1f} {n_s:8d}{mark}")
    return out


# ================================================================
# 方向 D: ETH∩BTC 两币 gate (同时强信号才下注)
# ================================================================
def run_cross_coin_gate():
    print("\n" + "=" * 70)
    print("方向 D: 两币 GATE — ETH 和 BTC 同时强信号才下注")
    print("=" * 70)
    # 思路: 用 h15 最好的 ens, ETH top-q1 且 BTC top-q2 同时满足 → 下注
    for split in ["meta_val", "test"]:
        print(f"\n--- {split} ---")
        eth_arrs = load_P("ETH", 15, split)
        btc_arrs = load_P("BTC", 15, split)
        if not eth_arrs or not btc_arrs:
            print("  缺数据")
            continue
        p_eth = rank_ens(eth_arrs)
        p_btc = rank_ens(btc_arrs)

        # 两币时间戳 (应该完全对齐, 但 horizon 不同可能有 N 差异)
        ctx_eth = AssetContext("ETH", horizon=15)
        ctx_btc = AssetContext("BTC", horizon=15)
        mask_e = ctx_eth.split_rows[split]
        mask_b = ctx_btc.split_rows[split]
        y_e = ctx_eth.label[mask_e]
        y_b = ctx_btc.label[mask_b]
        ts_e = np.asarray(ctx_eth.times(split)).astype("datetime64[s]").astype(np.int64)
        ts_b = np.asarray(ctx_btc.times(split)).astype("datetime64[s]").astype(np.int64)

        # 对齐: 取交集时间戳 (取 min 长度)
        n_min = min(len(p_eth), len(p_btc))
        p_e, y_e2, te = p_eth[:n_min], y_e[:n_min], ts_e[:n_min]
        p_b, y_b2, tb = p_btc[:n_min], y_b[:n_min], ts_b[:n_min]
        assert (te == tb).all(), "时间戳不对齐"

        conf_e = np.abs(p_e - 0.5) * 2
        conf_b = np.abs(p_b - 0.5) * 2
        pred_e = (p_e >= 0.5).astype(np.int8)
        pred_b = (p_b >= 0.5).astype(np.int8)

        day_of = te // 86400
        days = np.unique(day_of)

        # 扫 qe, qb
        print(f"  {'qe':>5} {'qb':>5} {'ETH acc':>9} {'BTC acc':>9} {'combined acc':>14} {'tpd':>6} {'n':>8}")
        for qe in [97, 98, 98.5, 99, 99.5]:
            for qb in [97, 98, 98.5, 99, 99.5]:
                sel_e = np.zeros(n_min, dtype=bool)
                sel_b = np.zeros(n_min, dtype=bool)
                for di, d in enumerate(days.astype(int).tolist()):
                    prior = days.astype(int).tolist()[max(0, di - 30):di]
                    if len(prior) < 30:
                        continue
                    today = day_of == d
                    for side in [1, 0]:
                        m = today & (pred_e == side)
                        hist = np.isin(day_of, prior) & (pred_e == side)
                        if hist.sum() == 0 or m.sum() == 0: continue
                        tau = float(np.percentile(conf_e[hist], qe))
                        sel_e[m & (conf_e >= tau)] = True
                        m2 = today & (pred_b == side)
                        hist2 = np.isin(day_of, prior) & (pred_b == side)
                        if hist2.sum() == 0 or m2.sum() == 0: continue
                        tau2 = float(np.percentile(conf_b[hist2], qb))
                        sel_b[m2 & (conf_b >= tau2)] = True

                # 交集: ETH 和 BTC 在同一方向都强信号才下注 ETH
                gate = sel_e & sel_b
                # 方向必须一致
                same_dir = pred_e == pred_b
                final = gate & same_dir & sel_e  # 下注 ETH 的方向

                n_final = final.sum()
                if n_final == 0:
                    continue
                tpd = n_final / max(len(days), 1)
                acc_eth_e = (pred_e[sel_e] == y_e2[sel_e]).mean() * 100 if sel_e.sum() else 0
                acc_btc_b = (pred_b[sel_b] == y_b2[sel_b]).mean() * 100 if sel_b.sum() else 0
                acc_gate = (pred_e[final] == y_e2[final]).mean() * 100

                mark = " ★" if acc_gate >= 66 and tpd >= 8 else ""
                print(f"  {qe:5.1f} {qb:5.1f} {acc_eth_e:8.2f}% {acc_btc_b:8.2f}% {acc_gate:13.2f}% {tpd:5.1f} {n_final:8d}{mark}")


# ================================================================
# 方向 E: ETH h15 基线 更多 seed (30-seed LGB)
# ================================================================
def run_more_seeds():
    print("\n" + "=" * 70)
    print("方向 E: ETH h15 更多 seed 评估 (现有 10 seed)")
    print("=" * 70)
    for sym in ["ETH", "BTC"]:
        for split in ["meta_val", "test"]:
            arrs = load_P(sym, 15, split)
            if not arrs:
                continue
            print(f"\n{sym} h15 {split}: AUC={roc_auc_score(AssetContext(sym,15).label[AssetContext(sym,15).split_rows[split]], rank_ens(arrs)):.4f}")
            # 单 seed AUC
            for s, a in zip(SEEDS, arrs):
                try:
                    auc_s = roc_auc_score(AssetContext(sym,15).label[AssetContext(sym,15).split_rows[split]], a)
                except:
                    auc_s = float("nan")
                print(f"  seed{s}: auc={auc_s:.4f}")


def main():
    # 1. 方向 A: 各 horizon 单独评估
    single = run_single_horizon_eval()

    # 2. 方向 B: 跨 horizon stacking
    stack = run_cross_horizon_stack()

    # 3. 方向 D: 两币 gate
    cross = run_cross_coin_gate()

    # 4. 方向 E: 更多 seed 分析
    more = run_more_seeds()

    print("\n" + "=" * 70)
    print("✅ 全部评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
