"""综合评估 v2: 修复各 horizon 长度对齐问题。"""
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
    times_s = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)

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


# ================================================================
# 方向 B: 跨 horizon stacking (时间戳对齐)
# ================================================================
def run_cross_horizon_stack():
    print("=" * 70)
    print("方向 B: 跨 HORIZON STACKING — H5+H15+H30 → meta LGB")
    print("=" * 70)

    horizons = [5, 15, 30]

    for sym in ["ETH", "BTC"]:
        print(f"\n### {sym}")

        # 每个 horizon 提取 (ts, rank_ens_P, label) 然后按 ts 对齐
        def collect_h(split):
            all_data = {}
            for h in horizons:
                ctx = AssetContext(sym, horizon=h)
                mask = ctx.split_rows[split]
                ts_arr = ctx.ds_ts[mask]
                arrs = load_P(sym, h, split)
                if not arrs:
                    return None
                p = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
                y_h = ctx.label[mask]
                retf_h = ctx.retf(split)
                all_data[h] = {"ts": ts_arr, "p": p, "y": y_h, "retf": retf_h}
            return all_data

        # meta_val 对齐 → 训练
        all_mv = collect_h("meta_val")
        # 找到共同的 ts
        ts_common = all_mv[5]["ts"]
        for h in [15, 30]:
            mask_h = np.isin(all_mv[h]["ts"], ts_common)
            all_mv[h] = {k: v[mask_h] for k, v in all_mv[h].items()}
        # 再确认都是 same ts (h5 的 ts 应该都在 h15/h30 里)
        for h in horizons:
            assert len(all_mv[h]["ts"]) == len(ts_common), f"h{h} 长度不对"

        # 构建 meta 特征
        Xmv_parts = []
        for h in horizons:
            p_h = all_mv[h]["p"]
            c_h = np.abs(p_h - 0.5) * 2
            Xmv_parts.append(p_h)
            Xmv_parts.append(c_h)
        Xmv = np.stack(Xmv_parts, axis=1)
        ymv = all_mv[15]["y"].astype(np.float64)  # 目标: h15 方向
        wmv = np.clip(np.abs(all_mv[15]["retf"]) * 50, 0.5, 5.0)

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

        # meta_val 评估 (直接用训练好的 meta LGB)
        P_meta_mv = m.predict(Xmv)
        auc_meta_mv = roc_auc_score(ymv, P_meta_mv)
        print(f"\n  meta_val stacking AUC={auc_meta_mv:.4f}")
        _print_curve(P_meta_mv, all_mv[15]["y"], all_mv[15]["ts"], label_prefix="  ")

        # test 评估 (对齐后预测)
        all_ts = {}
        for h in horizons:
            ctx = AssetContext(sym, horizon=h)
            mask = ctx.split_rows["test"]
            ts_h = ctx.ds_ts[mask]
            arrs = load_P(sym, h, "test")
            p_h = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
            y_h = ctx.label[mask]
            retf_h = ctx.retf("test")
            all_ts[h] = {"ts": ts_h, "p": p_h, "y": y_h, "retf": retf_h}

        # 取交集 ts
        ts_sets = [set(all_ts[h]["ts"].tolist()) for h in horizons]
        ts_inter = sorted(ts_sets[0] & ts_sets[1] & ts_sets[2])
        ts_inter_arr = np.array(ts_inter, dtype=np.int64)
        print(f"\n  test 交集 ts: {len(ts_inter_arr)} rows")

        Xte_parts = []
        for h in horizons:
            ih = np.searchsorted(all_ts[h]["ts"], ts_inter_arr)
            p_h = all_ts[h]["p"][ih]
            c_h = np.abs(p_h - 0.5) * 2
            Xte_parts.append(p_h)
            Xte_parts.append(c_h)
        Xte = np.stack(Xte_parts, axis=1)
        P_meta_te = m.predict(Xte)

        ih15 = np.searchsorted(all_ts[15]["ts"], ts_inter_arr)
        yte = all_ts[15]["y"][ih15]
        auc_meta_te = roc_auc_score(yte, P_meta_te)
        print(f"  test stacking AUC={auc_meta_te:.4f}")
        _print_curve(P_meta_te, yte, ts_inter_arr, label_prefix="  ")


def _print_curve(P, y, ts, label_prefix=""):
    """无前视评估 + 打印曲线。"""
    conf = np.abs(P - 0.5) * 2
    pred = (P >= 0.5).astype(np.int8)
    day_of = ts // 86400
    days = np.unique(day_of)
    n = len(P)

    QLIST = [97.0, 98.0, 98.5, 99.0, 99.2, 99.5]
    print(f"{label_prefix} {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
    for q in QLIST:
        sel = np.zeros(n, dtype=bool)
        for di, d in enumerate(days.astype(int).tolist()):
            prior = days.astype(int).tolist()[max(0, di - 30):di]
            if len(prior) < 30:
                continue
            today = day_of == d
            for side in [1, 0]:
                mm = today & (pred == side)
                hist = np.isin(day_of, prior) & (pred == side)
                if hist.sum() == 0 or mm.sum() == 0:
                    continue
                tau = float(np.percentile(conf[hist], q))
                sel[mm & (conf >= tau)] = True
        acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0.0
        tpd = sel.sum() / max(len(days), 1)
        mark = " ★" if acc * 100 >= 64 and tpd >= 11 else ""
        print(f"{label_prefix} {q:6.1f} {acc*100:7.2f}% {tpd:7.1f} {sel.sum():8d}{mark}")


# ================================================================
# 方向 D: ETH∩BTC 两币 gate (同时强信号才下注)
# ================================================================
def run_cross_coin_gate():
    print("\n" + "=" * 70)
    print("方向 D: 两币 GATE — ETH 和 BTC 同时强信号才下注 ETH")
    print("=" * 70)
    for split in ["meta_val", "test"]:
        print(f"\n--- {split} ---")
        eth_arrs = load_P("ETH", 15, split)
        btc_arrs = load_P("BTC", 15, split)
        if not eth_arrs or not btc_arrs:
            print("  缺数据")
            continue
        p_eth = rank_ens(eth_arrs)
        p_btc = rank_ens(btc_arrs)

        ctx_eth = AssetContext("ETH", horizon=15)
        ctx_btc = AssetContext("BTC", horizon=15)
        mask_e = ctx_eth.split_rows[split]
        mask_b = ctx_btc.split_rows[split]

        ts_e = ctx_eth.ds_ts[mask_e]
        ts_b = ctx_btc.ds_ts[mask_b]
        y_e = ctx_eth.label[mask_e]
        y_b = ctx_btc.label[mask_b]
        te_s = np.asarray(ctx_eth.times(split)).astype("datetime64[s]").astype(np.int64)
        tb_s = np.asarray(ctx_btc.times(split)).astype("datetime64[s]").astype(np.int64)

        # 时间戳应该完全对齐 (同 horizon=15)
        assert (ts_e == ts_b).all(), "h15 时间戳不对齐"

        n = len(p_eth)
        conf_e = np.abs(p_eth - 0.5) * 2
        conf_b = np.abs(p_btc - 0.5) * 2
        pred_e = (p_eth >= 0.5).astype(np.int8)
        pred_b = (p_btc >= 0.5).astype(np.int8)

        day_of = te_s // 86400
        days = np.unique(day_of)

        print(f"  {'qe':>5} {'qb':>5} {'ETH acc':>9} {'BTC acc':>9} {'GATE acc':>11} {'tpd':>6} {'n':>8}")
        for qe in [97, 98, 98.5, 99, 99.5]:
            for qb in [97, 98, 98.5, 99, 99.5]:
                sel_e = np.zeros(n, dtype=bool)
                sel_b = np.zeros(n, dtype=bool)
                for di, d in enumerate(days.astype(int).tolist()):
                    prior = days.astype(int).tolist()[max(0, di - 30):di]
                    if len(prior) < 30: continue
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

                same_dir = pred_e == pred_b
                final = sel_e & sel_b & same_dir

                n_final = final.sum()
                if n_final < 10: continue
                tpd = n_final / max(len(days), 1)
                acc_gate = (pred_e[final] == y_e[final]).mean() * 100

                mark = " ★" if acc_gate >= 65 and tpd >= 8 else ""
                print(f"  {qe:5.1f} {qb:5.1f} {'-':>9} {'-':>9} {acc_gate:10.2f}% {tpd:5.1f} {n_final:8d}{mark}")


def main():
    run_cross_horizon_stack()
    run_cross_coin_gate()
    print("\n✅ 全部完成")


if __name__ == "__main__":
    main()
