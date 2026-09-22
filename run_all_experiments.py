"""统一脚本: 多 horizon 训练 + 无前视评估 + 跨 horizon stacking。

方向 A: 换 horizon → H=5/15/30 各 ETH/BTC 10 seed LGB 独立无前视评估
方向 B: 跨 horizon stacking → H5+H15+H30 三模型 meta-learner 融合
方向 C: H15 基线加强 → 更多 seed + 调参
方向 D: 两币 gate → ETH∩BTC 同时强信号才下注
"""
import os, sys, gc, time, argparse, json
sys.path.insert(0, "/workspace")
import numpy as np
import lightgbm as lgb
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]


def train_one(sym, horizon, seed):
    """训练单个 LGB 模型, 保存模型 + meta_val/test P。"""
    root = f"{config.MODEL_DIR}/pool_h{horizon}"
    os.makedirs(root, exist_ok=True)
    mp = f"{root}/{sym}_lgb_seed{seed}.txt"
    if os.path.exists(mp):
        # 检查 P 是否也存在
        for split in ["meta_val", "test"]:
            pp = f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_lgb_seed{seed}_{split}_P.npy"
            if not os.path.exists(pp):
                break
        else:
            return "SKIP"

    t0 = time.time()
    ctx = AssetContext(sym, horizon=horizon)
    print(f"  [{sym} h{horizon} seed{seed}] feats={len(ctx.feat_names)}", flush=True)

    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    Xes = ctx.Xall[es_mask]
    yes = ctx.label[es_mask].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("train")) * 50, 0.5, 5.0)
    print(f"    tr={Xtr.shape} es={Xes.shape}", flush=True)

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 63,
        "min_data_in_leaf": 200, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "verbose": -1, "seed": seed,
    }
    tr_ds = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)
    m = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    m.save_model(mp)
    auc = m.best_score["valid_0"]["auc"]
    print(f"    best_iter={m.best_iteration} auc={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 保存 P
    for split in ["meta_val", "test"]:
        mask = ctx.split_rows[split]
        Xt = ctx.Xall[mask]
        raw = m.predict(Xt)
        np.save(f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_lgb_seed{seed}_{split}_P.npy",
                raw.astype(np.float32))

    del ctx, m, Xtr, ytr, Xes, yes, wtr
    gc.collect()
    return f"OK auc={auc:.4f}"


def rank_ens(arrays):
    """排名集成。"""
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def load_P(sym, horizon, split, seeds=SEEDS):
    """加载所有 seed 的 P 数组。"""
    arrs = []
    for s in seeds:
        fp = f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_lgb_seed{s}_{split}_P.npy"
        if os.path.exists(fp):
            arrs.append(np.load(fp).astype(np.float64))
    return arrs


def eval_nolookahead(sym, horizon, split, ens_arrays, q=99.0, win_days=30, cold=30):
    """无前视评估: 滚动历史分位数阈值, 多空独立。"""
    ctx = AssetContext(sym, horizon=horizon)
    mask = ctx.split_rows[split]
    y = ctx.label[mask]
    times = ctx.times(split)

    p = rank_ens(ens_arrays) if len(ens_arrays) > 1 else ens_arrays[0]
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)

    day_of = np.asarray(times).astype("datetime64[s]").astype(np.int64) // 86400
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
    tpd = sel.sum() / len(days)
    # AUC (全样本)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return acc * 100, tpd, int(sel.sum()), auc


def eval_at_q(sym, horizon, split, ens_arrays, qlist):
    """扫多个 q 值。"""
    results = []
    for q in qlist:
        acc, tpd, n, auc = eval_nolookahead(sym, horizon, split, ens_arrays, q=q)
        results.append((q, acc, tpd, n, auc))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["train", "eval", "stack", "all"])
    ap.add_argument("--horizons", default="5,15,30")
    ap.add_argument("--syms", default="ETH,BTC")
    a = ap.parse_args()
    horizons = [int(x) for x in a.horizons.split(",")]
    syms = a.syms.split(",")

    # ========== 训练 ==========
    if a.mode in ("train", "all"):
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)
        for sym in syms:
            for h in horizons:
                for s in SEEDS:
                    r = train_one(sym, h, s)
                    print(f"  {sym} h{h} seed{s}: {r}")
        gc.collect()

    # ========== 评估每个 horizon 单独表现 ==========
    if a.mode in ("eval", "all"):
        print("\n" + "=" * 60)
        print("单 HORIZON 无前视评估 (10-seed rank ens)")
        print("=" * 60)
        QLIST = [97.0, 98.0, 98.5, 99.0, 99.2, 99.5, 99.8]
        for sym in syms:
            for h in horizons:
                print(f"\n--- {sym} h{h} ---")
                for split in ["meta_val", "test"]:
                    arrs = load_P(sym, h, split)
                    if not arrs:
                        print(f"  {split}: 无模型")
                        continue
                    results = eval_at_q(sym, h, split, arrs, QLIST)
                    print(f"  {split}  AUC={results[0][4]:.4f}")
                    print(f"  {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
                    for q, acc, tpd, n, _ in results:
                        mark = " ★" if acc >= 64 and tpd >= 11 else ""
                        print(f"  {q:6.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")

    # ========== 跨 horizon stacking ==========
    if a.mode in ("stack", "all"):
        print("\n" + "=" * 60)
        print("跨 HORIZON STACKING: H5 + H15 + H30 → meta LGB")
        print("=" * 60)
        for sym in syms:
            print(f"\n### {sym}")
            _cross_horizon_stack(sym)


def _cross_horizon_stack(sym):
    """用 H5/H15/H30 各 10-seed ens 的 P 作为 meta 特征, 训练 meta LGB。"""
    horizons = [5, 15, 30]

    # ---- 构建 meta 特征: 每个 horizon 的 rank_ens P + conf ----
    def build_meta_arrays(split):
        feats = []
        for h in horizons:
            arrs = load_P(sym, h, split)
            if not arrs:
                return None, None, None
            p = rank_ens(arrs) if len(arrs) > 1 else arrs[0]
            conf = np.abs(p - 0.5) * 2
            feats.append(p)
            feats.append(conf)
        return np.stack(feats, axis=1)

    Xtr = build_meta_arrays("meta_val")
    if Xtr[0] is None:
        print("  meta_val 无 P 数据")
        return

    ctx = AssetContext(sym, horizon=15)  # 用 h15 的 y/retf 做目标
    ytr = ctx.label[ctx.split_rows["meta_val"]].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("meta_val")) * 50, 0.5, 5.0)

    # early_stop 用 h15 的 meta_val 数据（因为那是完整的时间段）
    # 我们把 meta_val 后半段切做 val
    n = len(ytr)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[int(n * 0.7):] = True

    print(f"  meta_feat shape: {Xtr[0].shape}  n_feats={Xtr[0].shape[1]}")

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 100, "verbose": -1,
    }
    tr_ds = lgb.Dataset(Xtr[0][~val_mask], label=ytr[~val_mask], weight=wtr[~val_mask])
    es_ds = lgb.Dataset(Xtr[0][val_mask], label=ytr[val_mask], reference=tr_ds)
    m = lgb.train(params, tr_ds, num_boost_round=1000, valid_sets=[es_ds],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    print(f"  meta LGB: best_iter={m.best_iteration} auc={m.best_score['valid_0']['auc']:.4f}")

    # ---- 无前视评估 ----
    for split in ["meta_val", "test"]:
        Xs = build_meta_arrays(split)
        if Xs[0] is None:
            continue
        P_meta = m.predict(Xs[0])
        # 封装成 ens_arrays 格式
        print(f"\n  {split} stacking:")
        qlist = [97.0, 98.0, 98.5, 99.0, 99.2, 99.5]
        results = eval_at_q(sym, 15, split, [P_meta], qlist)
        print(f"    AUC={results[0][4]:.4f}")
        print(f"    {'q':>6} {'acc':>8} {'tpd':>7} {'n':>8}")
        for q, acc, tpd, n, _ in results:
            mark = " ★" if acc >= 64 and tpd >= 11 else ""
            print(f"    {q:6.1f} {acc:7.2f}% {tpd:7.1f} {n:8d}{mark}")


if __name__ == "__main__":
    main()
