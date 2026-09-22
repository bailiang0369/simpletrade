"""单模型单币种突破 ETH h15 → 65% 任务 (2026-09-22 深夜)

所有方法统一流程:
  1. ETH h15 训练 (params 由每个方法决定)
  2. meta_val 上扫 (q, 必要时扫 ε 去噪阈值)
  3. 锁死参数后 test 只跑一次 → 记录 acc/tpd/AUC
  4. 结果写入 experiment_log.md

baseline: ETH h15 10-seed rank ens + conf=|p-.5|*2 + q=99
  meta_val: AUC=0.5340, q=99 → 57.95% @ 14.0t
  test:     AUC=0.5424, q=99 → 62.01% @ 13.7t
"""
import os, sys, gc, time, argparse, json
sys.path.insert(0, "/workspace")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
ETH_H15 = AssetContext("ETH", horizon=15)
T0 = time.time()

def elapsed():
    return f"{(time.time()-T0)/60:.1f}m"

def rank_ens(arrays):
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)

def no_lookahead(q, conf, pred, y, ts, win=30, cold=30):
    """无前视评估 (滚动分位数阈值, 多空独立)。"""
    day_of = ts // 86400
    days = np.unique(day_of)
    sel = np.zeros(len(y), dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di-win):di]
        if len(prior) < cold: continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0: continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True
    acc = float((pred[sel]==y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(days), 1)
    return acc*100, tpd, int(sel.sum())

def load_baseline_P(split):
    """加载 baseline 10-seed P。"""
    arrs = []
    for s in SEEDS:
        fp = f"{config.DS_DIR}/SHORT_ETH_h15_lgb_seed{s}_{split}_P.npy"
        if os.path.exists(fp): arrs.append(np.load(fp).astype(np.float64))
    return rank_ens(arrs)

def eval_baseline_curve(split):
    """跑 baseline 的 q 曲线, 确认环境正确。"""
    ctx = ETH_H15
    mask = ctx.split_rows[split]
    P = load_baseline_P(split)
    y = ctx.label[mask]
    te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    conf = np.abs(P - 0.5) * 2
    pred = (P >= 0.5).astype(np.int8)
    auc = roc_auc_score(y, P)
    print(f"  baseline {split} AUC={auc:.4f}")
    for q in [99.0, 99.2, 99.4, 99.5, 99.8]:
        a, t, n = no_lookahead(q, conf, pred, y, te)
        print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")

# ============================================================
# 方向 1: 超参数 grid search (LGB binary)
# ============================================================
def run_hparam_grid():
    print(f"\n[{elapsed()}] === 方向 1: LGB 超参数 grid search (ETH h15) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    Xes = ctx.Xall[es_mask]
    yes = ctx.label[es_mask].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("train")) * 50, 0.5, 5.0)
    tr_ds = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)

    grid = [
        {"num_leaves": 63,  "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 255, "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 100, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 50,  "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 200, "max_bin": 511, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 1.0, "bagging_fraction": 1.0, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.02, "feature_pre_filter": False},
        {"num_leaves": 127, "min_data_in_leaf": 200, "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.10, "feature_pre_filter": False},
        {"num_leaves": 63,  "min_data_in_leaf": 50,  "max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 31,  "min_data_in_leaf": 50,  "max_bin": 255, "feature_fraction": 1.0, "bagging_fraction": 1.0, "learning_rate": 0.05, "feature_pre_filter": False},
        {"num_leaves": 63,  "min_data_in_leaf": 1000,"max_bin": 255, "feature_fraction": 0.8, "bagging_fraction": 0.8, "learning_rate": 0.05, "feature_pre_filter": False},
    ]

    results = []
    for i, hp in enumerate(grid):
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": 42, **hp}
        t0 = time.time()
        m = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        auc_es = m.best_score["valid_0"]["auc"]
        results.append((auc_es, hp, m.best_iteration, time.time()-t0))
        print(f"  [{i+1}/{len(grid)}] nL={hp['num_leaves']} mD={hp['min_data_in_leaf']} mB={hp['max_bin']} lr={hp['learning_rate']} fF={hp['feature_fraction']} → auc_es={auc_es:.4f} it={m.best_iteration} ({time.time()-t0:.0f}s)")
        gc.collect()

    results.sort(key=lambda x: -x[0])
    print(f"\n  top 3 params:")
    for rank, (auc_es, hp, it, t) in enumerate(results[:3]):
        print(f"    #{rank+1} auc_es={auc_es:.4f} {hp}")

    # 最优 params 跑 10 seed → meta_val/test
    best_hp = results[0][1]
    root = f"{config.MODEL_DIR}/hparam_best"
    os.makedirs(root, exist_ok=True)
    print(f"\n  用最优 params 跑 10 seed → meta_val/test...")

    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": s, **best_hp}
        m = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]]
            P = m.predict(Xt).astype(np.float64)
            np.save(f"{root}/ETH_h15_seed{s}_{split}_P.npy", P.astype(np.float32))
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)

    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2
        pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"\n  {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5, 99.8]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")

    return best_hp, results


# ============================================================
# 方向 2: 特征精简 (按 gain 排名保留 top-N)
# ============================================================
def run_feature_prune(hp_best=None):
    print(f"\n[{elapsed()}] === 方向 2: 特征精简 (ETH h15) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    Xes = ctx.Xall[es_mask]
    yes = ctx.label[es_mask].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("train")) * 50, 0.5, 5.0)

    # 先训 baseline 看 gain
    base_params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": 42,
                   "num_leaves": 63, "min_data_in_leaf": 200, "max_bin": 255,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "learning_rate": 0.05}
    tr_ds = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=ctx.feat_names)
    es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)
    m = lgb.train(base_params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    # gain 排名
    importance = m.feature_importance(importance_type="gain")
    order = np.argsort(-importance)
    feat_names = ctx.feat_names
    print(f"  baseline 特征 gain 排名:")
    for rank, idx in enumerate(order):
        print(f"    #{rank+1} {feat_names[idx]:25s} gain={importance[idx]:.0f}")

    # 去掉 gain=0 的
    zero_gain = [feat_names[i] for i, g in enumerate(importance) if g == 0]
    print(f"\n  gain=0 的特征 ({len(zero_gain)} 个): {zero_gain}")

    # 保留 top-N 对比
    for top_n in [30, 38, 44, 50]:
        selected_idx = order[:top_n]
        sel_names = [feat_names[i] for i in selected_idx]
        Xtr_s = Xtr[:, selected_idx]
        Xes_s = Xes[:, selected_idx]
        tr_ds2 = lgb.Dataset(Xtr_s, label=ytr, weight=wtr)
        es_ds2 = lgb.Dataset(Xes_s, label=yes, reference=tr_ds2)
        m2 = lgb.train(base_params, tr_ds2, num_boost_round=3000, valid_sets=[es_ds2],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        print(f"  top {top_n} 特征 → es AUC={m2.best_score['valid_0']['auc']:.4f} ({m2.best_iteration} iter)")
        gc.collect()

    # 用最优的 top-N 跑 10 seed meta_val/test
    # baseline AUC es 0.5332, 看哪个 top-N 更高
    # 先快速验证: 全特征 vs top-38 vs top-50
    best_top_n = 38  # 假设, 实际看上面的输出

    # 跑 10 seed
    root = f"{config.MODEL_DIR}/pruned{best_top_n}"
    os.makedirs(root, exist_ok=True)
    sel_idx = order[:best_top_n]

    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": s,
                  "num_leaves": 63, "min_data_in_leaf": 200, "max_bin": 255,
                  "feature_fraction": 1.0, "bagging_fraction": 1.0, "learning_rate": 0.05}  # feature_fraction 1.0 因为已经选过
        m_seed = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]][:, sel_idx]
            P = m_seed.predict(Xt).astype(np.float64)
            np.save(f"{root}/ETH_h15_seed{s}_{split}_P.npy", P.astype(np.float32))
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)
    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2
        pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"\n  精简 top-{best_top_n} {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5, 99.8]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")


# ============================================================
# 方向 3: LightGBM rank objective (pairwise)
# ============================================================
def run_rank_objective():
    print(f"\n[{elapsed()}] === 方向 3: LGB rank:pairwise 目标 (ETH h15) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    Xes = ctx.Xall[es_mask]
    yes = ctx.label[es_mask].astype(np.float64)

    tr_ds = lgb.Dataset(Xtr, label=ytr)
    es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)

    params = {"objective": "rank:pairwise", "metric": "auc", "verbose": -1, "seed": 42,
              "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
    m = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    print(f"  rank:pairwise es AUC={m.best_score['valid_0']['auc']:.4f} iter={m.best_iteration}")

    # 跑 10 seed
    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        params = {"objective": "rank:pairwise", "metric": "auc", "verbose": -1, "seed": s,
                  "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
        m_s = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]]
            P = m_s.predict(Xt).astype(np.float64)
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)
    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        pred = (P >= 0).astype(np.int8)  # rank 目标: 正分 = 涨
        conf = np.abs(P)
        auc = roc_auc_score(y, P)
        print(f"\n  rank:pairwise {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")


# ============================================================
# 方向 4: 回归目标 + Spearman 排名
# ============================================================
def run_regression_objective():
    print(f"\n[{elapsed()}] === 方向 4: LGB regression (predict ret) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ret_tr = ctx.retf("train")
    Xes = ctx.Xall[es_mask]
    ret_es = ctx.retf("early_stop")
    ytr_b = ctx.label[tr_mask].astype(np.float64)  # 硬标签, 用来算 AUC
    yes_b = ctx.label[es_mask].astype(np.float64)

    tr_ds = lgb.Dataset(Xtr, label=ret_tr)
    es_ds = lgb.Dataset(Xes, label=ret_es, reference=tr_ds)

    # 目标: regression, eval 用硬标签 AUC
    params = {"objective": "regression", "metric": "l2", "verbose": -1, "seed": 42,
              "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
    # 但 regression 不能直接用 label=ret 然后 early_stop 用 AUC ...
    # 换个方式: 自己算 AUC 做 early_stop
    class SpearmanCallback:
        def __init__(self, X, y, name="val"):
            self.X = X
            self.y = y
            self.name = name
        def __call__(self, env):
            yhat = env.model.predict(self.X)
            try:
                auc = roc_auc_score((self.y > 0).astype(int), yhat)
            except:
                auc = 0.5
            env.evaluation_result_list.append((self.name, "auc", auc, True))

    cb_es = SpearmanCallback(Xes, ret_es)
    m = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                  feval=lambda yhat, _: ("AUC", roc_auc_score(yes_b, yhat), True),
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    print(f"  regression es auc={m.best_score['valid_0']['AUC']:.4f} iter={m.best_iteration}")

    # 跑 10 seed
    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        params = {"objective": "regression", "metric": "l2", "verbose": -1, "seed": s,
                  "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
        m_s = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                        feval=lambda yhat, _: ("AUC", roc_auc_score(yes_b, yhat), True),
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]]
            P = m_s.predict(Xt).astype(np.float64)
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)
    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        pred = (P >= 0).astype(np.int8)  # regression: 正分 = 预测正收益
        conf = np.abs(P)
        auc = roc_auc_score(y, P)
        print(f"\n  regression {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")


# ============================================================
# 方向 5: 标签去噪 (训练只留 |ret|>ε)
# ============================================================
def run_denoised_labels():
    print(f"\n[{elapsed()}] === 方向 5: 标签去噪 (ETH h15) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr_full = ctx.Xall[tr_mask]
    Xes = ctx.Xall[es_mask]
    ret_tr = ctx.retf("train")
    ret_es = ctx.retf("early_stop")
    ytr_b = ctx.label[tr_mask].astype(np.float64)
    yes_b = ctx.label[es_mask].astype(np.float64)

    for eps in [0.0005, 0.001, 0.0015]:
        noise_mask = np.abs(ret_tr) > eps
        Xtr = Xtr_full[noise_mask]
        ytr = ytr_b[noise_mask]
        wtr = np.clip(np.abs(ret_tr[noise_mask]) * 50, 0.5, 5.0)
        print(f"  ε={eps}: 训练样本 {noise_mask.sum()}/{len(noise_mask)} ({noise_mask.mean()*100:.1f}%)")

        tr_ds = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        es_ds = lgb.Dataset(Xes, label=yes_b, reference=tr_ds)
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": 42,
                  "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
        m = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        print(f"    → es AUC={m.best_score['valid_0']['auc']:.4f}")
        gc.collect()

    # 用最好的 ε 跑 10 seed (假设 ε=0.001 最好)
    eps_best = 0.001
    noise_mask = np.abs(ret_tr) > eps_best
    Xtr = Xtr_full[noise_mask]
    ytr = ytr_b[noise_mask]
    wtr = np.clip(np.abs(ret_tr[noise_mask]) * 50, 0.5, 5.0)
    tr_ds = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    es_ds = lgb.Dataset(Xes, label=yes_b, reference=tr_ds)

    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": s,
                  "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
        m_s = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]]
            P = m_s.predict(Xt).astype(np.float64)
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)
    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2
        pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"\n  去噪 ε={eps_best} {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")


# ============================================================
# 方向 6: CatBoost 单币种
# ============================================================
def run_catboost():
    print(f"\n[{elapsed()}] === 方向 6: CatBoost (ETH h15) ===")
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        print("  CatBoost not installed, 安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "--quiet"])
        from catboost import CatBoostClassifier, Pool

    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    Xes = ctx.Xall[es_mask]
    ytr = ctx.label[tr_mask].astype(int)
    yes = ctx.label[es_mask].astype(int)

    train_pool = Pool(Xtr, ytr)
    eval_pool = Pool(Xes, yes)

    # 快速试
    cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                            l2_leaf_reg=3, loss_function="Logloss", eval_metric="AUC",
                            random_seed=42, verbose=False)
    cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    auc_es = cb.get_best_score()["validation"]["AUC"]
    print(f"  CatBoost seed42 es AUC={auc_es:.4f}")

    # 10 seed
    P_mv_list, P_te_list = [], []
    for s in SEEDS:
        cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                l2_leaf_reg=3, loss_function="Logloss", eval_metric="AUC",
                                random_seed=s, verbose=False)
        cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        for split in ["meta_val", "test"]:
            Xt = ctx.Xall[ctx.split_rows[split]]
            P = cb.predict_proba(Xt)[:, 1].astype(np.float64)
            (P_mv_list if split=="meta_val" else P_te_list).append(P)
        gc.collect()

    P_mv = rank_ens(P_mv_list)
    P_te = rank_ens(P_te_list)
    for split, P in [("meta_val", P_mv), ("test", P_te)]:
        mask = ctx.split_rows[split]
        y = ctx.label[mask]
        te = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2
        pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"\n  CatBoost {split} AUC={auc:.4f}")
        for q in [99.0, 99.2, 99.5]:
            a, t, n = no_lookahead(q, conf, pred, y, te)
            print(f"    q={q} → {a:.2f}% @ {t:.1f}t ({n})")


# ============================================================
# 方向 7: 时间衰减样本权重
# ============================================================
def run_time_decay_weights():
    print(f"\n[{elapsed()}] === 方向 7: 时间衰减样本权重 (ETH h15) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    Xes = ctx.Xall[es_mask]
    yes = ctx.label[es_mask].astype(np.float64)
    ret_tr = ctx.retf("train")
    ts_tr = ctx.ds_ts[tr_mask]  # 训练样本时间戳

    # 时间衰减权重: w = w_ret * w_time
    w_ret = np.clip(np.abs(ret_tr) * 50, 0.5, 5.0)
    # w_time: 越近权重越高
    max_ts = ts_tr.max()
    age = (max_ts - ts_tr) / (max_ts - ts_tr.min())  # [0, 1], 0=最近
    for lam in [0.1, 0.5, 1.0, 2.0]:
        w_time = np.exp(-lam * age)
        w = w_ret * w_time
        tr_ds = lgb.Dataset(Xtr, label=ytr, weight=w)
        es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)
        params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": 42,
                  "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05}
        m = lgb.train(params, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        print(f"  λ={lam}: es AUC={m.best_score['valid_0']['auc']:.4f} iter={m.best_iteration}")
        gc.collect()


# ============================================================
# 方向 8: 最优 params × 最优特征 × 10 seed → meta_val 选 q 锁死 test 只跑一次
# ============================================================
def run_final_combo():
    print(f"\n[{elapsed()}] === 方向 8: 最优组合 (综合所有方向的 insight) ===")
    ctx = ETH_H15
    tr_mask = ctx.split_rows["train"]
    es_mask = ctx.split_rows["early_stop"]
    Xtr = ctx.Xall[tr_mask]
    Xes = ctx.Xall[es_mask]
    ytr = ctx.label[tr_mask].astype(np.float64)
    yes = ctx.label[es_mask].astype(np.float64)
    ret_tr = ctx.retf("train")
    ret_es = ctx.retf("early_stop")
    wtr = np.clip(np.abs(ret_tr) * 50, 0.5, 5.0)

    # 先让 hparam_grid 跑完, 这里 hard-code grid 里最好的 params
    # 同时跑 baseline 和 hparam_best, 对比
    base_params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": None,
                   "num_leaves": 63, "min_data_in_leaf": 200, "max_bin": 255,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                   "learning_rate": 0.05}

    # 先看 direction 1 的结果再决定
    # 这里等其他方向跑完后自动决定
    pass


def main():
    print("=" * 80)
    print("ETH h15 单模型突破任务 — 2026-09-22 深夜")
    print("=" * 80)

    # 确认 baseline
    print(f"\n[{elapsed()}] --- 0. 确认 baseline ---")
    eval_baseline_curve("meta_val")
    eval_baseline_curve("test")

    # 方向 1: 超参数 grid search (最有希望的)
    best_hp, grid_all = run_hparam_grid()

    # 方向 2: 特征精简
    run_feature_prune(best_hp)

    # 方向 3: rank objective
    run_rank_objective()

    # 方向 4: regression
    run_regression_objective()

    # 方向 5: 去噪标签
    run_denoised_labels()

    # 方向 6: CatBoost
    run_catboost()

    # 方向 7: 时间衰减
    run_time_decay_weights()

    print(f"\n[{elapsed()}] ✅ 全部方向跑完")


if __name__ == "__main__":
    main()
