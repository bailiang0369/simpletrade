#!/usr/bin/env python3
"""实验: XGBoost Pairwise Ranking 优化 top-1% 准确率。

核心思路:
- LightGBM lambdarank (listwise) 效果不佳 (0.541)
- XGBoost 使用 pairwise ranking, 算法与 listwise 不同, 可能更适合此类数据
- 按天分组, 每 24h 内样本作为一个 query group
- 标签按未来收益分 5 级 relevance (与 experiment_rank.py 一致)
- 5-fold CV 确定最佳迭代轮数, 5-seed bagging
- 使用增强特征集 (与 experiment_enhanced.py 一致)

输出: 打印 test 上 top-1% 准确率, 与 baseline 对比。
"""
import os
import sys
import gc
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk


# ============================================================
# 特征定义 (与 experiment_enhanced.py 完全一致)
# ============================================================
FEATURES = [
    "lr_5", "lr_15", "lr_30", "lr_120", "lr_240", "mom_60",
    "z_10", "z_30", "z_60", "z_120",
    "rvol_30", "rvol_60", "rvol_ratio_60_5", "rvol_z_60", "rvol_dir",
    "pos_30", "pos_60", "pos_120", "pos_240",
    "dd_240", "ru_240",
    "hh_dd_60", "ll_ru_60", "body_pos_60",
    "body_ratio", "up_wick", "lo_wick", "ngreen_10", "gap", "max_range_30",
    "tbr_z_30", "cvd_30", "cvd_60",
    "buyvol_strength_30", "tb_act_60", "ts_act_60", "tb_acc_30",
    "cvd_dir_30", "tbr_hi_60", "lr_skew_60", "up_body_ratio_30", "mom_align_30_240",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us", "is_eu", "ret_day",
    # 交叉特征
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel",
    "vol_cvd_interact", "di_plus",
]

EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us",       # 美盘时段交互
    "hour_cos_is_eu",       # 欧盘时段交互
    "hour_sin_rvol_60",     # 小时 * 波动率交互
    "consec_up",            # 连续上涨分钟数
    "consec_dn",            # 连续下跌分钟数
    "session_minutes",      # 当前交易时段已过分钟数
    "hour_sin_hour_cos",    # 小时交互 (类 one-hot)
]

BAGGED_SEEDS = [42, 49, 56, 63, 70]


# ============================================================
# Relevance 标签和 Group 构建 (与 experiment_rank.py 一致)
# ============================================================
def compute_relevance(retf):
    """将未来收益映射到 5 级 relevance (0-4)。"""
    fin = np.isfinite(retf)
    z = np.zeros_like(retf, dtype=np.float64)
    if fin.sum() > 10:
        mu = np.nanmean(retf)
        sd = np.nanstd(retf) + 1e-10
        z[fin] = (retf[fin] - mu) / sd
    return np.digitize(z, [-1.0, -0.3, 0.3, 1.0]).astype(np.int32)  # [0,4]


def build_daily_groups(ts_sec):
    """按天构建 query group。ts_sec: int64 epoch 秒。返回 group 边界数组。"""
    days = ts_sec // 86400
    _, counts = np.unique(days, return_counts=True)
    return counts.astype(np.int32)


# ============================================================
# 新增特征计算 (与 experiment_enhanced.py 完全一致)
# ============================================================
def compute_extra_raw(ctx):
    """为全部原始数据计算新增特征, 返回 dict of float32 arrays。"""
    t0 = time.time()
    close = ctx.c
    raw_ts = ctx.raw_ts
    n = len(close)

    # ---- 1. 分钟对数收益 (用于 rvol_60 滚动计算) ----
    lr_1 = np.zeros(n, dtype=np.float64)
    lr_1[1:] = np.log(close[1:] / close[:-1])

    # ---- 2. rvol_60 (raw级别, rolling std of 1-min log returns, 60窗, ddof=1) ----
    rvol_60 = np.zeros(n, dtype=np.float32)
    w = 60
    if n >= w:
        cs = np.cumsum(lr_1)
        cs2 = np.cumsum(lr_1 ** 2)
        roll_sum = cs[w:] - cs[:-w]
        roll_sum2 = cs2[w:] - cs2[:-w]
        roll_mean = roll_sum / w
        roll_var = roll_sum2 / w - roll_mean ** 2
        roll_var = np.maximum(roll_var, 0)
        roll_std = np.sqrt(roll_var * w / (w - 1))  # ddof=1
        rvol_60[w:] = (roll_std * 100).astype(np.float32)

    # ---- 3. 时间特征 (raw级别) ----
    hour = (raw_ts % 86400) // 3600
    minute_of_day = (raw_ts % 86400) // 60
    hour_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    hour_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    is_us = ((hour >= 13) & (hour < 21)).astype(np.float32)
    is_eu = ((hour >= 8) & (hour < 13)).astype(np.float32)

    # ---- 4. 连续同向K线计数 (consec_up / consec_dn) ----
    consec_up = np.zeros(n, dtype=np.int32)
    consec_dn = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            consec_up[i] = consec_up[i - 1] + 1
        else:
            consec_dn[i] = consec_dn[i - 1] + 1
    # 归一化到 [0, 1] 左右
    consec_up_f = np.clip(consec_up.astype(np.float32) / 50.0, 0, 1)
    consec_dn_f = np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1)

    # ---- 5. Session minutes (当前交易时段已过分钟数) ----
    session_minutes = np.zeros(n, dtype=np.float32)
    for i in range(n):
        h = hour[i]
        m = minute_of_day[i]
        if h < 8:
            session_minutes[i] = m                                 # Asian 0-8
        elif h < 13:
            session_minutes[i] = m - 8 * 60                        # London 8-16
        elif h < 21:
            session_minutes[i] = m - 13 * 60                       # NY 13-21
        else:
            session_minutes[i] = m - 21 * 60                       # post-NY
    session_minutes = session_minutes / 480.0

    # ---- 组装 ----
    extra = {
        "hour_sin_is_us": (hour_sin * is_us).astype(np.float32),
        "hour_cos_is_eu": (hour_cos * is_eu).astype(np.float32),
        "hour_sin_rvol_60": (hour_sin * rvol_60).astype(np.float32),
        "consec_up": consec_up_f.astype(np.float32),
        "consec_dn": consec_dn_f.astype(np.float32),
        "session_minutes": session_minutes.astype(np.float32),
        "hour_sin_hour_cos": (hour_sin * hour_cos).astype(np.float32),
    }

    print(f"  [extra] 计算了 {len(extra)} 个新增特征, "
          f"耗时 {time.time() - t0:.1f}s", flush=True)
    return extra


def get_extra_for_mask(extra_raw, ctx, mask):
    """从 extra_raw 中提取 mask 对应的行, 按 EXTRA_FEATURE_NAMES 顺序返回矩阵。"""
    ri = ctx.ds_to_raw[mask].astype(int)
    cols = []
    for name in EXTRA_FEATURE_NAMES:
        cols.append(extra_raw[name][ri])
    return np.column_stack(cols)


# ============================================================
# 交叉验证折构建 (按天分组, 确保同一日样本在同一折)
# ============================================================
def build_day_folds(ts_sec, n_folds=5, seed=42):
    """按天分组构建交叉验证折。确保同一日期的样本在同一折内。

    Returns
    -------
    folds : list of (train_idx, val_idx) tuples
    """
    days = ts_sec // 86400
    uniq_days = np.unique(days)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_days)
    fold_splits = np.array_split(uniq_days, n_folds)
    folds = []
    for val_days in fold_splits:
        val_mask = np.isin(days, val_days)
        train_mask = ~val_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


# ============================================================
# XGBoost Ranking 训练
# ============================================================
def train_xgb_rank(ctx, extra_raw, seeds=None, max_train=2_600_000, n_rounds=5000):
    """训练 XGBoost ranking 模型。

    对每个 seed:
      1. 在训练集上做 5-fold CV (按天分组) 确定最佳迭代轮数
      2. 在全量训练集上训练 best_rounds 轮
      3. 用 early_stop 集评估

    Parameters
    ----------
    ctx : AssetContext
    extra_raw : dict
        compute_extra_raw() 返回的新增特征.
    seeds : list of int, optional
    max_train : int
        训练集最大采样数.
    n_rounds : int
        CV 的最大轮数.

    Returns
    -------
    models : list of xgb.Booster
    """
    if seeds is None:
        seeds = BAGGED_SEEDS

    t0 = time.time()
    import xgboost as xgb

    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # ---- 早停集: 基础特征 + 新增特征 ----
    print("  [data] 加载早停集特征...", flush=True)
    Xes_base = ctx.X_subset(FEATURES, esm)
    Xes_extra = get_extra_for_mask(extra_raw, ctx, esm)
    Xes = np.column_stack([Xes_base, Xes_extra])
    del Xes_base, Xes_extra
    gc.collect()

    retf_es = ctx.retf("early_stop")
    rel_es = compute_relevance(retf_es)
    ts_es = ctx.times("early_stop").astype("datetime64[s]").astype(np.int64)
    grp_es = build_daily_groups(ts_es)

    des = xgb.DMatrix(Xes, label=rel_es)
    des.set_group(grp_es)
    del Xes
    gc.collect()
    print(f"  [data] 早停集: {len(rel_es)} 样本, {len(grp_es)} 天", flush=True)

    # 训练集全局 time 和 retf
    ts_tr_all = ctx.times("train").astype("datetime64[s]").astype(np.int64)
    retf_tr_all = ctx.retf("train")

    models = []
    for seed in seeds:
        print(f"\n  === seed {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        has_sampling = len(tr_idx) > max_train
        if has_sampling:
            keep = rng.choice(len(tr_idx), max_train, replace=False)
            tr_idx = tr_idx[keep]
            print(f"  [sample] 采样 {len(tr_idx)}/{len(tr_idx_all)}", flush=True)

        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True

        # 加载特征: 基础 + 新增
        Xtr_base = ctx.X_subset(FEATURES, train_mask)
        Xtr_extra = get_extra_for_mask(extra_raw, ctx, train_mask)
        Xtr = np.column_stack([Xtr_base, Xtr_extra])
        del Xtr_base, Xtr_extra
        gc.collect()

        # 获取采样后的 retf 和 times
        if has_sampling:
            keep_local = np.where(train_mask[tr_idx_all])[0]
            retf_tr = retf_tr_all[keep_local]
            ts_tr = ts_tr_all[keep_local]
        else:
            retf_tr = retf_tr_all
            ts_tr = ts_tr_all

        rel_tr = compute_relevance(retf_tr)
        grp_tr = build_daily_groups(ts_tr)
        print(f"  [data] 训练集: {len(rel_tr)} 样本, {len(grp_tr)} 天", flush=True)

        # ---- 5-fold CV 确定最佳 n_rounds ----
        print(f"  [CV] 5-fold cross-validation (按天分组)...", flush=True)
        cv_folds = build_day_folds(ts_tr, n_folds=5, seed=seed)
        cv_best_rounds = []

        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@1",
            "learning_rate": 0.02,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 100,
            "lambda": 1.0,
            "alpha": 0.05,
            "nthread": config.N_JOBS,
            "seed": seed,
            "verbosity": 0,
        }

        for fold_i, (tr_cv_idx, val_cv_idx) in enumerate(cv_folds):
            X_cv_tr = Xtr[tr_cv_idx]
            y_cv_tr = rel_tr[tr_cv_idx]
            X_cv_val = Xtr[val_cv_idx]
            y_cv_val = rel_tr[val_cv_idx]

            # 构建各折的 group
            ts_cv_tr = ts_tr[tr_cv_idx]
            ts_cv_val = ts_tr[val_cv_idx]
            grp_cv_tr = build_daily_groups(ts_cv_tr)
            grp_cv_val = build_daily_groups(ts_cv_val)

            dtrain_cv = xgb.DMatrix(X_cv_tr, label=y_cv_tr)
            dtrain_cv.set_group(grp_cv_tr)
            dval_cv = xgb.DMatrix(X_cv_val, label=y_cv_val)
            dval_cv.set_group(grp_cv_val)

            m_cv = xgb.train(
                params, dtrain_cv,
                num_boost_round=n_rounds,
                evals=[(dval_cv, "val")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            cv_best_rounds.append(m_cv.best_iteration)
            print(f"    fold {fold_i + 1}: best_iter={m_cv.best_iteration}, "
                  f"ndcg@1={m_cv.best_score:.4f}", flush=True)

            del dtrain_cv, dval_cv, X_cv_tr, y_cv_tr, X_cv_val, y_cv_val
            gc.collect()

        best_rounds = int(np.median(cv_best_rounds))
        print(f"  [CV] median best_rounds={best_rounds} "
              f"(range [{min(cv_best_rounds)}, {max(cv_best_rounds)}])", flush=True)

        # ---- 在全量训练集上训练 ----
        print(f"  [train] 全量训练 {best_rounds} 轮...", flush=True)
        dtrain = xgb.DMatrix(Xtr, label=rel_tr)
        dtrain.set_group(grp_tr)

        m = xgb.train(
            params, dtrain,
            num_boost_round=best_rounds,
            evals=[(des, "early_stop")],
            verbose_eval=False,
        )
        models.append(m)

        # 评估 early_stop 集上的 NDCG (取最后轮次的 score)
        ndcg_val = m.best_score
        print(f"  [result] seed{seed} rounds={best_rounds} "
              f"early_stop_ndcg@1={ndcg_val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        del Xtr, dtrain
        gc.collect()

    print(f"\n  XGB Rank {len(models)} seeds done in {time.time() - t0:.0f}s", flush=True)
    return models


# ============================================================
# 预测: 排名平均 (rank averaging)
# ============================================================
def predict_rank(models, ctx, extra_raw, split):
    """排名平均: 每棵树的 raw score 转百分位秩再平均, 返回 float64 分数。"""
    import xgboost as xgb
    mask = ctx.split_rows[split]
    X_base = ctx.X_subset(FEATURES, mask)
    X_extra = get_extra_for_mask(extra_raw, ctx, mask)
    X = np.column_stack([X_base, X_extra])
    del X_base, X_extra
    gc.collect()

    d = xgb.DMatrix(X)
    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        s = m.predict(d)
        R[i] = np.argsort(np.argsort(s)).astype(np.float64) / (n - 1)
    del X, d
    gc.collect()
    return np.asarray(R.mean(axis=0), dtype=np.float64)


# ============================================================
# 主函数
# ============================================================
def main():
    symbol = "BTC"
    print(f"\n{'=' * 65}")
    print(f"  XGBoost Pairwise Ranking 实验: {symbol} H30")
    print(f"  - 特征: {len(FEATURES)} 基础 + {len(EXTRA_FEATURE_NAMES)} 新增 = "
          f"{len(FEATURES) + len(EXTRA_FEATURE_NAMES)} 维")
    print(f"  - 目标: rank:pairwise (vs LightGBM lambdarank)")
    print(f"  - 5-fold CV 确定最佳轮数 + 5-seed bagging")
    print(f"  - 5 级 relevance (0-4) 按未来收益 z-score 分桶")
    print(f"  - 按天分组 (daily query groups)")
    print(f"  - LightGBM lambdarank 基线: 0.541")
    print(f"  - 原始 GBDT 基线: ~0.617")
    print(f"{'=' * 65}\n")

    # 1. 加载数据
    print("[1/4] 加载数据...")
    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    print(f"  [ctx] 数据集行数: {len(ctx.ds_ts)}, "
          f"特征数: {len(ctx.feat_names)}", flush=True)

    # 2. 计算新增特征
    print("[2/4] 计算新增特征...")
    extra_raw = compute_extra_raw(ctx)

    # 3. 训练
    print("[3/4] 训练 XGBoost Ranking...")
    models = train_xgb_rank(ctx, extra_raw)

    # 4. 测试集评估
    print("[4/4] 评估测试集...")
    pt = predict_rank(models, ctx, extra_raw, "test")
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")

    result = evaluate_topk(pt, y_te, retf_te, ts_te)

    print(f"\n{'=' * 65}")
    print(f"  XGBoost Rank 测试集结果 (top {config.COVERAGE * 100:.0f}%)")
    print(f"{'=' * 65}")
    print(f"  准确率:       {result['accuracy']:.4f}")
    print(f"  交易次数:     {result['k']}")
    print(f"  交易/天:      {result['trades_per_day']:.1f}")
    print(f"  覆盖天数:     {result['n_days_selected']}")
    print(f"  平均收益:     {result['avg_ret_bps']:.1f} bps")
    print(f"  上涨收益:     {result['avg_ret_up_bps']:.1f} bps")
    print(f"  下跌收益:     {result['avg_ret_dn_bps']:.1f} bps")
    print(f"  平均置信度:   {result['conf_mean']:.4f}")
    if result['acc_by_month']:
        min_acc = min(result['acc_by_month'].values())
        min_month = min(result['acc_by_month'], key=result['acc_by_month'].get)
        print(f"  月度最低:     {min_acc:.4f} ({min_month[-7:]})")
        print(f"\n  月度明细:")
        for m in sorted(result['acc_by_month'].keys()):
            bar = "#" * int(result['acc_by_month'][m] * 40)
            print(f"    {m[-7:]}: acc={result['acc_by_month'][m]:.4f}  "
                  f"n={result['n_by_month'][m]:4d}  {bar}")

    # 5. 对比基线
    print(f"\n{'=' * 65}")
    print(f"  对比分析")
    print(f"{'=' * 65}")

    # 对比原始 GBDT 基线
    baseline_path = f"{config.DS_DIR}/{symbol}_gbdt_h30_pt.npy"
    alt_path = f"{config.DS_DIR}/{symbol}_gbdt_pt.npy"
    if not os.path.exists(baseline_path) and os.path.exists(alt_path):
        baseline_path = alt_path

    if os.path.exists(baseline_path):
        pt_base = np.load(baseline_path).astype(np.float64)
        r_base = evaluate_topk(pt_base, y_te, retf_te, ts_te)
        print(f"  原始 GBDT 基线:")
        print(f"    准确率:     {r_base['accuracy']:.4f}")
        print(f"    收益:       {r_base['avg_ret_bps']:.1f} bps")
        delta = (result['accuracy'] - r_base['accuracy']) * 100
        print(f"  XGB Rank vs GBDT: {delta:+.2f} pp")

    # 对比 LightGBM lambdarank
    lr_path = f"{config.DS_DIR}/{symbol}_lambdarank_pt.npy"
    if os.path.exists(lr_path):
        pt_lr = np.load(lr_path).astype(np.float64)
        r_lr = evaluate_topk(pt_lr, y_te, retf_te, ts_te)
        print(f"  LightGBM Lambdarank:")
        print(f"    准确率:     {r_lr['accuracy']:.4f}")
        print(f"    收益:       {r_lr['avg_ret_bps']:.1f} bps")
        delta = (result['accuracy'] - r_lr['accuracy']) * 100
        print(f"  XGB Rank vs Lambdarank: {delta:+.2f} pp")

    # 6. 保存预测结果
    save_path = f"{config.DS_DIR}/{symbol}_xgb_rank_pt.npy"
    np.save(save_path, pt.astype(np.float32))
    print(f"\n  [saved] {save_path}")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)

    # 清理
    del ctx, extra_raw, models
    gc.collect()
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()