#!/usr/bin/env python3
"""实验: 增强GBDT模型 - 新增特征 + 降低正样本偏置 + 坏时段加权。

基于诊断分析 (experiment_diagnose.py) 发现:
1. 模型准确率在不同小时差异巨大 (hour 2 为 86%, hour 17 为 46%)
2. 72% 的错误为假阳性 (预测涨但实际跌)

改进策略:
- 新增 session 交互特征: hour_sin * is_us, hour_cos * is_eu, hour_sin * rvol_60
- 新增连续同向K线计数: consec_up, consec_dn
- 新增 session 分钟数: 当前交易时段已过分钟数
- 新增 hour 交互: hour_sin * hour_cos
- 降低正样本偏置权重: SCALE_POS_WEIGHT = 1.0 (原 2.0)
- 去除正样本权重放大 (raw_w[ytr>0.5] *= 2.0 已删除)
- 对坏时段 (hour 17-20, 0-5) 的样本赋予更高权重 (2x)

训练模式与原 gbdt.py 一致:
- 加权采样 (基于未来收益绝对值)
- 5-seed bagging (seeds: 42, 49, 56, 63, 70)
- Platt 后校准 (在 meta_val 上拟合)
- top1_acc_eval 监控
- 排名平均组合
"""

import os
import sys
import gc
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk


# ============================================================
# 特征定义
# ============================================================
# 基础特征 (与 gbdt.py 完全一致, 59维)
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
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]

# 新增特征名称 (这些特征不在预构建的数据集中, 需从 raw 数据计算)
EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us",       # 美盘时段交互
    "hour_cos_is_eu",       # 欧盘时段交互
    "hour_sin_rvol_60",     # 小时 * 波动率交互
    "consec_up",            # 连续上涨分钟数
    "consec_dn",            # 连续下跌分钟数
    "session_minutes",      # 当前交易时段已过分钟数
    "hour_sin_hour_cos",    # 小时交互 (类 one-hot)
]

# 5 种子 bagging
BAGGED_SEEDS = [42, 49, 56, 63, 70]


# ============================================================
# 自定义评价函数
# ============================================================
def topk_acc_eval(preds, train_data):
    """自定义评价函数: 监控 top-1% 准确率 (早停仍用AUC)。"""
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs)
    k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


# ============================================================
# 新增特征计算 (基于原始K线数据)
# ============================================================
def compute_extra_raw(ctx):
    """为全部原始数据计算新增特征, 返回 dict of float32 arrays。

    这些特征不在预构建的数据集中 (ds_{symbol}.parquet 中无对应列),
    需要从 raw_ts / close 等原始数据重新计算。

    返回 dict, key=特征名, value=长度为 len(ctx.c) 的 float32 数组。
    """
    t0 = time.time()
    close = ctx.c
    raw_ts = ctx.raw_ts
    n = len(close)

    # ---- 1. 分钟对数收益 (用于 rvol_60 滚动计算) ----
    lr_1 = np.zeros(n, dtype=np.float64)
    lr_1[1:] = np.log(close[1:] / close[:-1])

    # ---- 2. rvol_60 (raw级别, rolling std of 1-min log returns, 60窗, ddof=1) ----
    # 使用 cumsum 做 O(n) 滚动窗口计算, 避免 O(n*w) 循环
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
    # 自前向后的累计计数, 遇反向K线则重置为0
    consec_up = np.zeros(n, dtype=np.int32)
    consec_dn = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            consec_up[i] = consec_up[i - 1] + 1
        else:
            consec_dn[i] = consec_dn[i - 1] + 1

    # 归一化到 [0, 1] 左右, 避免树模型对超大整数值的偏好
    consec_up_f = np.clip(consec_up.astype(np.float32) / 50.0, 0, 1)
    consec_dn_f = np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1)

    # ---- 5. Session minutes (当前交易时段已过分钟数) ----
    # 按 UTC 划分: Asian 0-8, London 8-16, NY 13-21, 其余为 post-NY
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

    # 归一化: 最大 session 约 480 分钟
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
    """从 extra_raw 中提取 mask 对应的行, 按 EXTRA_FEATURE_NAMES 顺序返回矩阵。

    Parameters
    ----------
    extra_raw : dict
        compute_extra_raw() 返回的 dict.
    ctx : AssetContext
    mask : ndarray
        布尔掩码, 长度 = len(ctx.ds_ts).

    Returns
    -------
    ndarray of shape (mask.sum(), len(EXTRA_FEATURE_NAMES)), dtype=float32
    """
    ri = ctx.ds_to_raw[mask].astype(int)
    cols = []
    for name in EXTRA_FEATURE_NAMES:
        cols.append(extra_raw[name][ri])
    return np.column_stack(cols)


# ============================================================
# 训练
# ============================================================
def train_enhanced_gbdt(ctx, extra_raw, max_train=2_600_000):
    """训练增强版 GBDT: 5-seed bagging + 新特征 + 降低正偏置 + 坏时段加权。

    Parameters
    ----------
    ctx : AssetContext
    extra_raw : dict
        compute_extra_raw() 返回的新增特征.
    max_train : int
        训练集最大采样数, 默认 2_600_000.

    Returns
    -------
    models : list of lgb.Booster
    calibrator : LogisticRegression or None
    """
    t0 = time.time()
    feats = list(FEATURES)
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # ---- 早停集: 基础特征 + 新增特征 ----
    Xes_base = ctx.X_subset(feats, esm)
    Xes_extra = get_extra_for_mask(extra_raw, ctx, esm)
    Xes = np.column_stack([Xes_base, Xes_extra])
    yes = ctx.label[esm].astype(np.float64)
    del Xes_base, Xes_extra
    gc.collect()

    # 训练集未来收益 (用于加权)
    train_retf = ctx.retf("train")

    models = []
    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > max_train:
            keep = rng.choice(len(tr_idx), max_train, replace=False)
            tr_idx = tr_idx[keep]

        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True

        # 加载特征: 基础 + 新增
        Xtr_base = ctx.X_subset(feats, train_mask)
        Xtr_extra = get_extra_for_mask(extra_raw, ctx, train_mask)
        Xtr = np.column_stack([Xtr_base, Xtr_extra])
        del Xtr_base, Xtr_extra
        gc.collect()

        ytr = ctx.label[train_mask].astype(np.float64)

        # ---- 加权采样 (改进版) ----
        if len(tr_idx) < len(tr_idx_all):
            # 有采样: train_mask[tr_idx_all] 指示保留的原始训练行
            keep_local = np.where(train_mask[tr_idx_all])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)

        # 基于未来收益绝对值加权, 裁剪极端值
        # 注意: 已移除正样本权重放大 (raw_w[ytr > 0.5] *= 2.0)
        w = np.clip(raw_w * 50, 0.5, 5.0)

        # ---- 坏时段加权: hour 17-20, 0-5 ----
        train_ds_ts = ctx.ds_ts[train_mask]
        train_hour = (train_ds_ts % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any():
            w[bad_hour] *= 2.0
            print(f"  [weight] 坏时段样本 {bad_hour.sum()}/{len(bad_hour)} "
                  f"({bad_hour.mean() * 100:.1f}%) 权重 2x", flush=True)

        # ---- LightGBM 参数 (降低正样本偏置) ----
        params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=2,
            min_data_in_leaf=100,
            lambda_l1=0.05,
            lambda_l2=1.0,
            scale_pos_weight=1.0,  # 从 2.0 降低到 1.0
            num_threads=config.N_JOBS,
            verbosity=-1,
            seed=seed,
            min_data=1,
        )

        dtr = lgb.Dataset(Xtr, ytr, weight=w)
        des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(
            params, dtr,
            num_boost_round=5000,
            valid_sets=[des],
            valid_names=['early_stop'],
            feval=topk_acc_eval,
            callbacks=[
                lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                lgb.log_evaluation(0),
            ],
        )
        models.append(m)

        auc_val = m.best_score['early_stop'].get('auc', -1)
        topk_val = m.best_score['early_stop'].get('top1_acc', -1)
        print(f"  [enhanced] seed{seed} best_iter={m.best_iteration} "
              f"auc={auc_val:.4f} top1_acc={topk_val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        del Xtr, dtr, ytr, w
        gc.collect()

    # ---- Platt 后校准 (在 meta_val 上拟合逻辑回归校准曲线) ----
    calibrator = None
    if "meta_val" in ctx.split_rows:
        from sklearn.linear_model import LogisticRegression
        cal_mask = ctx.split_rows["meta_val"]
        cal_p = _predict_raw(ctx, models, extra_raw, cal_mask)
        cal_y = ctx.label[cal_mask]
        fin = np.isfinite(cal_p)
        if fin.sum() > 1000:
            calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            logit_p = np.clip(cal_p[fin], 1e-7, 1 - 1e-7)
            logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
            calibrator.fit(logit_p, cal_y[fin])
            print(f"  [enhanced] Platt校准: 在meta_val上拟合 "
                  f"({fin.sum()}样本, coef={calibrator.coef_[0][0]:.4f})", flush=True)
        else:
            print(f"  [enhanced] 跳过Platt校准: 有效样本 {fin.sum()} 不足", flush=True)

    nfeat = len(feats) + len(EXTRA_FEATURE_NAMES)
    print(f"  [enhanced] {len(models)}-seed bagging fit done in "
          f"{time.time() - t0:.0f}s (nfeat={nfeat})", flush=True)
    return models, calibrator


def _predict_raw(ctx, models, extra_raw, mask):
    """原始预测 (排名平均, 未校准), 返回 float64 概率。"""
    X_base = ctx.X_subset(FEATURES, mask)
    X_extra = get_extra_for_mask(extra_raw, ctx, mask)
    X = np.column_stack([X_base, X_extra])
    del X_base, X_extra
    gc.collect()

    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
    del X
    gc.collect()
    return np.asarray(R.mean(axis=0), dtype=np.float64)


def predict_enhanced(ctx, models, calibrator, extra_raw, split):
    """预测: 排名平均 + 可选 Platt 校准。返回 [0,1] float32。"""
    mask = ctx.split_rows[split]
    p_raw = _predict_raw(ctx, models, extra_raw, mask)
    if calibrator is not None:
        logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7)
        logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
        p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    else:
        p = p_raw
    return np.asarray(p, dtype=np.float32)


# ============================================================
# 主函数
# ============================================================
def main():
    symbol = "BTC"
    print(f"\n{'=' * 65}")
    print(f"  增强GBDT模型实验: {symbol} H30")
    print(f"  - 新增 {len(EXTRA_FEATURE_NAMES)} 个特征")
    for fn in EXTRA_FEATURE_NAMES:
        print(f"    * {fn}")
    print(f"  - SCALE_POS_WEIGHT = 1.0 (原 2.0)")
    print(f"  - 去除正样本权重放大")
    print(f"  - 坏时段 (hour 17-20, 0-5) 样本权重 2x")
    print(f"  - {len(BAGGED_SEEDS)}-seed bagging + Platt 校准")
    print(f"  - 训练模式: 加权采样 + 排名平均 + top1_acc_eval")
    print(f"{'=' * 65}\n")

    # 1. 加载数据
    print("[1/4] 加载数据...")
    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    print(f"  [ctx] 数据集行数: {len(ctx.ds_ts)}, "
          f"特征数: {len(ctx.feat_names)}", flush=True)

    # 2. 计算新增特征 (全部原始数据)
    print("[2/4] 计算新增特征...")
    extra_raw = compute_extra_raw(ctx)

    # 3. 训练
    print("[3/4] 训练增强版 GBDT...")
    models, calibrator = train_enhanced_gbdt(ctx, extra_raw)

    # 4. 测试集评估
    print("[4/4] 评估测试集...")
    pt = predict_enhanced(ctx, models, calibrator, extra_raw, "test")
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")

    result = evaluate_topk(pt, y_te, retf_te, ts_te)

    # 保存预测 (用于后续校准)
    enhanced_pt_path = f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy"
    np.save(enhanced_pt_path, pt)
    print(f"  [save] test 预测保存至: {enhanced_pt_path}")

    # 也保存 meta_val 预测 (用于校准脚本)
    pv = predict_enhanced(ctx, models, calibrator, extra_raw, "meta_val")
    enhanced_pv_path = f"{config.DS_DIR}/BTC_gbdt_enhanced_pv.npy"
    np.save(enhanced_pv_path, pv)
    print(f"  [save] meta_val 预测保存至: {enhanced_pv_path}")

    print(f"\n{'=' * 65}")
    print(f"  增强GBDT 测试集结果 (top {config.COVERAGE * 100:.0f}%)")
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
            bar = "█" * int(result['acc_by_month'][m] * 40)
            print(f"    {m[-7:]}: acc={result['acc_by_month'][m]:.4f}  "
                  f"n={result['n_by_month'][m]:4d}  {bar}")

    # 5. 对比基线
    baseline_path = f"{config.DS_DIR}/{symbol}_gbdt_h30_pt.npy"
    alt_path = f"{config.DS_DIR}/{symbol}_gbdt_pt.npy"
    if not os.path.exists(baseline_path) and os.path.exists(alt_path):
        baseline_path = alt_path

    if os.path.exists(baseline_path):
        print(f"\n{'=' * 65}")
        print(f"  对比: 原始 GBDT 基线")
        print(f"{'=' * 65}")
        pt_base = np.load(baseline_path).astype(np.float64)
        r_base = evaluate_topk(pt_base, y_te, retf_te, ts_te)
        print(f"  基线准确率:   {r_base['accuracy']:.4f}")
        print(f"  增强准确率:   {result['accuracy']:.4f}")
        delta = (result['accuracy'] - r_base['accuracy']) * 100
        print(f"  提升:         {delta:+.2f} pp")
        print(f"  基线收益:     {r_base['avg_ret_bps']:.1f} bps")
        print(f"  增强收益:     {result['avg_ret_bps']:.1f} bps")
    else:
        print(f"\n  [对比] 基线预测文件不存在: {baseline_path}")
        print(f"  [对比] 跳过基线对比")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)

    # 清理
    del ctx, extra_raw, models, calibrator
    gc.collect()
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()