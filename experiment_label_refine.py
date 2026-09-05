#!/usr/bin/env python3
"""实验: 标签精细化 - 分位数过滤 + 清晰信号加权。

基于 experiment_enhanced.py 的完整结构, 改进:
- 移除噪声标签: 剔除未来收益绝对值最小的 40% 训练样本
- 清晰信号加权: 保留样本的权重 = |ret_future| / median(|ret_future|), 上限 10x
- 与原增强模型对比

核心思路:
  当前二分类标签噪声大, 因为很多样本的收益极小 (<0.02%)。
  这些样本的涨跌方向基本随机, 会混淆模型。
  移除中间 40% 的低信号样本, 让模型专注于学习清晰信号。

过滤仅应用于 TRAINING 集。early_stop / meta_val / test 集保持不变。
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
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]

EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us",
    "hour_cos_is_eu",
    "hour_sin_rvol_60",
    "consec_up",
    "consec_dn",
    "session_minutes",
    "hour_sin_hour_cos",
]

BAGGED_SEEDS = [42, 49, 56, 63, 70]


# ============================================================
# 自定义评价函数
# ============================================================
def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs)
    k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


# ============================================================
# 新增特征计算 (与 experiment_enhanced.py 完全一致)
# ============================================================
def compute_extra_raw(ctx):
    t0 = time.time()
    close = ctx.c
    raw_ts = ctx.raw_ts
    n = len(close)

    lr_1 = np.zeros(n, dtype=np.float64)
    lr_1[1:] = np.log(close[1:] / close[:-1])

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
        roll_std = np.sqrt(roll_var * w / (w - 1))
        rvol_60[w:] = (roll_std * 100).astype(np.float32)

    hour = (raw_ts % 86400) // 3600
    minute_of_day = (raw_ts % 86400) // 60
    hour_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    hour_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    is_us = ((hour >= 13) & (hour < 21)).astype(np.float32)
    is_eu = ((hour >= 8) & (hour < 13)).astype(np.float32)

    consec_up = np.zeros(n, dtype=np.int32)
    consec_dn = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            consec_up[i] = consec_up[i - 1] + 1
        else:
            consec_dn[i] = consec_dn[i - 1] + 1
    consec_up_f = np.clip(consec_up.astype(np.float32) / 50.0, 0, 1)
    consec_dn_f = np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1)

    session_minutes = np.zeros(n, dtype=np.float32)
    for i in range(n):
        h = hour[i]
        m = minute_of_day[i]
        if h < 8:
            session_minutes[i] = m
        elif h < 13:
            session_minutes[i] = m - 8 * 60
        elif h < 21:
            session_minutes[i] = m - 13 * 60
        else:
            session_minutes[i] = m - 21 * 60
    session_minutes = session_minutes / 480.0

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
    ri = ctx.ds_to_raw[mask].astype(int)
    cols = []
    for name in EXTRA_FEATURE_NAMES:
        cols.append(extra_raw[name][ri])
    return np.column_stack(cols)


# ============================================================
# 标签过滤函数 (基于未来收益绝对值分位数)
# ============================================================
def filter_by_ret_percentile(retf, keep_ratio=0.6):
    """按未来收益绝对值分位数过滤训练样本。

    Parameters
    ----------
    retf : ndarray
        未来收益数组 (训练集).
    keep_ratio : float
        保留比例 (0.6 = 保留 |ret| 最大的 60%, 移除最小的 40%).

    Returns
    -------
    keep_mask : ndarray of bool
    stats : dict
    """
    abs_ret = np.abs(retf)
    threshold = np.percentile(abs_ret, (1 - keep_ratio) * 100)
    keep_mask = abs_ret >= threshold

    # 统计
    n_total = len(retf)
    n_keep = keep_mask.sum()
    n_remove = n_total - n_keep
    stats = {
        "keep_ratio": keep_ratio,
        "n_total": n_total,
        "n_keep": n_keep,
        "n_remove": n_remove,
        "remove_pct": n_remove / n_total * 100,
        "threshold": threshold,
        "abs_ret_mean": abs_ret.mean(),
        "abs_ret_median": np.median(abs_ret),
        "abs_ret_keep_mean": abs_ret[keep_mask].mean(),
        "abs_ret_remove_mean": abs_ret[~keep_mask].mean(),
    }
    return keep_mask, stats


# ============================================================
# 训练函数
# ============================================================
def train_enhanced_gbdt_filtered(ctx, extra_raw, z_threshold=None, keep_ratio=0.6,
                                  max_train=2_600_000):
    """训练增强版 GBDT + 可选标签过滤。

    z_threshold 兼容旧接口, 为 None 时按 keep_ratio 过滤。
    keep_ratio=1.0 时不过滤 (增强基线)。
    """
    t0 = time.time()
    feats = list(FEATURES)
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # ---- 早停集 ----
    Xes_base = ctx.X_subset(feats, esm)
    Xes_extra = get_extra_for_mask(extra_raw, ctx, esm)
    Xes = np.column_stack([Xes_base, Xes_extra])
    yes = ctx.label[esm].astype(np.float64)
    del Xes_base, Xes_extra
    gc.collect()

    # ---- 训练集未来收益 (用于过滤和加权, 保持全量用于权重索引) ----
    train_retf_full = ctx.retf("train")

    # ---- 标签过滤 (仅过滤 tr_idx_all, 保持 train_retf_full 全量) ----
    use_filter = (z_threshold is not None and z_threshold > 0) or keep_ratio < 1.0
    filter_stats = None

    if use_filter and keep_ratio < 1.0 and z_threshold is None:
        keep_mask, filter_stats = filter_by_ret_percentile(train_retf_full, keep_ratio)
        tr_idx_all = tr_idx_all[keep_mask]
        print(f"  [filter p{keep_ratio:.0%}] 保留 {len(tr_idx_all)}/{len(keep_mask)} "
              f"({len(tr_idx_all)/len(keep_mask)*100:.1f}%) 样本, "
              f"阈值 |ret|>={filter_stats['threshold']:.6f}", flush=True)
        print(f"  [filter] 保留样本 |ret| 均值={filter_stats['abs_ret_keep_mean']:.6f}, "
              f"移除样本 |ret| 均值={filter_stats['abs_ret_remove_mean']:.6f}", flush=True)
    elif use_filter and z_threshold is not None and z_threshold > 0:
        rvol_30 = ctx.X_subset(["rvol_30"], trm).ravel()
        ret_z = train_retf_full / (rvol_30 + 1e-10)
        keep_mask = np.abs(ret_z) > z_threshold
        tr_idx_all = tr_idx_all[keep_mask]
        print(f"  [filter z>{z_threshold}] 保留 {len(tr_idx_all)} 样本 "
              f"({len(tr_idx_all)/len(keep_mask)*100:.1f}%)", flush=True)
    else:
        print(f"  [filter] 未启用标签过滤 (增强基线)", flush=True)

    models = []
    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > max_train:
            keep = rng.choice(len(tr_idx), max_train, replace=False)
            tr_idx = tr_idx[keep]

        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True

        Xtr_base = ctx.X_subset(feats, train_mask)
        Xtr_extra = get_extra_for_mask(extra_raw, ctx, train_mask)
        Xtr = np.column_stack([Xtr_base, Xtr_extra])
        del Xtr_base, Xtr_extra
        gc.collect()

        ytr = ctx.label[train_mask].astype(np.float64)

        # ---- 加权采样 ----
        if len(tr_idx) < len(np.where(trm)[0]):
            keep_local = np.where(train_mask[np.where(trm)[0]])[0]
            raw_w = np.abs(train_retf_full[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf_full).astype(np.float64)

        # 基于 |ret| 中位数归一化, 让权重更有意义
        w_median = np.median(raw_w) + 1e-10
        w = np.clip(raw_w / w_median, 0.3, 10.0)

        # 坏时段加权
        train_ds_ts = ctx.ds_ts[train_mask]
        train_hour = (train_ds_ts % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any():
            w[bad_hour] *= 2.0
            print(f"  [weight] 坏时段样本 {bad_hour.sum()}/{len(bad_hour)} "
                  f"({bad_hour.mean() * 100:.1f}%) 权重 2x", flush=True)

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
            scale_pos_weight=1.0,
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
        print(f"  [seed{seed}] best_iter={m.best_iteration} "
              f"auc={auc_val:.4f} top1_acc={topk_val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        del Xtr, dtr, ytr, w
        gc.collect()

    # ---- Platt 后校准 ----
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
            print(f"  [calibrate] Platt校准: coef={calibrator.coef_[0][0]:.4f} "
                  f"({fin.sum()}样本)", flush=True)
        else:
            print(f"  [calibrate] 跳过Platt校准: 有效样本 {fin.sum()} 不足", flush=True)

    nfeat = len(feats) + len(EXTRA_FEATURE_NAMES)
    print(f"  [done] {len(models)}-seed bagging in {time.time()-t0:.0f}s "
          f"(nfeat={nfeat})", flush=True)
    return models, calibrator, filter_stats


def _predict_raw(ctx, models, extra_raw, mask):
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
    print(f"  标签精细化实验: {symbol} H30")
    print(f"  - 基于增强GBDT ({len(EXTRA_FEATURE_NAMES)} 个额外特征)")
    print(f"  - 分位数过滤: 保留 |ret| 最大的 60%/70%/80% 训练样本")
    print(f"  - 清晰信号加权: weight = |ret| / median(|ret|), 上限 10x")
    print(f"  - 5-seed bagging + Platt 校准")
    print(f"  - 过滤仅用于 TRAINING, 其余切分不变")
    print(f"{'=' * 65}\n")

    label = "label_refine"

    # 1. 加载数据
    print("[1/5] 加载数据...")
    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    print(f"  [ctx] 数据集行数: {len(ctx.ds_ts)}, "
          f"特征数: {len(ctx.feat_names)}", flush=True)

    # 2. 计算新增特征
    print("[2/5] 计算新增特征...")
    extra_raw = compute_extra_raw(ctx)

    # 3. 训练基线 (增强版, keep_ratio=1.0 = 不过滤)
    print("[3/5] 训练增强版基线 (不过滤)...")
    models_base, cal_base, _ = train_enhanced_gbdt_filtered(
        ctx, extra_raw, keep_ratio=1.0)

    pt_base = predict_enhanced(ctx, models_base, cal_base, extra_raw, "test")
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")
    r_base = evaluate_topk(pt_base, y_te, retf_te, ts_te)

    # 保存基线预测
    base_path = f"{config.DS_DIR}/BTC_{label}_base_pt.npy"
    np.save(base_path, pt_base)
    print(f"  [save] 基线预测保存至: {base_path}")

    # 4. 训练过滤模型 (尝试不同保留比例)
    print("[4/5] 训练过滤模型...")
    keep_ratios = [0.7, 0.6, 0.5]
    results = {}
    results["基线"] = r_base

    for kr in keep_ratios:
        tag = f"p{kr:.0%}"
        print(f"\n  {'=' * 60}")
        print(f"  训练: keep_ratio = {kr:.0%} (保留 |ret| 最大的 {kr:.0%})")
        print(f"  {'=' * 60}")
        models_f, cal_f, fs = train_enhanced_gbdt_filtered(
            ctx, extra_raw, keep_ratio=kr)

        pt_f = predict_enhanced(ctx, models_f, cal_f, extra_raw, "test")
        r_f = evaluate_topk(pt_f, y_te, retf_te, ts_te)
        results[tag] = r_f

        save_path = f"{config.DS_DIR}/BTC_{label}_{tag}_pt.npy"
        np.save(save_path, pt_f)
        print(f"  [save] 预测保存至: {save_path}")
        print(f"  [{tag}] 测试集: acc={r_f['accuracy']:.4f} "
              f"tpd={r_f['trades_per_day']:.1f} "
              f"ret={r_f['avg_ret_bps']:.1f}bps"
              f"  (移除 {fs['remove_pct']:.1f}% 训练样本)", flush=True)

        del models_f, cal_f, pt_f
        gc.collect()

    # 5. 结果对比
    print(f"\n[5/5] 结果对比...\n")
    print(f"{'=' * 65}")
    print(f"  测试集对比结果总览")
    print(f"{'=' * 65}")
    header = f"  {'Model':<18} {'Acc':>8} {'T/D':>6} {'Ret(bps)':>9} {'UpRet':>8} {'DnRet':>8} {'Conf':>7}"
    print(header)
    print(f"  {'-' * 64}")

    best_model = "基线"
    best_acc = r_base['accuracy']

    for name, r in results.items():
        delta = (r['accuracy'] - r_base['accuracy']) * 100
        marker = " <<<" if r['accuracy'] > best_acc else ""
        print(f"  {name:<18} {r['accuracy']:>8.4f} {r['trades_per_day']:>6.1f} "
              f"{r['avg_ret_bps']:>9.1f} {r['avg_ret_up_bps']:>8.1f} "
              f"{r['avg_ret_dn_bps']:>8.1f} {r['conf_mean']:>7.4f}"
              f"{marker}")
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_model = name

    print(f"  {'-' * 64}")
    for name, r in results.items():
        if name == "基线":
            continue
        delta = (r['accuracy'] - r_base['accuracy']) * 100
        print(f"  {name:<18} acc {delta:+.2f}pp  ret {r['avg_ret_bps'] - r_base['avg_ret_bps']:+.1f}bps")

    # 月度最低准确率
    print(f"\n  {'=' * 65}")
    print(f"  月度最低准确率对比")
    print(f"  {'=' * 65}")
    for name, r in results.items():
        if r['acc_by_month']:
            min_acc = min(r['acc_by_month'].values())
            min_month = min(r['acc_by_month'], key=r['acc_by_month'].get)
            print(f"  {name:<18} min_acc={min_acc:.4f} ({min_month[-7:]})")

    print(f"\n  {'=' * 65}")
    print(f"  最佳模型: {best_model}  (准确率 {best_acc:.4f})")
    print(f"  相比基线提升: {(best_acc - r_base['accuracy']) * 100:+.2f} pp")
    print(f"  {'=' * 65}")

    if best_acc >= config.TARGET_ACCURACY:
        print(f"\n  *** 达到目标: {best_acc:.4f} >= {config.TARGET_ACCURACY} ***")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)

    del ctx, extra_raw, models_base, cal_base
    gc.collect()
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()