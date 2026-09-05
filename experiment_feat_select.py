#!/usr/bin/env python3
"""实验: GBDT 特征选择 - 评估不同特征子集对模型性能的影响。

核心思路:
- 当前模型有 66 维特征 (59 基础 + 7 新增)
- 部分特征可能为噪声, 通过特征选择减少过拟合, 提升泛化能力
- 基于完整模型训练后的特征重要性排序, 选择不同子集重新训练

方法:
1. 先训练完整模型 (同 experiment_enhanced.py)
2. 提取所有 5 个种子的特征重要性, 按平均重要性排序
3. 用不同特征子集训练:
   - Top 50% (33 特征)
   - Top 30% (20 特征)
   - Top 20% (13 特征)
4. 去除 10 个交叉特征, 测试是否引入噪声
5. 保持其他一切不变: 加权采样, 坏时段加权, Platt 校准, bagging
6. 保存各子集预测结果用于后续对比

特征选择基于 FIRST training 的重要性 (不再对每个子集重新计算重要性)。
仅过滤训练集使用的特征列, 模型架构 (num_leaves / learning_rate 等) 保持不变。
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
# 基础特征 (59 维)
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
    # 交叉特征 (10 个)
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]

# 新增特征 (7 个, 从 raw 数据计算)
EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us",       # 美盘时段交互
    "hour_cos_is_eu",       # 欧盘时段交互
    "hour_sin_rvol_60",     # 小时 * 波动率交互
    "consec_up",            # 连续上涨分钟数
    "consec_dn",            # 连续下跌分钟数
    "session_minutes",      # 当前交易时段已过分钟数
    "hour_sin_hour_cos",    # 小时交互 (类 one-hot)
]

# 全部特征 (66 维)
ALL_FEATURES = FEATURES + EXTRA_FEATURE_NAMES

# 交叉特征名称列表 (用于 "去除交叉特征" 实验)
CROSS_FEATURES = [
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence",
    "cvd_accel", "vol_cvd_interact", "di_plus",
]

# 5 种子 bagging
BAGGED_SEEDS = [42, 49, 56, 63, 70]


# ============================================================
# 自定义评价函数 (与 enhanced 一致)
# ============================================================
def topk_acc_eval(preds, train_data):
    """自定义评价函数: 监控 top-1% 准确率 (早停仍用 AUC)。"""
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs)
    k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


# ============================================================
# 新增特征计算 (与 enhanced 一致)
# ============================================================
def compute_extra_raw(ctx):
    """为全部原始数据计算新增特征, 返回 dict of float32 arrays。

    这些特征不在预构建的数据集 (ds_{symbol}.parquet) 中,
    需要从 raw_ts / close 等原始数据重新计算。
    """
    t0 = time.time()
    close = ctx.c
    raw_ts = ctx.raw_ts
    n = len(close)

    # ---- 1. 分钟对数收益 ----
    lr_1 = np.zeros(n, dtype=np.float64)
    lr_1[1:] = np.log(close[1:] / close[:-1])

    # ---- 2. rvol_60 (raw 级别滚动计算) ----
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

    # ---- 3. 时间特征 ----
    hour = (raw_ts % 86400) // 3600
    minute_of_day = (raw_ts % 86400) // 60
    hour_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    hour_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    is_us = ((hour >= 13) & (hour < 21)).astype(np.float32)
    is_eu = ((hour >= 8) & (hour < 13)).astype(np.float32)

    # ---- 4. 连续同向K线计数 ----
    consec_up = np.zeros(n, dtype=np.int32)
    consec_dn = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            consec_up[i] = consec_up[i - 1] + 1
        else:
            consec_dn[i] = consec_dn[i - 1] + 1
    consec_up_f = np.clip(consec_up.astype(np.float32) / 50.0, 0, 1)
    consec_dn_f = np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1)

    # ---- 5. Session minutes ----
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
    """从 extra_raw 中提取 mask 对应的行, 按 EXTRA_FEATURE_NAMES 顺序返回矩阵。"""
    ri = ctx.ds_to_raw[mask].astype(int)
    cols = []
    for name in EXTRA_FEATURE_NAMES:
        cols.append(extra_raw[name][ri])
    return np.column_stack(cols)


# ============================================================
# 特征重要性提取
# ============================================================
def extract_feature_importance(models, feat_names):
    """从已训练的模型列表中提取并平均特征重要性。

    Parameters
    ----------
    models : list of lgb.Booster
    feat_names : list of str

    Returns
    -------
    sorted_idx : ndarray
        按重要性降序排列的索引
    sorted_names : list of str
        按重要性降序排列的特征名
    sorted_avg : ndarray
        平均重要性 (gain), 降序
    sorted_std : ndarray
        重要性标准差, 降序
    """
    n_feat = len(feat_names)
    all_imp = np.zeros((len(models), n_feat), dtype=np.float64)
    for i, m in enumerate(models):
        imp = m.feature_importance(importance_type='gain')
        all_imp[i] = imp

    avg_imp = all_imp.mean(axis=0)
    std_imp = all_imp.std(axis=0)

    sorted_idx = np.argsort(-avg_imp)
    sorted_names = [feat_names[i] for i in sorted_idx]
    sorted_avg = avg_imp[sorted_idx]
    sorted_std = std_imp[sorted_idx]

    return sorted_idx, sorted_names, sorted_avg, sorted_std


def print_importance_ranking(sorted_names, sorted_avg, sorted_std, top_n=None):
    """打印特征重要性排序。"""
    n = top_n or len(sorted_names)
    print(f"\n  {'=' * 55}")
    print(f"  特征重要性排名 (Top {n}/{len(sorted_names)})")
    print(f"  {'=' * 55}")
    print(f"  {'Rank':>4s}  {'Feature':<30s}  {'Avg Gain':>10s}  {'Std Gain':>8s}")
    print(f"  {'-' * 55}")
    for i in range(n):
        name = sorted_names[i]
        marker = " <-- CROSS" if name in CROSS_FEATURES else ""
        print(f"  {i + 1:4d}  {name:<30s}  {sorted_avg[i]:>10.2f}  {sorted_std[i]:>8.2f}{marker}")
    print(f"  {'=' * 55}", flush=True)


# ============================================================
# 特征集训练函数 (参数化特征列表)
# ============================================================
def train_gbdt_with_features(ctx, extra_raw, selected_feats, tag="",
                             max_train=2_600_000):
    """使用指定特征子集训练 GBDT。

    逻辑与 experiment_enhanced.py 中的 train_enhanced_gbdt 完全一致,
    仅替换特征列表。

    Parameters
    ----------
    ctx : AssetContext
    extra_raw : dict
    selected_feats : list of str
        选中的特征名列表, 来自 ALL_FEATURES 的子集
    tag : str
        日志标签
    max_train : int

    Returns
    -------
    models : list of lgb.Booster
    calibrator : LogisticRegression or None
    """
    t0 = time.time()

    # 拆分选中的特征为基础/新增
    base_sel = [n for n in selected_feats if n in FEATURES]
    extra_sel = [n for n in selected_feats if n in EXTRA_FEATURE_NAMES]

    if not base_sel:
        print(f"  [{tag}] 错误: 未选中任何基础特征!", flush=True)
        return [], None

    n_feat = len(base_sel) + len(extra_sel)
    print(f"  [{tag}] 特征子集: {n_feat} 个特征 "
          f"(基础 {len(base_sel)}, 新增 {len(extra_sel)})", flush=True)

    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # ---- 早停集 ----
    Xes_base = ctx.X_subset(base_sel, esm)
    if extra_sel:
        Xes_extra = get_extra_for_mask(extra_raw, ctx, esm)
        Xes_extra = Xes_extra[:, [EXTRA_FEATURE_NAMES.index(n) for n in extra_sel]]
    else:
        Xes_extra = np.empty((esm.sum(), 0), dtype=np.float32)
    Xes = np.column_stack([Xes_base, Xes_extra])
    yes = ctx.label[esm].astype(np.float64)
    del Xes_base, Xes_extra
    gc.collect()

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

        # 加载特征: 仅选中的基础 + 新增
        Xtr_base = ctx.X_subset(base_sel, train_mask)
        if extra_sel:
            Xtr_extra = get_extra_for_mask(extra_raw, ctx, train_mask)
            Xtr_extra = Xtr_extra[:, [EXTRA_FEATURE_NAMES.index(n) for n in extra_sel]]
        else:
            Xtr_extra = np.empty((train_mask.sum(), 0), dtype=np.float32)
        Xtr = np.column_stack([Xtr_base, Xtr_extra])
        del Xtr_base, Xtr_extra
        gc.collect()

        ytr = ctx.label[train_mask].astype(np.float64)

        # ---- 加权采样 ----
        if len(tr_idx) < len(tr_idx_all):
            keep_local = np.where(train_mask[tr_idx_all])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)

        # ---- 坏时段加权 ----
        train_ds_ts = ctx.ds_ts[train_mask]
        train_hour = (train_ds_ts % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any():
            w[bad_hour] *= 2.0
            print(f"  [{tag}] seed{seed} 坏时段样本 {bad_hour.sum()}/{len(bad_hour)} "
                  f"({bad_hour.mean() * 100:.1f}%) 权重 2x", flush=True)

        # ---- LightGBM 参数 ----
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
        print(f"  [{tag}] seed{seed} best_iter={m.best_iteration} "
              f"auc={auc_val:.4f} top1_acc={topk_val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        del Xtr, dtr, ytr, w
        gc.collect()

    # ---- Platt 后校准 ----
    calibrator = None
    if "meta_val" in ctx.split_rows:
        from sklearn.linear_model import LogisticRegression
        cal_mask = ctx.split_rows["meta_val"]
        cal_p = _predict_raw_feat_select(ctx, models, extra_raw, cal_mask,
                                         base_sel, extra_sel)
        cal_y = ctx.label[cal_mask]
        fin = np.isfinite(cal_p)
        if fin.sum() > 1000:
            calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            logit_p = np.clip(cal_p[fin], 1e-7, 1 - 1e-7)
            logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
            calibrator.fit(logit_p, cal_y[fin])
            print(f"  [{tag}] Platt校准: 在meta_val上拟合 "
                  f"({fin.sum()}样本, coef={calibrator.coef_[0][0]:.4f})", flush=True)
        else:
            print(f"  [{tag}] 跳过Platt校准: 有效样本 {fin.sum()} 不足", flush=True)

    print(f"  [{tag}] {len(models)}-seed bagging fit done in "
          f"{time.time() - t0:.0f}s (nfeat={n_feat})", flush=True)
    return models, calibrator


def _predict_raw_feat_select(ctx, models, extra_raw, mask,
                             base_sel, extra_sel):
    """原始预测 (排名平均, 未校准), 使用指定特征子集。"""
    X_base = ctx.X_subset(base_sel, mask)
    if extra_sel:
        X_extra = get_extra_for_mask(extra_raw, ctx, mask)
        X_extra = X_extra[:, [EXTRA_FEATURE_NAMES.index(n) for n in extra_sel]]
    else:
        X_extra = np.empty((mask.sum(), 0), dtype=np.float32)
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


def predict_feat_select(ctx, models, calibrator, extra_raw, split,
                        base_sel, extra_sel):
    """预测: 排名平均 + 可选 Platt 校准。返回 [0,1] float32。"""
    mask = ctx.split_rows[split]
    p_raw = _predict_raw_feat_select(ctx, models, extra_raw, mask,
                                     base_sel, extra_sel)
    if calibrator is not None:
        logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7)
        logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
        p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    else:
        p = p_raw
    return np.asarray(p, dtype=np.float32)


# ============================================================
# 评估辅助函数
# ============================================================
def evaluate_and_report(pt, y_te, retf_te, ts_te, label=""):
    """评估并打印结果, 返回 metrics dict。"""
    result = evaluate_topk(pt, y_te, retf_te, ts_te)
    print(f"\n  [{label}] 测试集结果 (top {config.COVERAGE * 100:.0f}%)")
    print(f"  {'-' * 40}")
    print(f"    准确率:       {result['accuracy']:.4f}")
    print(f"    交易次数:     {result['k']}")
    print(f"    交易/天:      {result['trades_per_day']:.1f}")
    print(f"    平均收益:     {result['avg_ret_bps']:.1f} bps")
    print(f"    上涨收益:     {result['avg_ret_up_bps']:.1f} bps")
    print(f"    下跌收益:     {result['avg_ret_dn_bps']:.1f} bps")
    print(f"    平均置信度:   {result['conf_mean']:.4f}")
    if result['acc_by_month']:
        min_acc = min(result['acc_by_month'].values())
        min_month = min(result['acc_by_month'], key=result['acc_by_month'].get)
        print(f"    月度最低:     {min_acc:.4f} ({min_month[-7:]})")
    return result


# ============================================================
# 主函数
# ============================================================
def main():
    symbol = "BTC"
    print(f"\n{'=' * 70}")
    print(f"  GBDT 特征选择实验: {symbol} H30")
    print(f"  总特征数: {len(ALL_FEATURES)} ({len(FEATURES)} 基础 + "
          f"{len(EXTRA_FEATURE_NAMES)} 新增)")
    print(f"  交叉特征: {len(CROSS_FEATURES)} 个")
    print(f"  方法: 先训练完整模型 → 提取重要性 → 用 Top-N 子集重新训练")
    print(f"  - 保持加权采样, 坏时段加权, Platt 校准, 5-seed bagging")
    print(f"{'=' * 70}\n")

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("[1/5] 加载数据...")
    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    print(f"  [ctx] 数据集行数: {len(ctx.ds_ts)}", flush=True)

    # ============================================================
    # 2. 计算新增特征
    # ============================================================
    print("[2/5] 计算新增特征...")
    extra_raw = compute_extra_raw(ctx)

    # ============================================================
    # 3. 训练完整模型 (66 特征) + 提取特征重要性
    # ============================================================
    print("[3/5] 训练完整模型 (ALL 66 features)...")
    models_full, calibrator_full = train_gbdt_with_features(
        ctx, extra_raw, ALL_FEATURES, tag="full"
    )

    # 提取特征重要性
    sorted_idx, sorted_names, sorted_avg, sorted_std = \
        extract_feature_importance(models_full, ALL_FEATURES)
    print_importance_ranking(sorted_names, sorted_avg, sorted_std, top_n=66)

    # ============================================================
    # 4. 定义实验子集
    # ============================================================
    print("[4/5] 定义特征子集并训练...")

    # 按重要性排序的特征名
    feat_by_rank = sorted_names  # 已按重要性降序

    n_total = len(ALL_FEATURES)
    subsets = [
        ("top50", feat_by_rank[:33], f"Top 50% ({n_total // 2} features)"),
        ("top30", feat_by_rank[:20], f"Top 30% (20 features)"),
        ("top20", feat_by_rank[:13], f"Top 20% (13 features)"),
    ]

    # 去除交叉特征: ALL_FEATURES 中过滤掉 CROSS_FEATURES
    no_cross_feats = [n for n in ALL_FEATURES if n not in CROSS_FEATURES]
    subsets.append(
        ("no_cross", no_cross_feats,
         f"No-Cross (remove {len(CROSS_FEATURES)} cross features, "
         f"{len(no_cross_feats)} remaining)")
    )

    # 存储所有实验结果
    all_results = {}

    # 完整模型结果
    print(f"\n  --- 评估完整模型 ---")
    pt_full = predict_feat_select(
        ctx, models_full, calibrator_full, extra_raw, "test",
        FEATURES, EXTRA_FEATURE_NAMES
    )
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")
    result_full = evaluate_and_report(pt_full, y_te, retf_te, ts_te,
                                      label="full")
    all_results["full"] = {
        "result": result_full,
        "pt": pt_full,
        "n_feat": n_total,
        "tag": "ALL 66 features",
    }

    # 保存完整模型预测
    full_pt_path = f"{config.DS_DIR}/BTC_gbdt_featselect_full_pt.npy"
    np.save(full_pt_path, pt_full)
    print(f"  [save] full test 预测: {full_pt_path}")

    full_pv_path = f"{config.DS_DIR}/BTC_gbdt_featselect_full_pv.npy"
    pv_full = predict_feat_select(
        ctx, models_full, calibrator_full, extra_raw, "meta_val",
        FEATURES, EXTRA_FEATURE_NAMES
    )
    np.save(full_pv_path, pv_full)
    print(f"  [save] full meta_val 预测: {full_pv_path}")

    # 逐个训练子集
    for tag, sel_feats, desc in subsets:
        print(f"\n  --- 训练: {desc} ---")
        base_sel = [n for n in sel_feats if n in FEATURES]
        extra_sel = [n for n in sel_feats if n in EXTRA_FEATURE_NAMES]

        models_sub, calibrator_sub = train_gbdt_with_features(
            ctx, extra_raw, sel_feats, tag=tag
        )

        pt_sub = predict_feat_select(
            ctx, models_sub, calibrator_sub, extra_raw, "test",
            base_sel, extra_sel
        )
        result_sub = evaluate_and_report(pt_sub, y_te, retf_te, ts_te,
                                         label=tag)
        all_results[tag] = {
            "result": result_sub,
            "pt": pt_sub,
            "n_feat": len(sel_feats),
            "tag": desc,
        }

        # 保存预测
        pt_path = f"{config.DS_DIR}/BTC_gbdt_featselect_{tag}_pt.npy"
        np.save(pt_path, pt_sub)
        print(f"  [save] {tag} test 预测: {pt_path}")

        pv_sub = predict_feat_select(
            ctx, models_sub, calibrator_sub, extra_raw, "meta_val",
            base_sel, extra_sel
        )
        pv_path = f"{config.DS_DIR}/BTC_gbdt_featselect_{tag}_pv.npy"
        np.save(pv_path, pv_sub)
        print(f"  [save] {tag} meta_val 预测: {pv_path}")

        # 清理
        del models_sub, calibrator_sub, pt_sub, pv_sub
        gc.collect()

    # ============================================================
    # 5. 结果对比
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print(f"  特征选择结果对比汇总")
    print(f"{'=' * 70}")

    baseline = all_results["full"]["result"]
    print(f"\n  {'Model':<25s}  {'#Feat':>5s}  {'Acc':>6s}  {'Trades/D':>8s}  "
          f"{'AvgRet':>7s}  {'RetUp':>7s}  {'RetDn':>7s}  {'Conf':>6s}  "
          f"{'ΔAcc':>6s}")
    print(f"  {'-' * 85}")

    # 按 n_feat 排序展示
    order = ["full", "no_cross", "top50", "top30", "top20"]
    for tag in order:
        if tag not in all_results:
            continue
        r = all_results[tag]["result"]
        nf = all_results[tag]["n_feat"]
        label = all_results[tag]["tag"][:25]
        delta = (r["accuracy"] - baseline["accuracy"]) * 100
        print(f"  {label:<25s}  {nf:5d}  {r['accuracy']:.4f}  "
              f"{r['trades_per_day']:>8.1f}  {r['avg_ret_bps']:>7.1f}  "
              f"{r['avg_ret_up_bps']:>7.1f}  {r['avg_ret_dn_bps']:>7.1f}  "
              f"{r['conf_mean']:.4f}  {delta:+6.2f}pp")

    print(f"\n{'=' * 70}")
    print(f"  月度稳定性对比")
    print(f"{'=' * 70}")

    # 收集所有月份
    all_months = set()
    for tag in order:
        if tag in all_results:
            all_months.update(all_results[tag]["result"]["acc_by_month"].keys())
    all_months = sorted(all_months)

    # 表头
    header = f"  {'Month':<12s}"
    for tag in order:
        if tag in all_results:
            header += f"  {tag:>8s}"
    print(header)
    print(f"  {'-' * (12 + len([t for t in order if t in all_results]) * 10)}")

    for m in all_months:
        line = f"  {m[-7:]:<12s}"
        for tag in order:
            if tag in all_results:
                acc = all_results[tag]["result"]["acc_by_month"].get(m, -1)
                if acc >= 0:
                    line += f"  {acc:>8.4f}"
                else:
                    line += f"  {'':>8s}"
        print(line)

    # 找出最佳模型
    best_tag = max(all_results, key=lambda t: all_results[t]["result"]["accuracy"])
    best_acc = all_results[best_tag]["result"]["accuracy"]
    print(f"\n  最佳模型: {best_tag} ({all_results[best_tag]['tag']}) "
          f"acc={best_acc:.4f}")
    print(f"  (与完整模型对比: {best_tag} 优于 full "
          f"{'是' if best_acc > baseline['accuracy'] else '否'})")

    # 打印特征重要性文件
    imp_path = f"{config.DS_DIR}/BTC_gbdt_featselect_importance.txt"
    with open(imp_path, "w") as f:
        f.write(f"{'Rank':>4s}  {'Feature':<30s}  {'Avg Gain':>10s}  "
                f"{'Std Gain':>8s}  {'Type':>10s}\n")
        f.write(f"{'=' * 65}\n")
        for i in range(len(sorted_names)):
            ftype = "CROSS" if sorted_names[i] in CROSS_FEATURES else "BASE" if sorted_names[i] in FEATURES else "EXTRA"
            f.write(f"{i + 1:4d}  {sorted_names[i]:<30s}  "
                    f"{sorted_avg[i]:>10.2f}  {sorted_std[i]:>8.2f}  "
                    f"{ftype:>10s}\n")
    print(f"\n  [save] 特征重要性排序: {imp_path}")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)

    # 清理
    del ctx, extra_raw, models_full, calibrator_full
    for tag in all_results:
        del all_results[tag]["pt"]
    gc.collect()
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()