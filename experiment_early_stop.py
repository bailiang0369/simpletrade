#!/usr/bin/env python3
"""实验: 用 top1_acc 替代 AUC 做早停。

实验_enhanced 使用 metric="auc" 做早停, 但 AUC 仅 0.51-0.53 (近随机)。
用 top1_acc 做早停, 直接优化我们关心的目标指标。

注意: 这会导致更长的训练 (top1_acc 波动大), 但可能找到更好的 checkpoint。
"""

import os, sys, gc, time
import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

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
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]

EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us", "hour_cos_is_eu", "hour_sin_rvol_60",
    "consec_up", "consec_dn", "session_minutes", "hour_sin_hour_cos",
]

BAGGED_SEEDS = [42, 49, 56, 63, 70]


def topk_acc_metric(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs)
    k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


def compute_extra_raw(ctx):
    t0 = time.time()
    close = ctx.c; raw_ts = ctx.raw_ts; n = len(close)
    lr_1 = np.zeros(n, dtype=np.float64)
    lr_1[1:] = np.log(close[1:] / close[:-1])
    rvol_60 = np.zeros(n, dtype=np.float32)
    w = 60
    if n >= w:
        cs = np.cumsum(lr_1); cs2 = np.cumsum(lr_1 ** 2)
        roll_sum = cs[w:] - cs[:-w]; roll_sum2 = cs2[w:] - cs2[:-w]
        roll_mean = roll_sum / w; roll_var = np.maximum(roll_sum2 / w - roll_mean ** 2, 0)
        rvol_60[w:] = (np.sqrt(roll_var * w / (w - 1)) * 100).astype(np.float32)
    hour = (raw_ts % 86400) // 3600; minute_of_day = (raw_ts % 86400) // 60
    hour_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)
    hour_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)
    is_us = ((hour >= 13) & (hour < 21)).astype(np.float32)
    is_eu = ((hour >= 8) & (hour < 13)).astype(np.float32)
    consec_up = np.zeros(n, dtype=np.int32); consec_dn = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if close[i] > close[i - 1]: consec_up[i] = consec_up[i - 1] + 1
        else: consec_dn[i] = consec_dn[i - 1] + 1
    session_minutes = np.zeros(n, dtype=np.float32)
    for i in range(n):
        h = hour[i]; m = minute_of_day[i]
        if h < 8: session_minutes[i] = m
        elif h < 13: session_minutes[i] = m - 8 * 60
        elif h < 21: session_minutes[i] = m - 13 * 60
        else: session_minutes[i] = m - 21 * 60
    extra = {
        "hour_sin_is_us": (hour_sin * is_us).astype(np.float32),
        "hour_cos_is_eu": (hour_cos * is_eu).astype(np.float32),
        "hour_sin_rvol_60": (hour_sin * rvol_60).astype(np.float32),
        "consec_up": np.clip(consec_up.astype(np.float32) / 50.0, 0, 1),
        "consec_dn": np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1),
        "session_minutes": (session_minutes / 480.0).astype(np.float32),
        "hour_sin_hour_cos": (hour_sin * hour_cos).astype(np.float32),
    }
    print(f"  [extra] 耗时 {time.time() - t0:.1f}s", flush=True)
    return extra


def get_extra_for_mask(extra_raw, ctx, mask):
    ri = ctx.ds_to_raw[mask].astype(int)
    return np.column_stack([extra_raw[n][ri] for n in EXTRA_FEATURE_NAMES])


def main():
    symbol = "BTC"
    print(f"\n{'=' * 65}")
    print(f"  Top1-Acc 早停实验: {symbol} H30")
    print(f"  - 替代 AUC 早停, 直接用 top1_acc 做早停")
    print(f"  - 其余参数与 experiment_enhanced 完全一致")
    print(f"{'=' * 65}\n")

    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    extra_raw = compute_extra_raw(ctx)

    feats = list(FEATURES)
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # 早停集
    Xes = np.column_stack([ctx.X_subset(feats, esm), get_extra_for_mask(extra_raw, ctx, esm)])
    yes = ctx.label[esm].astype(np.float64)

    train_retf = ctx.retf("train")
    models = []

    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        max_train = 2_600_000
        if len(tr_idx) > max_train:
            keep = rng.choice(len(tr_idx), max_train, replace=False)
            tr_idx = tr_idx[keep]

        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True

        Xtr = np.column_stack([ctx.X_subset(feats, train_mask), get_extra_for_mask(extra_raw, ctx, train_mask)])
        ytr = ctx.label[train_mask].astype(np.float64)

        # 加权
        if len(tr_idx) < len(tr_idx_all):
            keep_local = np.where(train_mask[np.where(trm)[0]])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)

        # 坏时段
        train_hour = (ctx.ds_ts[train_mask] % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any():
            w[bad_hour] *= 2.0

        # 关键改动: metric="none", 只用 top1_acc 做早停
        params = dict(
            objective="binary", metric="none",
            learning_rate=0.02, num_leaves=127, max_depth=-1,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
            min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0,
            scale_pos_weight=1.0, num_threads=config.N_JOBS,
            verbosity=-1, seed=seed, min_data=1,
        )

        dtr = lgb.Dataset(Xtr, ytr, weight=w)
        des = lgb.Dataset(Xes, yes, reference=dtr)

        # 用 top1_acc 早停, 更高的 patience (500 轮) 因为 top1_acc 波动大
        m = lgb.train(
            params, dtr,
            num_boost_round=5000,
            valid_sets=[des],
            valid_names=['early_stop'],
            feval=topk_acc_metric,
            callbacks=[
                lgb.early_stopping(500, verbose=False, min_delta=1e-5),
                lgb.log_evaluation(0),
            ],
        )
        models.append(m)

        topk_val = m.best_score['early_stop'].get('top1_acc', -1)
        print(f"  [seed{seed}] best_iter={m.best_iteration} "
              f"top1_acc={topk_val:.4f} ({time.time() - t_start:.0f}s)", flush=True)

        del Xtr, dtr, ytr, w
        gc.collect()

    # 评估
    from sklearn.linear_model import LogisticRegression
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = np.column_stack([ctx.X_subset(feats, cal_mask), get_extra_for_mask(extra_raw, ctx, cal_mask)])
    n_cal = len(Xcal)
    R_cal = np.zeros((len(models), n_cal), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(Xcal, num_iteration=m.best_iteration)
        R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
    cal_p = R_cal.mean(axis=0)
    cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7)
    logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    calibrator.fit(logit_p, cal_y)
    print(f"  [calibrate] coef={calibrator.coef_[0][0]:.4f}", flush=True)
    del Xcal, R_cal, cal_p
    gc.collect()

    # 测试
    test_mask = ctx.split_rows["test"]
    Xte = np.column_stack([ctx.X_subset(feats, test_mask), get_extra_for_mask(extra_raw, ctx, test_mask)])
    n_te = len(Xte)
    R_te = np.zeros((len(models), n_te), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(Xte, num_iteration=m.best_iteration)
        R_te[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
    pt_raw = R_te.mean(axis=0)
    logit_te = np.clip(pt_raw, 1e-7, 1 - 1e-7)
    logit_te = np.log(logit_te / (1 - logit_te)).reshape(-1, 1)
    pt = calibrator.predict_proba(logit_te)[:, 1].astype(np.float32)
    del Xte, R_te
    gc.collect()

    # 评估
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"\n{'=' * 65}")
    print(f"  Top1-Acc 早停 测试集结果 (top {config.COVERAGE * 100:.0f}%)")
    print(f"{'=' * 65}")
    print(f"  准确率:       {r['accuracy']:.4f}")
    print(f"  交易次数:     {r['k']}")
    print(f"  交易/天:      {r['trades_per_day']:.1f}")
    print(f"  平均收益:     {r['avg_ret_bps']:.1f} bps")

    # 对比 AUC 早停基线
    base_path = f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy"
    if os.path.exists(base_path):
        pt_base = np.load(base_path).astype(np.float64)
        r_base = evaluate_topk(pt_base, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
        print(f"\n  AUC早停基线:  {r_base['accuracy']:.4f}")
        print(f"  Top1早停:     {r['accuracy']:.4f}")
        print(f"  提升:         {(r['accuracy'] - r_base['accuracy']) * 100:+.2f} pp")

    if r['acc_by_month']:
        print(f"\n  月度明细:")
        for m in sorted(r['acc_by_month'].keys()):
            print(f"    {m[-7:]}: acc={r['acc_by_month'][m]:.4f}  n={r['n_by_month'][m]:4d}")

    # 保存
    save_path = f"{config.DS_DIR}/BTC_gbdt_top1stop_pt.npy"
    np.save(save_path, pt)
    print(f"\n  [save] 预测保存至: {save_path}")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)
    del ctx, extra_raw, models, calibrator
    gc.collect()


if __name__ == "__main__":
    main()