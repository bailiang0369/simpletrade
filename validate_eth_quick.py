#!/usr/bin/env python3
"""ETH 跨币种验证: 用 BTC 训练好的模型预测 ETH 数据。

流程:
  1. 增强 GBDT (BTC训练 → ETH预测)
  2. CatBoost (BTC训练 → ETH预测)
  3. 原始 GBDT (已有 ETH 预测)
  4. 集成 → 评估跨币种差异
"""

import os, sys, gc, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk, stability_report

# 特征 (与 experiment_enhanced.py 一致, 不含交叉特征)
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
]
EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us", "hour_cos_is_eu", "hour_sin_rvol_60",
    "consec_up", "consec_dn", "session_minutes", "hour_sin_hour_cos",
]
BAGGED_SEEDS = [42, 49, 56, 63, 70]


def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
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


def get_X(ctx, extra_raw, mask):
    X_base = ctx.X_subset(FEATURES, mask)
    X_extra = get_extra_for_mask(extra_raw, ctx, mask)
    return np.column_stack([X_base, X_extra])


def train_enhanced_save_models(ctx, extra_raw, model_dir):
    """训练增强 GBDT (BTC), 保存模型文件。"""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    models = []
    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > 2_600_000:
            keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
            tr_idx = tr_idx[keep]
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = get_X(ctx, extra_raw, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
        if len(tr_idx) < len(tr_idx_all):
            keep_local = np.where(train_mask[tr_idx_all])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)
        train_hour = (ctx.ds_ts[train_mask] % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any(): w[bad_hour] *= 2.0
        params = dict(objective="binary", metric="auc", learning_rate=0.02,
                       num_leaves=127, max_depth=-1, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
                       lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                       num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
                      valid_names=['early_stop'], feval=topk_acc_eval,
                      callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                 lgb.log_evaluation(0)])
        m.save_model(f"{model_dir}/enhanced_seed{seed}.txt")
        models.append(m)
        topk_val = m.best_score['early_stop'].get('top1_acc', -1)
        print(f"  [enhanced] seed{seed} best_iter={m.best_iteration} "
              f"top1_acc={topk_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del Xtr, dtr, ytr, w; gc.collect()
    # Platt 校准
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = get_X(ctx, extra_raw, cal_mask)
    n_cal = len(Xcal); R_cal = np.zeros((5, n_cal), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(Xcal, num_iteration=m.best_iteration)
        R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
    cal_p = R_cal.mean(axis=0); cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7)
    logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    calibrator.fit(logit_p, cal_y)
    import joblib
    joblib.dump(calibrator, f"{model_dir}/enhanced_calib.joblib")
    print(f"  [enhanced] Platt校准: coef={calibrator.coef_[0][0]:.4f} done in {time.time()-t0:.0f}s", flush=True)
    del Xcal, Xes, R_cal; gc.collect()
    return models, calibrator


def predict_enhanced(ctx, extra_raw, model_dir, split):
    """用 BTC 训练的增强 GBDT 模型预测。"""
    import lightgbm as lgb
    import joblib
    mask = ctx.split_rows[split]
    X = get_X(ctx, extra_raw, mask); n = len(X)
    models = []
    for seed in BAGGED_SEEDS:
        m = lgb.Booster(model_file=f"{model_dir}/enhanced_seed{seed}.txt")
        models.append(m)
    R = np.zeros((5, n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{model_dir}/enhanced_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7)
    logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, models, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def train_catboost_save_models(ctx, extra_raw, model_dir):
    """训练 CatBoost (BTC), 保存模型。"""
    from catboost import CatBoostClassifier, Pool
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    models = []
    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > 2_600_000:
            keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
            tr_idx = tr_idx[keep]
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = get_X(ctx, extra_raw, train_mask); ytr = ctx.label[train_mask].astype(np.int32)
        if len(tr_idx) < len(tr_idx_all):
            keep_local = np.where(train_mask[tr_idx_all])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)
        train_pool = Pool(Xtr, ytr, weight=w); eval_pool = Pool(Xes, yes)
        model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8,
            l2_leaf_reg=1.0, random_seed=seed, task_type="CPU",
            thread_count=config.N_JOBS, verbose=False, early_stopping_rounds=200,
            loss_function="Logloss")
        model.fit(train_pool, eval_set=eval_pool, verbose_eval=False)
        model.save_model(f"{model_dir}/catboost_seed{seed}.cbm")
        models.append(model)
        print(f"  [catboost] seed{seed} best_iter={model.best_iteration_} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xtr, train_pool, ytr, w; gc.collect()
    # 校准
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = get_X(ctx, extra_raw, cal_mask)
    n_cal = len(Xcal); R_cal = np.zeros((5, n_cal), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(Xcal, prediction_type="Probability")[:, 1]
        R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
    cal_p = R_cal.mean(axis=0); cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    calibrator.fit(logit_p, cal_y)
    import joblib; joblib.dump(calibrator, f"{model_dir}/catboost_calib.joblib")
    print(f"  [catboost] Platt校准: coef={calibrator.coef_[0][0]:.4f} done in {time.time()-t0:.0f}s", flush=True)
    del Xcal, Xes, R_cal; gc.collect()
    return models, calibrator


def predict_catboost(ctx, extra_raw, model_dir, split):
    """用 BTC 训练的 CatBoost 模型预测 ETH。"""
    from catboost import CatBoost
    import joblib
    mask = ctx.split_rows[split]
    X = get_X(ctx, extra_raw, mask); n = len(X)
    models = []
    for seed in BAGGED_SEEDS:
        m = CatBoost()
        m.load_model(f"{model_dir}/catboost_seed{seed}.cbm")
        models.append(m)
    R = np.zeros((5, n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, prediction_type="Probability")[:, 1]
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{model_dir}/catboost_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, models, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def main():
    print(f"\n{'=' * 65}")
    print(f"  ETH 跨币种验证: 用 BTC 训练模型 → ETH 预测")
    print(f"  BTC 集成准确率: 0.6516 (目标 >= {config.TARGET_ACCURACY})")
    print(f"  跨币种差异需 <= {config.CROSS_ASSET_MAX_DELTA*100:.0f}pp")
    print(f"{'=' * 65}\n")

    model_dir = "/workspace/simpletrade/models_saved/eth_validate"
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()

    # === 第一步: 训练 BTC 模型 ===
    print("[1/3] 在 BTC 上训练增强 GBDT + CatBoost ...")
    ctx_btc = AssetContext("BTC", horizon=30)
    extra_btc = compute_extra_raw(ctx_btc)
    m_enhanced, cal_enhanced = train_enhanced_save_models(ctx_btc, extra_btc, model_dir)
    print(f"  [BTC enhanced] 训练完成", flush=True)
    m_cat, cal_cat = train_catboost_save_models(ctx_btc, extra_btc, model_dir)
    print(f"  [BTC catboost] 训练完成", flush=True)
    del ctx_btc, extra_btc, m_enhanced, m_cat; gc.collect()

    # === 第二步: 用 BTC 模型预测 ETH ===
    print("\n[2/3] 用 BTC 模型预测 ETH 数据 ...")
    ctx_eth = AssetContext("ETH", horizon=30)
    extra_eth = compute_extra_raw(ctx_eth)

    pt_enhanced_eth = predict_enhanced(ctx_eth, extra_eth, model_dir, "test")
    r_enhanced_eth = evaluate_topk(pt_enhanced_eth, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))
    print(f"  [ETH enhanced] acc={r_enhanced_eth['accuracy']:.4f} "
          f"ret={r_enhanced_eth['avg_ret_bps']:.1f}bps", flush=True)

    pt_cat_eth = predict_catboost(ctx_eth, extra_eth, model_dir, "test")
    r_cat_eth = evaluate_topk(pt_cat_eth, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))
    print(f"  [ETH catboost] acc={r_cat_eth['accuracy']:.4f} "
          f"ret={r_cat_eth['avg_ret_bps']:.1f}bps", flush=True)

    # 加载原始 GBDT ETH 预测
    orig_path = f"{config.DS_DIR}/ETH_gbdt_h30_pt.npy"
    if not os.path.exists(orig_path):
        orig_path = f"{config.DS_DIR}/ETH_gbdt_pt.npy"
    pt_orig_eth = np.load(orig_path).astype(np.float64)
    r_orig_eth = evaluate_topk(pt_orig_eth, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))
    print(f"  [ETH original] acc={r_orig_eth['accuracy']:.4f} "
          f"ret={r_orig_eth['avg_ret_bps']:.1f}bps", flush=True)

    # === 第三步: 集成 ===
    print("\n[3/3] 应用集成权重 ...")
    w_e, w_o, w_c = 0.5, 0.4, 0.1
    pt_ens = pt_enhanced_eth * w_e + pt_orig_eth * w_o + pt_cat_eth * w_c
    r_ens = evaluate_topk(pt_ens, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))

    # === 输出 ===
    print(f"\n{'=' * 65}")
    print(f"  ETH 跨币种验证结果")
    print(f"{'=' * 65}")
    print(f"  {'Model':<20} {'Acc':>8} {'Ret(bps)':>9} {'T/D':>6}")
    print(f"  {'-' * 48}")
    print(f"  {'enhanced_gbdt':<20} {r_enhanced_eth['accuracy']:>8.4f} "
          f"{r_enhanced_eth['avg_ret_bps']:>9.1f} {r_enhanced_eth['trades_per_day']:>6.1f}")
    print(f"  {'original_gbdt':<20} {r_orig_eth['accuracy']:>8.4f} "
          f"{r_orig_eth['avg_ret_bps']:>9.1f} {r_orig_eth['trades_per_day']:>6.1f}")
    print(f"  {'catboost':<20} {r_cat_eth['accuracy']:>8.4f} "
          f"{r_cat_eth['avg_ret_bps']:>9.1f} {r_cat_eth['trades_per_day']:>6.1f}")
    print(f"  {'-' * 48}")
    print(f"  {'ENSEMBLE':<20} {r_ens['accuracy']:>8.4f} "
          f"{r_ens['avg_ret_bps']:>9.1f} {r_ens['trades_per_day']:>6.1f}")

    # 跨币种对比
    print(f"\n{'=' * 65}")
    print(f"  跨币种对比")
    print(f"{'=' * 65}")
    ctx_btc2 = AssetContext("BTC", horizon=30)
    btc_enh = np.load(f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy").astype(np.float64)
    btc_orig = np.load(f"{config.DS_DIR}/BTC_gbdt_h30_pt.npy").astype(np.float64)
    btc_cat = np.load(f"{config.DS_DIR}/BTC_catboost_pt.npy").astype(np.float64)
    pt_btc_ens = btc_enh * 0.5 + btc_orig * 0.4 + btc_cat * 0.1
    r_btc = evaluate_topk(pt_btc_ens, ctx_btc2.y("test"), ctx_btc2.retf("test"), ctx_btc2.times("test"))
    delta, ok = stability_report(r_btc["accuracy"], r_ens["accuracy"])
    print(f"  BTC 集成准确率:   {r_btc['accuracy']:.4f}")
    print(f"  ETH 集成准确率:   {r_ens['accuracy']:.4f}")
    print(f"  差异:             {delta*100:.2f}pp")
    print(f"  要求:             <= {config.CROSS_ASSET_MAX_DELTA*100:.0f}pp")
    print(f"  结果:             {'✓ 通过' if ok else '✗ 未通过'}")
    del ctx_btc2; gc.collect()

    # 保存
    np.save(f"{config.DS_DIR}/ETH_ensemble_pt.npy", pt_ens.astype(np.float32))
    print(f"\n  [save] ETH 集成预测: ETH_ensemble_pt.npy")

    print(f"\n  ETH 月度明细:")
    for m in sorted(r_ens['acc_by_month'].keys()):
        print(f"    {m[-7:]}: acc={r_ens['acc_by_month'][m]:.4f}  "
              f"n={r_ens['n_by_month'][m]:4d}")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)
    del ctx_eth, extra_eth; gc.collect()


if __name__ == "__main__":
    main()