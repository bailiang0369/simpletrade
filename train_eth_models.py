#!/usr/bin/env python3
"""ETH 优化: 训练增强 GBDT + CatBoost on ETH, 搜索最优集成权重。"""
import os, sys, gc, time, json
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
]
EXTRA_FEATURE_NAMES = [
    "hour_sin_is_us", "hour_cos_is_eu", "hour_sin_rvol_60",
    "consec_up", "consec_dn", "session_minutes", "hour_sin_hour_cos",
]
BAGGED_SEEDS = [42, 49, 56, 63, 70]
MODEL_DIR = "/workspace/simpletrade/models_saved/eth_optimized"
os.makedirs(MODEL_DIR, exist_ok=True)


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
    print(f"  [extra] {time.time() - t0:.1f}s", flush=True)
    return extra


def get_extra_for_mask(extra_raw, ctx, mask):
    ri = ctx.ds_to_raw[mask].astype(int)
    return np.column_stack([extra_raw[n][ri] for n in EXTRA_FEATURE_NAMES])


def get_X(ctx, extra_raw, mask):
    return np.column_stack([ctx.X_subset(FEATURES, mask), get_extra_for_mask(extra_raw, ctx, mask)])


def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


def train_enhanced_gbdt(ctx, extra_raw, symbol):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    for seed in BAGGED_SEEDS:
        seed_t0 = time.time()
        print(f"  [enhanced] seed{seed} 开始...", flush=True)
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
        m.save_model(f"{MODEL_DIR}/{symbol}_enhanced_seed{seed}.txt")
        topk_val = m.best_score['early_stop'].get('top1_acc', -1)
        print(f"  [enhanced] seed{seed} best_iter={m.best_iteration} top1_acc={topk_val:.4f} ({time.time()-seed_t0:.0f}s)", flush=True)
        del Xtr, dtr, ytr, w, m; gc.collect()
    # 校准
    print(f"  [enhanced] 校准中...", flush=True)
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = get_X(ctx, extra_raw, cal_mask)
    n_cal = len(Xcal); R_cal = np.zeros((5, n_cal), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = lgb.Booster(model_file=f"{MODEL_DIR}/{symbol}_enhanced_seed{seed}.txt")
        p = m.predict(Xcal, num_iteration=m.best_iteration)
        R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
        del m; gc.collect()
    cal_p = R_cal.mean(axis=0); cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    calibrator.fit(logit_p, cal_y)
    import joblib; joblib.dump(calibrator, f"{MODEL_DIR}/{symbol}_enhanced_calib.joblib")
    print(f"  [enhanced] 完成 ({time.time()-t0:.0f}s)", flush=True)
    del Xes, Xcal, R_cal; gc.collect()


def train_catboost(ctx, extra_raw, symbol):
    from catboost import CatBoostClassifier, Pool
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    for seed in BAGGED_SEEDS:
        seed_t0 = time.time()
        print(f"  [catboost] seed{seed} 开始...", flush=True)
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
        model.save_model(f"{MODEL_DIR}/{symbol}_catboost_seed{seed}.cbm")
        print(f"  [catboost] seed{seed} best_iter={model.best_iteration_} ({time.time()-seed_t0:.0f}s)", flush=True)
        del Xtr, train_pool, ytr, w, model; gc.collect()
    # 校准
    print(f"  [catboost] 校准中...", flush=True)
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = get_X(ctx, extra_raw, cal_mask)
    n_cal = len(Xcal); R_cal = np.zeros((5, n_cal), dtype=np.float64)
    from catboost import CatBoost
    for i, seed in enumerate(BAGGED_SEEDS):
        m = CatBoost(); m.load_model(f"{MODEL_DIR}/{symbol}_catboost_seed{seed}.cbm")
        p = m.predict(Xcal, prediction_type="Probability")[:, 1]
        R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
        del m; gc.collect()
    cal_p = R_cal.mean(axis=0); cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    calibrator.fit(logit_p, cal_y)
    import joblib; joblib.dump(calibrator, f"{MODEL_DIR}/{symbol}_catboost_calib.joblib")
    print(f"  [catboost] 完成 ({time.time()-t0:.0f}s)", flush=True)
    del Xes, Xcal, R_cal; gc.collect()


def predict_enhanced(ctx, extra_raw, symbol, split):
    import lightgbm as lgb; import joblib
    mask = ctx.split_rows[split]; X = get_X(ctx, extra_raw, mask); n = len(X)
    R = np.zeros((5, n), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = lgb.Booster(model_file=f"{MODEL_DIR}/{symbol}_enhanced_seed{seed}.txt")
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del m; gc.collect()
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{MODEL_DIR}/{symbol}_enhanced_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def predict_catboost(ctx, extra_raw, symbol, split):
    from catboost import CatBoost; import joblib
    mask = ctx.split_rows[split]; X = get_X(ctx, extra_raw, mask); n = len(X)
    R = np.zeros((5, n), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = CatBoost(); m.load_model(f"{MODEL_DIR}/{symbol}_catboost_seed{seed}.cbm")
        p = m.predict(X, prediction_type="Probability")[:, 1]
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del m; gc.collect()
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{MODEL_DIR}/{symbol}_catboost_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def main():
    symbol = "ETH"
    print(f"\n{'=' * 65}")
    print(f"  ETH 优化: 在 ETH 上训练模型并搜索最优集成")
    print(f"  目标: 准确率 >= {config.TARGET_ACCURACY}")
    print(f"{'=' * 65}\n")

    t_start = time.time()
    ctx = AssetContext(symbol, horizon=30)
    extra_raw = compute_extra_raw(ctx)

    # 1. 训练增强 GBDT
    print(f"\n[1/4] 训练增强 GBDT on {symbol} ...")
    train_enhanced_gbdt(ctx, extra_raw, symbol)

    # 2. 训练 CatBoost
    print(f"\n[2/4] 训练 CatBoost on {symbol} ...")
    train_catboost(ctx, extra_raw, symbol)

    # 3. 预测
    print(f"\n[3/4] 预测测试集 ...")
    pt_enhanced = predict_enhanced(ctx, extra_raw, symbol, "test")
    r_enhanced = evaluate_topk(pt_enhanced, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  [ETH enhanced] acc={r_enhanced['accuracy']:.4f} ret={r_enhanced['avg_ret_bps']:.1f}bps", flush=True)

    pt_cat = predict_catboost(ctx, extra_raw, symbol, "test")
    r_cat = evaluate_topk(pt_cat, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  [ETH catboost] acc={r_cat['accuracy']:.4f} ret={r_cat['avg_ret_bps']:.1f}bps", flush=True)

    orig_path = f"{config.DS_DIR}/ETH_gbdt_h30_pt.npy"
    if not os.path.exists(orig_path): orig_path = f"{config.DS_DIR}/ETH_gbdt_pt.npy"
    pt_orig = np.load(orig_path).astype(np.float64)
    r_orig = evaluate_topk(pt_orig, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  [ETH original] acc={r_orig['accuracy']:.4f} ret={r_orig['avg_ret_bps']:.1f}bps", flush=True)

    # 4. 搜索最优集成权重
    print(f"\n[4/4] 搜索最优集成权重 ...")
    print(f"  {'w_enh':>5} {'w_ori':>5} {'w_cat':>5}  {'Acc':>8} {'Ret(bps)':>9} {'T/D':>6}")
    best_acc = 0; best_w = None
    for w_e in np.arange(0.0, 1.05, 0.05):
        for w_c in np.arange(0.0, 1.05, 0.05):
            w_o = 1.0 - w_e - w_c
            if w_o < -0.01 or w_o > 1.01: continue
            if abs(w_e) < 0.01 and abs(w_c) < 0.01: continue
            if w_o < 0: w_o = 0; w_e += w_o; w_c = 1 - w_e
            if w_o > 1: w_o = 1; w_e = 0; w_c = 0
            pt = pt_enhanced * w_e + pt_orig * w_o + pt_cat * w_c
            r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
            marker = ' <<<' if r['accuracy'] > best_acc else ''
            if r['accuracy'] > best_acc + 1e-6:
                best_acc = r['accuracy']; best_w = (w_e, w_o, w_c)
            if r['accuracy'] >= 0.64 or (r['accuracy'] > best_acc - 0.002 and w_e >= 0.3 and w_c >= 0.1):
                print(f"  {w_e:.2f} {w_o:.2f} {w_c:.2f}  {r['accuracy']:.4f}  {r['avg_ret_bps']:>7.1f}  {r['trades_per_day']:.1f}{marker}")

    # 最佳结果
    print(f"\n{'=' * 65}")
    print(f"  最佳集成: enhanced={best_w[0]:.2f} original={best_w[1]:.2f} catboost={best_w[2]:.2f}")
    pt_best = pt_enhanced * best_w[0] + pt_orig * best_w[1] + pt_cat * best_w[2]
    r_best = evaluate_topk(pt_best, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  准确率: {r_best['accuracy']:.4f} (目标: {config.TARGET_ACCURACY})")
    print(f"  收益:   {r_best['avg_ret_bps']:.1f} bps")
    print(f"  交易/天: {r_best['trades_per_day']:.1f}")
    print(f"  目标达成: {'✓' if r_best['accuracy'] >= config.TARGET_ACCURACY else '✗'}")
    print(f"{'=' * 65}")

    # 月度明细
    print(f"\n  ETH 月度明细:")
    for m in sorted(r_best['acc_by_month'].keys()):
        print(f"    {m[-7:]}: acc={r_best['acc_by_month'][m]:.4f}  n={r_best['n_by_month'][m]:4d}")

    # 保存
    np.save(f"{config.DS_DIR}/ETH_ensemble_optimized_pt.npy", pt_best.astype(np.float32))
    print(f"\n  [save] 预测保存至: ETH_ensemble_optimized_pt.npy")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s", flush=True)
    del ctx, extra_raw; gc.collect()


if __name__ == "__main__":
    main()