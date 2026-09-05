#!/usr/bin/env python3
"""使用已训练好的 BTC 模型预测 ETH，然后集成评估。"""
import os, sys, gc, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk, stability_report

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
MODEL_DIR = "/workspace/simpletrade/models_saved/eth_validate"


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


def predict_enhanced(ctx, extra_raw, split):
    import lightgbm as lgb
    import joblib
    mask = ctx.split_rows[split]
    X = get_X(ctx, extra_raw, mask); n = len(X)
    R = np.zeros((5, n), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = lgb.Booster(model_file=f"{MODEL_DIR}/enhanced_seed{seed}.txt")
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del m; gc.collect()
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{MODEL_DIR}/enhanced_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7)
    logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def predict_catboost(ctx, extra_raw, split):
    from catboost import CatBoost
    import joblib
    mask = ctx.split_rows[split]
    X = get_X(ctx, extra_raw, mask); n = len(X)
    R = np.zeros((5, n), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = CatBoost(); m.load_model(f"{MODEL_DIR}/catboost_seed{seed}.cbm")
        p = m.predict(X, prediction_type="Probability")[:, 1]
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del m; gc.collect()
    p_raw = R.mean(axis=0)
    calibrator = joblib.load(f"{MODEL_DIR}/catboost_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    p = calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
    del X, R; gc.collect()
    return np.asarray(p, dtype=np.float32)


def main():
    print(f"\n{'=' * 65}")
    print(f"  ETH 跨币种验证: 用 BTC 训练模型 → ETH 预测")
    print(f"  BTC 集成准确率: 0.6516 (目标 >= {config.TARGET_ACCURACY})")
    print(f"  跨币种差异需 <= {config.CROSS_ASSET_MAX_DELTA*100:.0f}pp")
    print(f"{'=' * 65}\n")

    t_start = time.time()

    # 加载 ETH 数据
    print("[1/2] 加载 ETH 数据并计算特征...")
    ctx_eth = AssetContext("ETH", horizon=30)
    extra_eth = compute_extra_raw(ctx_eth)

    # 预测
    print("[2/2] 预测并集成...")
    pt_enhanced_eth = predict_enhanced(ctx_eth, extra_eth, "test")
    r_enhanced_eth = evaluate_topk(pt_enhanced_eth, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))
    print(f"  [ETH enhanced] acc={r_enhanced_eth['accuracy']:.4f} "
          f"ret={r_enhanced_eth['avg_ret_bps']:.1f}bps", flush=True)

    pt_cat_eth = predict_catboost(ctx_eth, extra_eth, "test")
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

    # 集成
    w_e, w_o, w_c = 0.5, 0.4, 0.1
    pt_ens = pt_enhanced_eth * w_e + pt_orig_eth * w_o + pt_cat_eth * w_c
    r_ens = evaluate_topk(pt_ens, ctx_eth.y("test"), ctx_eth.retf("test"), ctx_eth.times("test"))

    # 输出结果
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

    # 保存
    np.save(f"{config.DS_DIR}/ETH_ensemble_pt.npy", pt_ens.astype(np.float32))
    print(f"\n  [save] ETH 集成预测保存至: ETH_ensemble_pt.npy")

    # 月度明细
    print(f"\n  ETH 月度明细:")
    for m in sorted(r_ens['acc_by_month'].keys()):
        print(f"    {m[-7:]}: acc={r_ens['acc_by_month'][m]:.4f}  n={r_ens['n_by_month'][m]:4d}")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.0f}s", flush=True)
    del ctx_eth, extra_eth; gc.collect()
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()