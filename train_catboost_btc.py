#!/usr/bin/env python3
"""单独训练 CatBoost (BTC) 并保存模型。"""
import os, sys, gc, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from sklearn.linear_model import LogisticRegression

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

os.makedirs(MODEL_DIR, exist_ok=True)
t0 = time.time()
ctx = AssetContext("BTC", horizon=30)
extra_raw = compute_extra_raw(ctx)

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
    
    from catboost import CatBoostClassifier, Pool
    train_pool = Pool(Xtr, ytr, weight=w); eval_pool = Pool(Xes, yes)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8,
        l2_leaf_reg=1.0, random_seed=seed, task_type="CPU",
        thread_count=config.N_JOBS, verbose=False, early_stopping_rounds=200,
        loss_function="Logloss")
    model.fit(train_pool, eval_set=eval_pool, verbose_eval=False)
    model.save_model(f"{MODEL_DIR}/catboost_seed{seed}.cbm")
    print(f"  [catboost] seed{seed} best_iter={model.best_iteration_} ({time.time()-seed_t0:.0f}s)", flush=True)
    del Xtr, train_pool, ytr, w, model; gc.collect()

# 校准
cal_mask = ctx.split_rows["meta_val"]
Xcal = get_X(ctx, extra_raw, cal_mask)
n_cal = len(Xcal); R_cal = np.zeros((5, n_cal), dtype=np.float64)
from catboost import CatBoost
for i, seed in enumerate(BAGGED_SEEDS):
    m = CatBoost(); m.load_model(f"{MODEL_DIR}/catboost_seed{seed}.cbm")
    p = m.predict(Xcal, prediction_type="Probability")[:, 1]
    R_cal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_cal - 1)
    del m; gc.collect()
cal_p = R_cal.mean(axis=0); cal_y = ctx.label[cal_mask]
calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
logit_p = np.clip(cal_p, 1e-7, 1 - 1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
calibrator.fit(logit_p, cal_y)
import joblib; joblib.dump(calibrator, f"{MODEL_DIR}/catboost_calib.joblib")
print(f"  [catboost] Platt校准: coef={calibrator.coef_[0][0]:.4f} done in {time.time()-t0:.0f}s", flush=True)
del ctx, extra_raw, Xes, Xcal; gc.collect()
print(f"完成.", flush=True)