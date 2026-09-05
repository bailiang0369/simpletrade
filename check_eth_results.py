#!/usr/bin/env python3
"""验证: ETH 本地训练 vs BTC 训练模型在 ETH 上的表现对比。"""
import os, sys, gc, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

FEATURES = [
    'lr_5', 'lr_15', 'lr_30', 'lr_120', 'lr_240', 'mom_60',
    'z_10', 'z_30', 'z_60', 'z_120',
    'rvol_30', 'rvol_60', 'rvol_ratio_60_5', 'rvol_z_60', 'rvol_dir',
    'pos_30', 'pos_60', 'pos_120', 'pos_240',
    'dd_240', 'ru_240',
    'hh_dd_60', 'll_ru_60', 'body_pos_60',
    'body_ratio', 'up_wick', 'lo_wick', 'ngreen_10', 'gap', 'max_range_30',
    'tbr_z_30', 'cvd_30', 'cvd_60',
    'buyvol_strength_30', 'tb_act_60', 'ts_act_60', 'tb_acc_30',
    'cvd_dir_30', 'tbr_hi_60', 'lr_skew_60', 'up_body_ratio_30', 'mom_align_30_240',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_us', 'is_eu', 'ret_day',
]
EXTRA_FEATURE_NAMES = [
    'hour_sin_is_us', 'hour_cos_is_eu', 'hour_sin_rvol_60',
    'consec_up', 'consec_dn', 'session_minutes', 'hour_sin_hour_cos',
]
BAGGED_SEEDS = [42, 49, 56, 63, 70]

def compute_extra(ctx):
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
    extra = {k: np.zeros(n, dtype=np.float32) for k in EXTRA_FEATURE_NAMES}
    extra['hour_sin_is_us'] = (hour_sin * is_us).astype(np.float32)
    extra['hour_cos_is_eu'] = (hour_cos * is_eu).astype(np.float32)
    extra['hour_sin_rvol_60'] = (hour_sin * rvol_60).astype(np.float32)
    extra['consec_up'] = np.clip(consec_up.astype(np.float32) / 50.0, 0, 1)
    extra['consec_dn'] = np.clip(consec_dn.astype(np.float32) / 50.0, 0, 1)
    extra['session_minutes'] = (session_minutes / 480.0).astype(np.float32)
    extra['hour_sin_hour_cos'] = (hour_sin * hour_cos).astype(np.float32)
    return extra

def get_X(ctx, extra, mask):
    ri = ctx.ds_to_raw[mask].astype(int)
    Xb = ctx.X_subset(FEATURES, mask)
    Xe = np.column_stack([extra[n][ri] for n in EXTRA_FEATURE_NAMES])
    return np.column_stack([Xb, Xe])

# 加载ETH数据
ctx = AssetContext('ETH', horizon=30)
extra = compute_extra(ctx)
test_mask = ctx.split_rows['test']
X_test = get_X(ctx, extra, test_mask)
y_te = ctx.y('test')
retf_te = ctx.retf('test')
ts_te = ctx.times('test')
n_te = len(X_test)
print(f'ETH 测试集: {n_te} 行')

import lightgbm as lgb
import joblib
from catboost import CatBoost

btc_dir = '/workspace/simpletrade/models_saved/eth_validate'
eth_dir = '/workspace/simpletrade/models_saved/eth_optimized'

# 方案A: BTC训练 → ETH预测
print('\n=== 方案A: BTC训练模型 → ETH预测 ===')

R = np.zeros((5, n_te), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = lgb.Booster(model_file=f'{btc_dir}/enhanced_seed{seed}.txt')
    p = m.predict(X_test, num_iteration=m.best_iteration)
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
    del m
p_raw = R.mean(axis=0)
cal = joblib.load(f'{btc_dir}/enhanced_calib.joblib')
logit = np.clip(p_raw, 1e-7, 1-1e-7); logit = np.log(logit/(1-logit)).reshape(-1,1)
pt_enh_btc = cal.predict_proba(logit)[:, 1].astype(np.float64)
r = evaluate_topk(pt_enh_btc, y_te, retf_te, ts_te)
print(f'  BTC-enhanced → ETH: acc={r["accuracy"]:.4f} ret={r["avg_ret_bps"]:.1f}bps tpd={r["trades_per_day"]:.1f}')

R = np.zeros((5, n_te), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = CatBoost(); m.load_model(f'{btc_dir}/catboost_seed{seed}.cbm')
    p = m.predict(X_test, prediction_type='Probability')[:, 1]
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
    del m
p_raw = R.mean(axis=0)
cal = joblib.load(f'{btc_dir}/catboost_calib.joblib')
logit = np.clip(p_raw, 1e-7, 1-1e-7); logit = np.log(logit/(1-logit)).reshape(-1,1)
pt_cat_btc = cal.predict_proba(logit)[:, 1].astype(np.float64)
r = evaluate_topk(pt_cat_btc, y_te, retf_te, ts_te)
print(f'  BTC-catboost → ETH: acc={r["accuracy"]:.4f} ret={r["avg_ret_bps"]:.1f}bps tpd={r["trades_per_day"]:.1f}')

# 方案B: ETH本地训练
print('\n=== 方案B: ETH本地训练模型 → ETH预测 ===')

R = np.zeros((5, n_te), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = lgb.Booster(model_file=f'{eth_dir}/ETH_enhanced_seed{seed}.txt')
    p = m.predict(X_test, num_iteration=m.best_iteration)
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
    del m
p_raw = R.mean(axis=0)
cal = joblib.load(f'{eth_dir}/ETH_enhanced_calib.joblib')
logit = np.clip(p_raw, 1e-7, 1-1e-7); logit = np.log(logit/(1-logit)).reshape(-1,1)
pt_enh_eth = cal.predict_proba(logit)[:, 1].astype(np.float64)
r = evaluate_topk(pt_enh_eth, y_te, retf_te, ts_te)
print(f'  ETH-enhanced → ETH: acc={r["accuracy"]:.4f} ret={r["avg_ret_bps"]:.1f}bps tpd={r["trades_per_day"]:.1f}')

R = np.zeros((5, n_te), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = CatBoost(); m.load_model(f'{eth_dir}/ETH_catboost_seed{seed}.cbm')
    p = m.predict(X_test, prediction_type='Probability')[:, 1]
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
    del m
p_raw = R.mean(axis=0)
cal = joblib.load(f'{eth_dir}/ETH_catboost_calib.joblib')
logit = np.clip(p_raw, 1e-7, 1-1e-7); logit = np.log(logit/(1-logit)).reshape(-1,1)
pt_cat_eth = cal.predict_proba(logit)[:, 1].astype(np.float64)
r = evaluate_topk(pt_cat_eth, y_te, retf_te, ts_te)
print(f'  ETH-catboost → ETH: acc={r["accuracy"]:.4f} ret={r["avg_ret_bps"]:.1f}bps tpd={r["trades_per_day"]:.1f}')

# 方案C: 原始GBDT
print('\n=== 方案C: 原始GBDT (ETH已有) ===')
pt_orig = np.load(f'{config.DS_DIR}/ETH_gbdt_h30_pt.npy').astype(np.float64)
r = evaluate_topk(pt_orig, y_te, retf_te, ts_te)
print(f'  ETH-original:      acc={r["accuracy"]:.4f} ret={r["avg_ret_bps"]:.1f}bps tpd={r["trades_per_day"]:.1f}')

# 数据量对比
print('\n=== 数据量对比 ===')
ctx_btc = AssetContext('BTC', horizon=30)
print(f'  BTC 训练集: {ctx_btc.split_rows["train"].sum():,} 行')
print(f'  ETH 训练集: {ctx.split_rows["train"].sum():,} 行')
print(f'  BTC 总行数: {len(ctx_btc.ds_ts):,}')
print(f'  ETH 总行数: {len(ctx.ds_ts):,}')
print(f'  比例: ETH/BTC = {len(ctx.ds_ts)/len(ctx_btc.ds_ts):.2%}')

# 特征验证
print(f'\n=== 特征列验证 ===')
btc_cols = set(ctx_btc.ds.columns)
eth_cols = set(ctx.ds.columns)
missing = [f for f in FEATURES if f not in eth_cols]
if missing:
    print(f'  ETH 缺失特征: {missing}')
else:
    print(f'  ETH 所有特征均存在')
btc_only = btc_cols - eth_cols
print(f'  BTC 独有列: {sorted(btc_only)[:10]}...')

del ctx, ctx_btc, X_test, R; gc.collect()
print('\n完成.')