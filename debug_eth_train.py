#!/usr/bin/env python3
"""调试 ETH 训练: 用与 BTC 完全相同的流程, 对比每一步。"""
import os, sys, gc, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

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

# === 加载数据 ===
print("加载 ETH 数据...", flush=True)
ctx = AssetContext("ETH", horizon=30)
extra_raw = compute_extra_raw(ctx)

trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
tr_idx_all = np.where(trm)[0]
print(f"  训练集: {len(tr_idx_all):,} 行")
print(f"  early_stop: {esm.sum():,} 行")

# 检查 early_stop 标签分布
yes = ctx.label[esm].astype(np.float64)
print(f"  early_stop 涨: {(yes > 0.5).sum()}/{len(yes)} = {(yes > 0.5).mean():.3f}")
print(f"  early_stop 跌: {(yes <= 0.5).sum()}/{len(yes)} = {(yes <= 0.5).mean():.3f}")

# 检查早期验证集收益率
es_retf = ctx.retf("early_stop")
print(f"  early_stop retf: mean={np.nanmean(es_retf)*10000:.2f}bps std={np.nanstd(es_retf)*10000:.2f}bps")

# 构建 early_stop 特征矩阵
Xes = get_X(ctx, extra_raw, esm)
print(f"  Xes shape: {Xes.shape}, dtype={Xes.dtype}")
print(f"  Xes NaN: {np.isnan(Xes).sum()}, Inf: {np.isinf(Xes).sum()}")

# 检查训练集收益率
train_retf = ctx.retf("train")
print(f"\n  训练集 retf: mean={np.nanmean(train_retf)*10000:.2f}bps std={np.nanstd(train_retf)*10000:.2f}bps")
print(f"  训练集 retf NaN: {np.isnan(train_retf).sum()}")

# === 训练一个种子 ===
seed = 42
print(f"\n=== 训练 seed{seed} ===", flush=True)

rng = np.random.default_rng(seed)
tr_idx = tr_idx_all.copy()
if len(tr_idx) > 2_600_000:
    keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
    tr_idx = tr_idx[keep]
train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
Xtr = get_X(ctx, extra_raw, train_mask)
ytr = ctx.label[train_mask].astype(np.float64)
print(f"  训练 X shape: {Xtr.shape}")
print(f"  训练 y: 涨={(ytr > 0.5).sum()}, 跌={(ytr <= 0.5).sum()}, 比例={(ytr > 0.5).mean():.3f}")

# 检查训练特征
print(f"  训练 X NaN: {np.isnan(Xtr).sum()}, Inf: {np.isinf(Xtr).sum()}")

# 权重计算
if len(tr_idx) < len(tr_idx_all):
    keep_local = np.where(train_mask[tr_idx_all])[0]
    raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
else:
    raw_w = np.abs(train_retf).astype(np.float64)
print(f"  权重: min={raw_w.min()*10000:.2f}bps max={raw_w.max()*10000:.2f}bps median={np.median(raw_w)*10000:.2f}bps")

w = np.clip(raw_w * 50, 0.5, 5.0)
print(f"  裁剪后权重: min={w.min():.2f} max={w.max():.2f} median={np.median(w):.2f}")

train_hour = (ctx.ds_ts[train_mask] % 86400) // 3600
bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
print(f"  坏时段 (hour 17-20, 0-5): {bad_hour.sum()} 样本 ({bad_hour.mean()*100:.1f}%)")
if bad_hour.any(): w[bad_hour] *= 2.0
print(f"  最终权重: min={w.min():.2f} max={w.max():.2f} median={np.median(w):.2f}")

# 训练
params = dict(objective="binary", metric="auc", learning_rate=0.02,
              num_leaves=127, max_depth=-1, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
              lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
              num_threads=config.N_JOBS, verbosity=1, seed=seed,
              min_data=1)

dtr = lgb.Dataset(Xtr, ytr, weight=w)
des = lgb.Dataset(Xes, yes, reference=dtr)

print(f"\n  开始训练...", flush=True)
m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
              valid_names=['early_stop'], feval=topk_acc_eval,
              callbacks=[lgb.early_stopping(200, verbose=True, min_delta=1e-5),
                         lgb.log_evaluation(10)])

print(f"\n  训练完成: best_iter={m.best_iteration}")
print(f"  best_score keys: {list(m.best_score.keys())}")
print(f"  early_stop scores: {m.best_score.get('early_stop', {})}")
print(f"  总树数: {m.num_trees()}")

# 预测测试集
test_mask = ctx.split_rows["test"]
Xte = get_X(ctx, extra_raw, test_mask)
y_te = ctx.y("test").astype(np.float64)
retf_te = ctx.retf("test")

p = m.predict(Xte, num_iteration=m.best_iteration)
probs = 1.0 / (1.0 + np.exp(-p))
print(f"\n  测试集预测: mean={probs.mean():.4f} std={probs.std():.4f}")
print(f"  p99={np.percentile(probs, 99):.4f} p1={np.percentile(probs, 1):.4f}")

# top-1% 评估
n = len(probs); k = max(1, int(n * 0.01))
conf = np.abs(probs - 0.5) * 2
sel = np.argpartition(-conf, k)[:k]
acc = (y_te[sel] > 0.5).mean()
print(f"  top-1% 准确率: {acc:.4f}")

# 对比: 用同一模型预测BTC测试集
print(f"\n=== 对比: 同样模型预测BTC测试集 ===")
ctx_btc = AssetContext("BTC", horizon=30)
extra_btc = compute_extra_raw(ctx_btc)
btc_test_mask = ctx_btc.split_rows["test"]
Xte_btc = get_X(ctx_btc, extra_btc, btc_test_mask)
y_btc = ctx_btc.y("test").astype(np.float64)
retf_btc = ctx_btc.retf("test")

p_btc = m.predict(Xte_btc, num_iteration=m.best_iteration)
probs_btc = 1.0 / (1.0 + np.exp(-p_btc))
print(f"  BTC测试集预测: mean={probs_btc.mean():.4f} std={probs_btc.std():.4f}")
n = len(probs_btc); k = max(1, int(n * 0.01))
conf = np.abs(probs_btc - 0.5) * 2
sel = np.argpartition(-conf, k)[:k]
acc = (y_btc[sel] > 0.5).mean()
print(f"  BTC top-1% 准确率: {acc:.4f}")

del ctx, ctx_btc, Xtr, Xes, Xte, Xte_btc, m; gc.collect()
print("\n完成.")