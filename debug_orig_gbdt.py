#!/usr/bin/env python3
"""对比: 原版 GBDT (含交叉特征 + 正样本权重) vs 增强版 GBDT 在 ETH 上的表现。"""
import os, sys, gc, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 原版 GBDT 特征 (49 基础 + 10 交叉)
ORIG_FEATURES = [
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

def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True

print("加载 ETH 数据...", flush=True)
ctx = AssetContext("ETH", horizon=30)
trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
tr_idx_all = np.where(trm)[0]
print(f"  训练集: {len(tr_idx_all):,} 行")
print(f"  early_stop: {esm.sum():,} 行")

Xes = ctx.X_subset(ORIG_FEATURES, esm)
yes = ctx.label[esm].astype(np.float64)
print(f"  Xes shape: {Xes.shape}")
print(f"  Xes NaN: {np.isnan(Xes).sum()}, Inf: {np.isinf(Xes).sum()}")

train_retf = ctx.retf("train")

# 训练 1 个种子 (原版参数)
seed = 42
print(f"\n=== 原版 GBDT (含交叉特征) seed{seed} ===", flush=True)
rng = np.random.default_rng(seed)
tr_idx = tr_idx_all.copy()
if len(tr_idx) > 2_600_000:
    keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
    tr_idx = tr_idx[keep]
train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
Xtr = ctx.X_subset(ORIG_FEATURES, train_mask)
ytr = ctx.label[train_mask].astype(np.float64)
print(f"  训练 X shape: {Xtr.shape}, NaN: {np.isnan(Xtr).sum()}, Inf: {np.isinf(Xtr).sum()}")

# 原版权重: 正样本权重 * 2 + 按收益绝对值加权
if len(tr_idx) < len(tr_idx_all):
    keep_local = np.where(train_mask[tr_idx_all])[0]
    raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
else:
    raw_w = np.abs(train_retf).astype(np.float64)

# 正样本权重放大 (原版)
raw_w[ytr > 0.5] *= 2.0
w = np.clip(raw_w * 50, 0.5, 5.0)
print(f"  权重: min={w.min():.2f} max={w.max():.2f} median={np.median(w):.2f}")

params = dict(objective="binary", metric="auc", learning_rate=0.02,
              num_leaves=127, max_depth=-1, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
              lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,  # 原版参数
              num_threads=config.N_JOBS, verbosity=1, seed=seed, min_data=1)

dtr = lgb.Dataset(Xtr, ytr, weight=w)
des = lgb.Dataset(Xes, yes, reference=dtr)

print(f"\n  开始训练 (原版参数, scale_pos_weight=2.0 + 正样本权重x2)...", flush=True)
m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
              valid_names=['early_stop'], feval=topk_acc_eval,
              callbacks=[lgb.early_stopping(200, verbose=True, min_delta=1e-5),
                         lgb.log_evaluation(10)])

print(f"\n  训练完成: best_iter={m.best_iteration}")
print(f"  early_stop scores: {m.best_score.get('early_stop', {})}")
print(f"  总树数: {m.num_trees()}")

# 预测测试集
test_mask = ctx.split_rows["test"]
Xte = ctx.X_subset(ORIG_FEATURES, test_mask)
y_te = ctx.y("test").astype(np.float64)
retf_te = ctx.retf("test")

p = m.predict(Xte, num_iteration=m.best_iteration)
probs = 1.0 / (1.0 + np.exp(-p))
print(f"\n  测试集预测: mean={probs.mean():.4f} std={probs.std():.4f}")
print(f"  p99={np.percentile(probs, 99):.4f} p1={np.percentile(probs, 1):.4f}")

n = len(probs); k = max(1, int(n * 0.01))
conf = np.abs(probs - 0.5) * 2
sel = np.argpartition(-conf, k)[:k]
acc = (y_te[sel] > 0.5).mean()
print(f"  top-1% 准确率: {acc:.4f}")

# 对比: 用原版特征 + 增强版参数 (无交叉特征, 无正样本权重)
print(f"\n=== 对比实验: 交叉特征是否关键 ===", flush=True)
print(f"  用原版特征 (59维) + 增强版参数 (scale_pos_weight=1.0, 无正样本权重x2)")

rng = np.random.default_rng(seed)
tr_idx = tr_idx_all.copy()
if len(tr_idx) > 2_600_000:
    keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
    tr_idx = tr_idx[keep]
train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
Xtr2 = ctx.X_subset(ORIG_FEATURES, train_mask)
ytr2 = ctx.label[train_mask].astype(np.float64)

# 增强版参数: 无正样本权重, scale_pos_weight=1.0
if len(tr_idx) < len(tr_idx_all):
    keep_local = np.where(train_mask[tr_idx_all])[0]
    raw_w2 = np.abs(train_retf[keep_local]).astype(np.float64)
else:
    raw_w2 = np.abs(train_retf).astype(np.float64)
# 不加正样本权重
w2 = np.clip(raw_w2 * 50, 0.5, 5.0)
print(f"  权重: min={w2.min():.2f} max={w2.max():.2f} median={np.median(w2):.2f}")

params2 = dict(objective="binary", metric="auc", learning_rate=0.02,
               num_leaves=127, max_depth=-1, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
               lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,  # 增强版参数
               num_threads=config.N_JOBS, verbosity=1, seed=seed, min_data=1)

dtr2 = lgb.Dataset(Xtr2, ytr2, weight=w2)
des2 = lgb.Dataset(Xes, yes, reference=dtr2)
print(f"\n  开始训练...", flush=True)
m2 = lgb.train(params2, dtr2, num_boost_round=5000, valid_sets=[des2],
               valid_names=['early_stop'], feval=topk_acc_eval,
               callbacks=[lgb.early_stopping(200, verbose=True, min_delta=1e-5),
                          lgb.log_evaluation(10)])

print(f"\n  训练完成: best_iter={m2.best_iteration}")
print(f"  early_stop scores: {m2.best_score.get('early_stop', {})}")

p2 = m2.predict(Xte, num_iteration=m2.best_iteration)
probs2 = 1.0 / (1.0 + np.exp(-p2))
print(f"\n  测试集预测: mean={probs2.mean():.4f} std={probs2.std():.4f}")
n = len(probs2); k = max(1, int(n * 0.01))
conf2 = np.abs(probs2 - 0.5) * 2
sel2 = np.argpartition(-conf2, k)[:k]
acc2 = (y_te[sel2] > 0.5).mean()
print(f"  top-1% 准确率: {acc2:.4f}")

# 加载原版已保存的模型
print(f"\n=== 原版已保存的 GBDT 模型 (ETH) ===")
pt_orig = np.load(f"{config.DS_DIR}/ETH_gbdt_h30_pt.npy").astype(np.float64)
from evaluate import evaluate_topk
r = evaluate_topk(pt_orig, y_te, retf_te, ctx.times("test"))
print(f"  top-1% 准确率: {r['accuracy']:.4f}")

del ctx, Xtr, Xes, Xte, Xtr2; gc.collect()
print("\n完成.")