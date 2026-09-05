#!/usr/bin/env python3
"""用原始 GBDT 架构 (含交叉特征 + 正样本权重) 在重建后的 ETH 数据集上训练。"""
import os, sys, gc, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 原始 GBDT 特征 (49 基础 + 10 交叉)
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
    # 交叉特征 (现在 ETH 数据集也有这些列了)
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]
BAGGED_SEEDS = [42, 49, 56, 63, 70]
MODEL_DIR = "/workspace/simpletrade/models_saved/eth_orig"

def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True

os.makedirs(MODEL_DIR, exist_ok=True)
symbol = "ETH"
print(f"加载 {symbol} 数据...", flush=True)
ctx = AssetContext(symbol, horizon=30)
trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
tr_idx_all = np.where(trm)[0]
print(f"  训练集: {len(tr_idx_all):,} 行, early_stop: {esm.sum():,} 行")

Xes = ctx.X_subset(FEATURES, esm); yes = ctx.label[esm].astype(np.float64)
print(f"  Xes shape: {Xes.shape} (含交叉特征: {len(FEATURES)}维)")

train_retf = ctx.retf("train")
t0 = time.time()

for seed in BAGGED_SEEDS:
    print(f"\n  [orig] seed{seed} 开始...", flush=True)
    rng = np.random.default_rng(seed)
    tr_idx = tr_idx_all.copy()
    if len(tr_idx) > 2_600_000:
        keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
        tr_idx = tr_idx[keep]
    train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
    Xtr = ctx.X_subset(FEATURES, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
    
    # 原版权重: 正样本权重 * 2
    if len(tr_idx) < len(tr_idx_all):
        keep_local = np.where(train_mask[tr_idx_all])[0]
        raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
    else:
        raw_w = np.abs(train_retf).astype(np.float64)
    raw_w[ytr > 0.5] *= 2.0
    w = np.clip(raw_w * 50, 0.5, 5.0)
    
    params = dict(objective="binary", metric="auc", learning_rate=0.02,
                  num_leaves=127, max_depth=-1, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
                  lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,
                  num_threads=config.N_JOBS, verbosity=-1, seed=seed, min_data=1)
    
    dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
                  valid_names=['early_stop'], feval=topk_acc_eval,
                  callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                             lgb.log_evaluation(0)])
    auc_val = m.best_score['early_stop']['auc'] if 'auc' in m.best_score.get('early_stop', {}) else -1
    topk_val = m.best_score['early_stop']['top1_acc'] if 'top1_acc' in m.best_score.get('early_stop', {}) else -1
    print(f"  [orig] seed{seed} best_iter={m.best_iteration} auc={auc_val:.4f} top1_acc={topk_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
    
    m.save_model(f"{MODEL_DIR}/{symbol}_orig_seed{seed}.txt")
    del m, dtr, Xtr; gc.collect()

# Platt 校准
print(f"\n  [orig] 校准中...", flush=True)
from sklearn.linear_model import LogisticRegression
import joblib

cal_mask = ctx.split_rows["meta_val"]
Xcal = ctx.X_subset(FEATURES, cal_mask); ycal = ctx.label[cal_mask].astype(np.float64)
R = np.zeros((len(BAGGED_SEEDS), len(Xcal)), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = lgb.Booster(model_file=f"{MODEL_DIR}/{symbol}_orig_seed{seed}.txt")
    p = m.predict(Xcal, num_iteration=m.best_iteration)
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (len(Xcal) - 1)
    del m
p_raw = R.mean(axis=0)
cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
logit_p = np.clip(p_raw, 1e-7, 1-1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
cal.fit(logit_p, ycal)
joblib.dump(cal, f"{MODEL_DIR}/{symbol}_orig_calib.joblib")
print(f"  [orig] 校准完成 ({time.time()-t0:.0f}s)", flush=True)

# 预测测试集
print(f"\n  [orig] 预测测试集...", flush=True)
test_mask = ctx.split_rows["test"]
Xte = ctx.X_subset(FEATURES, test_mask)
y_te = ctx.y("test").astype(np.float64); retf_te = ctx.retf("test"); ts_te = ctx.times("test")

R = np.zeros((len(BAGGED_SEEDS), len(Xte)), dtype=np.float64)
for i, seed in enumerate(BAGGED_SEEDS):
    m = lgb.Booster(model_file=f"{MODEL_DIR}/{symbol}_orig_seed{seed}.txt")
    p = m.predict(Xte, num_iteration=m.best_iteration)
    R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (len(Xte) - 1)
    del m
p_raw = R.mean(axis=0)
logit_p = np.clip(p_raw, 1e-7, 1-1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
pt = cal.predict_proba(logit_p)[:, 1].astype(np.float64)

from evaluate import evaluate_topk
r = evaluate_topk(pt, y_te, retf_te, ts_te)
print(f"  [orig] 测试集: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f}", flush=True)

# 对比 BTC 模型预测
print(f"\n=== 对比: BTC 训练模型 → ETH 预测 ===")
btc_dir = "/workspace/simpletrade/models_saved/eth_validate"
for name, prefix in [("BTC-enhanced", "enhanced"), ("BTC-catboost", "catboost")]:
    R = np.zeros((len(BAGGED_SEEDS), len(Xte)), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        if name == "BTC-enhanced":
            m = lgb.Booster(model_file=f"{btc_dir}/{prefix}_seed{seed}.txt")
            p = m.predict(Xte, num_iteration=m.best_iteration)
        else:
            from catboost import CatBoost
            m = CatBoost(); m.load_model(f"{btc_dir}/{prefix}_seed{seed}.cbm")
            p = m.predict(Xte, prediction_type='Probability')[:, 1]
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (len(Xte) - 1)
        del m
    p_raw = R.mean(axis=0)
    cal = joblib.load(f"{btc_dir}/{prefix}_calib.joblib")
    logit_p = np.clip(p_raw, 1e-7, 1-1e-7); logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    pt = cal.predict_proba(logit_p)[:, 1].astype(np.float64)
    r = evaluate_topk(pt, y_te, retf_te, ts_te)
    print(f"  {name} → ETH: acc={r['accuracy']:.4f}")

# 保存
np.save(f"{config.DS_DIR}/ETH_orig_pt.npy", pt.astype(np.float32))
print(f"\n完成! 总耗时: {time.time()-t0:.0f}s")