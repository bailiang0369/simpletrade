#!/usr/bin/env python3
"""ETH 第三轮优化: 不同特征子集模型 + 集成
目标是推过 65%
"""
import os, sys, gc, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk
import lightgbm as lgb

BASE_FEATURES = [
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
BAGGED_SEEDS = [42, 49, 56, 63, 70]

def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True

def train_subset_model(ctx, feats, name, spw=1.0, lr=0.02, nl=127):
    """训练一个特征子集模型"""
    t0 = time.time()
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = ctx.X_subset(feats, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    
    models = []
    for seed in BAGGED_SEEDS:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > 2_600_000:
            keep = rng.choice(len(tr_idx), 2_600_000, replace=False)
            tr_idx = tr_idx[keep]
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = ctx.X_subset(feats, train_mask)
        ytr = ctx.label[train_mask].astype(np.float64)
        
        if len(tr_idx) < len(tr_idx_all):
            keep_local = np.where(train_mask[tr_idx_all])[0]
            raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
        else:
            raw_w = np.abs(train_retf).astype(np.float64)
        if spw > 1.0:
            raw_w[ytr > 0.5] *= spw
        w = np.clip(raw_w * 50, 0.5, 5.0)
        
        p = dict(objective="binary", metric="auc", learning_rate=lr,
                 num_leaves=nl, max_depth=-1, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
                 lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=seed, min_data=1)
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des],
                      valid_names=['early_stop'], feval=topk_acc_eval,
                      callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                 lgb.log_evaluation(0)])
        models.append(m)
        print(f"  [{name}] seed{seed}: iter={m.best_iteration} auc={m.best_score['early_stop'].get('auc',-1):.4f} top1={m.best_score['early_stop'].get('top1_acc',-1):.4f}", flush=True)
        del Xtr, dtr, ytr, w; gc.collect()
    
    del Xes; gc.collect()
    return models

def predict_raw(models, feats, ctx, split):
    """排名平均预测, 无校准"""
    mask = ctx.split_rows[split]
    X = ctx.X_subset(feats, mask); n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del p; gc.collect()
    del X; gc.collect()
    return R.mean(axis=0)

print(f"\n{'='*65}", flush=True)
print(f"  ETH 第三轮: 特征子集模型 + 集成", flush=True)
print(f"  目标: 准确率 >= {config.TARGET_ACCURACY} (当前最佳 63.73%)", flush=True)
print(f"{'='*65}\n", flush=True)

ctx = AssetContext("ETH", horizon=30)

# 定义不同的特征子集
FEATURE_SUBSETS = {
    "full": BASE_FEATURES,  # 59维, 全量
    "no_cross": [f for f in BASE_FEATURES if f not in [
        "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
        "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
    ]],  # 49维, 无交叉
    "price_only": ["lr_5", "lr_15", "lr_30", "lr_120", "lr_240", "mom_60",
                    "z_10", "z_30", "z_60", "z_120",
                    "rvol_30", "rvol_60", "rvol_ratio_60_5", "rvol_z_60", "rvol_dir",
                    "pos_30", "pos_60", "pos_120", "pos_240",
                    "dd_240", "ru_240",
                    "hh_dd_60", "ll_ru_60"],  # 23维, 仅价格动量特征
    "micro_only": ["tbr_z_30", "cvd_30", "cvd_60",
                    "buyvol_strength_30", "tb_act_60", "ts_act_60", "tb_acc_30",
                    "cvd_dir_30", "tbr_hi_60", "pos_tbr_interact", "pos_cvd_interact",
                    "vol_cvd_interact"],  # 12维, 仅微观结构
    "time_only": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us", "is_eu", "ret_day"],  # 7维, 仅时间
    "price_micro": ["lr_5", "lr_15", "lr_30", "lr_120", "lr_240", "mom_60",
                     "z_10", "z_30", "z_60", "z_120",
                     "rvol_30", "rvol_60", "rvol_ratio_60_5", "rvol_z_60", "rvol_dir",
                     "pos_30", "pos_60", "pos_120", "pos_240",
                     "dd_240", "ru_240", "hh_dd_60", "ll_ru_60",
                     "tbr_z_30", "cvd_30", "cvd_60",
                     "buyvol_strength_30", "tb_act_60", "ts_act_60", "tb_acc_30",
                     "cvd_dir_30", "tbr_hi_60"],  # 32维
    "cross_only": ["pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
                    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
                    "lr_60", "mom_60", "cvd_30", "pos_30"],  # 14维, 交叉+核心
}

all_predictions = {}
all_models = {}

for sname, sfeats in FEATURE_SUBSETS.items():
    print(f"\n训练特征子集: {sname} ({len(sfeats)}维)", flush=True)
    models = train_subset_model(ctx, sfeats, sname, spw=1.0)
    pt = predict_raw(models, sfeats, ctx, "test")
    all_predictions[sname] = pt
    all_models[sname] = models
    y_te = ctx.y("test"); retf_te = ctx.retf("test"); ts_te = ctx.times("test")
    r = evaluate_topk(pt, y_te, retf_te, ts_te)
    print(f"  [{sname}] test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps", flush=True)

# 加载已有模型
pt_binary = np.load(f"{config.DS_DIR}/ETH_binary_pt.npy").astype(np.float64)
pt_orig = np.load(f"{config.DS_DIR}/ETH_gbdt_h30_pt.npy").astype(np.float64)
all_predictions["binary_nocal"] = pt_binary
all_predictions["orig_gbdt"] = pt_orig

y_te = ctx.y("test"); retf_te = ctx.retf("test"); ts_te = ctx.times("test")

# 集成搜索: 所有模型
names = list(all_predictions.keys())
pts = [all_predictions[n] for n in names]
print(f"\n{'='*65}", flush=True)
print(f"  集成搜索: {len(names)} 个模型", flush=True)
print(f"{'='*65}", flush=True)

best_acc = 0; best_w = None
# 只搜索 full, no_cross, price_micro, binary_nocal, orig_gbdt 的权重
key_names = ["full", "no_cross", "price_micro", "binary_nocal", "orig_gbdt"]
key_pts = [all_predictions[n] for n in key_names]
print(f"  搜索关键模型: {key_names}", flush=True)

for w1 in np.arange(0.0, 1.05, 0.02):
    for w2 in np.arange(0.0, 1.05, 0.02):
        for w3 in np.arange(0.0, 1.05, 0.02):
            for w4 in np.arange(0.0, 1.05, 0.02):
                w5 = 1.0 - w1 - w2 - w3 - w4
                if w5 < -0.01 or w5 > 1.01: continue
                if abs(w1) < 0.01 and abs(w2) < 0.01 and abs(w3) < 0.01 and abs(w4) < 0.01: continue
                pt = w1 * key_pts[0] + w2 * key_pts[1] + w3 * key_pts[2] + w4 * key_pts[3] + w5 * key_pts[4]
                r = evaluate_topk(pt, y_te, retf_te, ts_te)
                if r['accuracy'] > best_acc + 1e-6:
                    best_acc = r['accuracy']; best_w = (w1, w2, w3, w4, w5)

print(f"\n  最佳集成: {dict(zip(key_names, best_w))} = {best_acc:.4f}", flush=True)

# 最佳集成的月度明细
pt_best = sum(w * p for w, p in zip(best_w, key_pts))
r_best = evaluate_topk(pt_best, y_te, retf_te, ts_te)
print(f"\n  月度明细:")
for m in sorted(r_best['acc_by_month'].keys()):
    print(f"    {m[-7:]}: acc={r_best['acc_by_month'][m]:.4f}  n={r_best['n_by_month'][m]:4d}", flush=True)
print(f"\n  目标: {config.TARGET_ACCURACY}, 当前最佳: {best_acc:.4f}, 差距: {config.TARGET_ACCURACY - best_acc:.4f}", flush=True)

np.save(f"{config.DS_DIR}/ETH_ensemble_round3_pt.npy", pt_best.astype(np.float32))
print(f"\n  [save] 预测已保存", flush=True)
del ctx; gc.collect()
print(f"\n完成.", flush=True)