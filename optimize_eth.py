#!/usr/bin/env python3
"""ETH 批量优化: 测试多个方向找出最佳方案。
不用 BTC 数据/模型，只在 ETH 上训练和评估。
"""
import os, sys, gc, time, json
import numpy as np
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
]
CROSS_FEATURES = [
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

def train_seed(feats, ctx, params, seed, max_train=2_600_000):
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = ctx.X_subset(feats, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    rng = np.random.default_rng(seed)
    tr_idx = tr_idx_all.copy()
    if len(tr_idx) > max_train:
        keep = rng.choice(len(tr_idx), max_train, replace=False)
        tr_idx = tr_idx[keep]
    train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
    Xtr = ctx.X_subset(feats, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
    if len(tr_idx) < len(tr_idx_all):
        keep_local = np.where(train_mask[tr_idx_all])[0]
        raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
    else:
        raw_w = np.abs(train_retf).astype(np.float64)
    spw = params.get('scale_pos_weight', 1.0)
    if spw > 1.0:
        raw_w[ytr > 0.5] *= spw
    w = np.clip(raw_w * 50, 0.5, 5.0)
    p = dict(objective="binary", metric="auc", learning_rate=params.get('lr', 0.02),
             num_leaves=params.get('num_leaves', 127), max_depth=-1, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
             lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
             num_threads=config.N_JOBS, verbosity=-1, seed=seed, min_data=1)
    dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des],
                  valid_names=['early_stop'], feval=topk_acc_eval,
                  callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                             lgb.log_evaluation(0)])
    auc_val = m.best_score['early_stop'].get('auc', -1)
    topk_val = m.best_score['early_stop'].get('top1_acc', -1)
    del Xtr, dtr, ytr, w, Xes; gc.collect()
    return m, m.best_iteration, auc_val, topk_val

def evaluate_ensemble(models, feats, ctx):
    from sklearn.linear_model import LogisticRegression
    mask = ctx.split_rows["test"]
    X = ctx.X_subset(feats, mask); n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del p; gc.collect()
    p_raw = R.mean(axis=0)
    cal_mask = ctx.split_rows["meta_val"]
    Xcal = ctx.X_subset(feats, cal_mask); ncal = len(Xcal)
    Rcal = np.zeros((len(models), ncal), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(Xcal, num_iteration=m.best_iteration)
        Rcal[i] = np.argsort(np.argsort(p)).astype(np.float64) / (ncal - 1)
        del p; gc.collect()
    cal_p = Rcal.mean(axis=0); cal_y = ctx.label[cal_mask]
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logit_p = np.clip(cal_p, 1e-7, 1-1e-7); logit_p = np.log(logit_p/(1-logit_p)).reshape(-1,1)
    calibrator.fit(logit_p, cal_y)
    logit_te = np.clip(p_raw, 1e-7, 1-1e-7); logit_te = np.log(logit_te/(1-logit_te)).reshape(-1,1)
    pt = calibrator.predict_proba(logit_te)[:, 1].astype(np.float64)
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    del X, Xcal, R, Rcal; gc.collect()
    return r

def run_experiment(name, feats, params, ctx, seed_count=3):
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  [{name}] feats={len(feats)} lr={params.get('lr',0.02)} nl={params.get('num_leaves',127)} spw={params.get('scale_pos_weight',1.0):.1f}", flush=True)
    print(f"{'='*60}", flush=True)
    models = []
    for seed in BAGGED_SEEDS[:seed_count]:
        m, bi, auc, topk = train_seed(feats, ctx, params, seed)
        models.append(m)
        print(f"  seed{seed}: iter={bi} auc={auc:.4f} top1={topk:.4f}", flush=True)
    r = evaluate_ensemble(models, feats, ctx)
    print(f"  >> test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    for m in models: del m; gc.collect()
    return r

def main():
    print(f"\n{'='*65}", flush=True)
    print(f"  ETH 批量优化: 探索多个方向", flush=True)
    print(f"  目标: 准确率 >= {config.TARGET_ACCURACY}", flush=True)
    print(f"  限制: 只使用 ETH 数据", flush=True)
    print(f"{'='*65}\n", flush=True)
    
    ctx = AssetContext("ETH", horizon=30)
    results = {}
    
    # A: 基线49维 (验证原61.12%)
    r = run_experiment("A_基线49维", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["A_基线49维"] = r
    
    # B: 59维含交叉特征
    r = run_experiment("B_含交叉特征59维", BASE_FEATURES + CROSS_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["B_含交叉特征59维"] = r
    
    # C1: 无正样本权重 (spw=1.0)
    r = run_experiment("C1_spw_1.0", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 1.0}, ctx, seed_count=3)
    results["C1_spw_1.0"] = r
    
    # C2: spw=1.5
    r = run_experiment("C2_spw_1.5", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 1.5}, ctx, seed_count=3)
    results["C2_spw_1.5"] = r
    
    # C3: spw=3.0
    r = run_experiment("C3_spw_3.0", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 3.0}, ctx, seed_count=3)
    results["C3_spw_3.0"] = r
    
    # C4: spw=5.0
    r = run_experiment("C4_spw_5.0", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 5.0}, ctx, seed_count=3)
    results["C4_spw_5.0"] = r
    
    # D1: lr=0.01
    r = run_experiment("D1_lr_0.01", BASE_FEATURES, {"lr": 0.01, "num_leaves": 127, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["D1_lr_0.01"] = r
    
    # D2: lr=0.03
    r = run_experiment("D2_lr_0.03", BASE_FEATURES, {"lr": 0.03, "num_leaves": 127, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["D2_lr_0.03"] = r
    
    # E1: num_leaves=63
    r = run_experiment("E1_nl_63", BASE_FEATURES, {"lr": 0.02, "num_leaves": 63, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["E1_nl_63"] = r
    
    # E2: num_leaves=255
    r = run_experiment("E2_nl_255", BASE_FEATURES, {"lr": 0.02, "num_leaves": 255, "scale_pos_weight": 2.0}, ctx, seed_count=3)
    results["E2_nl_255"] = r
    
    # 汇总
    print(f"\n{'='*65}", flush=True)
    print(f"  ETH 优化结果汇总", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Name':<20} {'Acc':>8} {'Ret(bps)':>9} {'T/D':>6} {'Conf':>7}", flush=True)
    print(f"  {'-'*55}", flush=True)
    best = ("", 0.0)
    for name, r in results.items():
        print(f"  {name:<20} {r['accuracy']:>8.4f} {r['avg_ret_bps']:>9.1f} {r['trades_per_day']:>6.1f} {r['conf_mean']:>7.4f}", flush=True)
        if r['accuracy'] > best[1]:
            best = (name, r['accuracy'])
    print(f"  {'-'*55}", flush=True)
    print(f"  最佳: {best[0]} = {best[1]:.4f}", flush=True)
    print(f"  目标: {config.TARGET_ACCURACY}", flush=True)
    print(f"  达成: {'✓' if best[1] >= config.TARGET_ACCURACY else '✗'}", flush=True)
    
    del ctx; gc.collect()

if __name__ == "__main__":
    main()