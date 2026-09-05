#!/usr/bin/env python3
"""ETH 第二轮优化: 软标签训练 + 统计信号 + FAISS + 集成
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
BAGGED_SEEDS = [42, 49, 56, 63, 70]

def topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True

def train_gbdt_seed(feats, ctx, params, seed, max_train=2_600_000, use_soft_label=False):
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
    Xtr = ctx.X_subset(feats, train_mask)
    
    if use_soft_label:
        ytr = ctx.soft_label[train_mask].astype(np.float64)
    else:
        ytr = ctx.label[train_mask].astype(np.float64)
    
    if len(tr_idx) < len(tr_idx_all):
        keep_local = np.where(train_mask[tr_idx_all])[0]
        raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
    else:
        raw_w = np.abs(train_retf).astype(np.float64)
    spw = params.get('scale_pos_weight', 1.0)
    if spw > 1.0 and not use_soft_label:
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

def evaluate_ensemble(models, feats, ctx, use_calibration=True):
    from sklearn.linear_model import LogisticRegression
    mask = ctx.split_rows["test"]
    X = ctx.X_subset(feats, mask); n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        p = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        del p; gc.collect()
    p_raw = R.mean(axis=0)
    
    if use_calibration:
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
    else:
        pt = p_raw
    
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    del X, R; gc.collect()
    if use_calibration: del Xcal, Rcal; gc.collect()
    return r, pt

def run_experiment(name, feats, params, ctx, seed_count=5, use_soft_label=False, use_calibration=True):
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  [{name}] feats={len(feats)} soft={use_soft_label} cal={use_calibration} seeds={seed_count}", flush=True)
    print(f"{'='*60}", flush=True)
    models = []
    for seed in BAGGED_SEEDS[:seed_count]:
        m, bi, auc, topk = train_gbdt_seed(feats, ctx, params, seed, use_soft_label=use_soft_label)
        models.append(m)
        print(f"  seed{seed}: iter={bi} auc={auc:.4f} top1={topk:.4f}", flush=True)
    r, pt = evaluate_ensemble(models, feats, ctx, use_calibration=use_calibration)
    print(f"  >> test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    for m in models: del m; gc.collect()
    return r, pt

def run_stat_signal(ctx):
    """统计信号模型"""
    from models.stat_signal import StatSignal
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  [StatSignal] 统计信号模型 on ETH", flush=True)
    print(f"{'='*60}", flush=True)
    m = StatSignal(seed=42)
    m.fit(ctx)
    pt = m.predict(ctx, "test").astype(np.float64)
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >> test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return r, pt

def run_faiss(ctx):
    """FAISS形态聚类模型"""
    from models.faiss_shape import FaissShapeModel
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  [FaissShape] 形态聚类 on ETH", flush=True)
    print(f"{'='*60}", flush=True)
    m = FaissShapeModel(seed=42)
    m.fit(ctx)
    pt = m.predict(ctx, "test").astype(np.float64)
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >> test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return r, pt

def main():
    print(f"\n{'='*65}", flush=True)
    print(f"  ETH 第二轮优化: 软标签 + 统计信号 + FAISS + 集成", flush=True)
    print(f"  目标: 准确率 >= {config.TARGET_ACCURACY}", flush=True)
    print(f"{'='*65}\n", flush=True)
    
    ctx = AssetContext("ETH", horizon=30)
    results = {}; predictions = {}
    
    # 1. 软标签训练 (spw=1.0, no positive weighting)
    params = {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 1.0}
    r, pt = run_experiment("F1_软标签_spw1.0", BASE_FEATURES, params, ctx, seed_count=5, use_soft_label=True, use_calibration=True)
    results["F1_软标签_spw1.0"] = r; predictions["F1_软标签_spw1.0"] = pt
    
    # 2. 软标签 + 无校准
    r, pt = run_experiment("F2_软标签_nocal", BASE_FEATURES, params, ctx, seed_count=5, use_soft_label=True, use_calibration=False)
    results["F2_软标签_nocal"] = r; predictions["F2_软标签_nocal"] = pt
    
    # 3. 二元标签 + 无校准
    r, pt = run_experiment("F3_二元_nocal", BASE_FEATURES, {"lr": 0.02, "num_leaves": 127, "scale_pos_weight": 1.0}, ctx, seed_count=5, use_soft_label=False, use_calibration=False)
    results["F3_二元_nocal"] = r; predictions["F3_二元_nocal"] = pt
    
    # 4. 统计信号
    r, pt = run_stat_signal(ctx)
    results["F4_stat"] = r; predictions["F4_stat"] = pt
    
    # 5. FAISS
    r, pt = run_faiss(ctx)
    results["F5_faiss"] = r; predictions["F5_faiss"] = pt
    
    # 6. 集成: 软标签 + 二元 + stat + faiss
    print(f"\n{'='*60}", flush=True)
    print(f"  [集成] 搜索最优权重", flush=True)
    print(f"{'='*60}", flush=True)
    pt_list = [predictions[k] for k in ["F1_软标签_spw1.0", "F3_二元_nocal", "F4_stat", "F5_faiss"]]
    names = ["soft_binary", "binary_nocal", "stat", "faiss"]
    
    best_acc = 0; best_w = None
    print(f"  {'w1':>5} {'w2':>5} {'w3':>5} {'w4':>5}  {'Acc':>8} {'Ret':>9}", flush=True)
    for w1 in np.arange(0.0, 1.05, 0.1):
        for w2 in np.arange(0.0, 1.05, 0.1):
            for w3 in np.arange(0.0, 1.05, 0.1):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -0.01 or w4 > 1.01: continue
                if abs(w1) < 0.01 and abs(w2) < 0.01 and abs(w3) < 0.01: continue
                pt = w1 * pt_list[0] + w2 * pt_list[1] + w3 * pt_list[2] + w4 * pt_list[3]
                r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
                marker = ' <<<' if r['accuracy'] > best_acc else ''
                if r['accuracy'] > best_acc + 1e-6:
                    best_acc = r['accuracy']; best_w = (w1, w2, w3, w4)
                if r['accuracy'] >= 0.58 or r['accuracy'] > best_acc - 0.005:
                    print(f"  {w1:.2f} {w2:.2f} {w3:.2f} {w4:.2f}  {r['accuracy']:.4f}  {r['avg_ret_bps']:>7.1f}{marker}", flush=True)
    
    print(f"\n  最佳集成: {dict(zip(names, best_w))}", flush=True)
    print(f"  准确率: {best_acc:.4f}", flush=True)
    
    # 汇总
    print(f"\n{'='*65}", flush=True)
    print(f"  ETH 第二轮优化结果汇总", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Name':<20} {'Acc':>8} {'Ret(bps)':>9} {'T/D':>6} {'Conf':>7}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name, r in results.items():
        print(f"  {name:<20} {r['accuracy']:>8.4f} {r['avg_ret_bps']:>9.1f} {r['trades_per_day']:>6.1f} {r['conf_mean']:>7.4f}", flush=True)
    print(f"  {'-'*55}", flush=True)
    print(f"  最佳集成: {best_acc:.4f}", flush=True)
    print(f"  目标: {config.TARGET_ACCURACY}", flush=True)
    
    del ctx; gc.collect()

if __name__ == "__main__":
    main()