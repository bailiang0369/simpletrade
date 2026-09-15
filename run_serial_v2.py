#!/usr/bin/env python3
"""串行三模型 + 堆叠融合 v2: 修复权重缩放 + 稳定早停 + 严格内存隔离。"""
import os, sys, gc, time, argparse, json
import numpy as np
sys.path.insert(0, '/workspace')
import config
from data_store import AssetContext
from evaluate import evaluate_topk
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES, BAGGED_SEEDS,
                                 compute_extra_raw, get_extra_for_mask)

CROSS_NAMES = ['lr_5','lr_15','lr_30','lr_60','lr_120','lr_240','lr_480','lr_960',
               'z_30','z_60','z_120','z_240','z_480','rvol_60','rvol_240','cvd_30','cvd_60']


def get_X(ctx, extra_raw, mask):
    X_base = ctx.X_subset(FEATURES, mask)
    X_extra = get_extra_for_mask(extra_raw, ctx, mask)
    prefix = 'BTC_' if ctx.symbol == 'ETH' else 'ETH_'
    X_cross = ctx.X_subset([prefix + f for f in CROSS_NAMES], mask)
    return np.column_stack([X_base, X_extra, X_cross])


def make_weights(ctx, train_mask):
    raw_w = np.abs(ctx.retf("train")).astype(np.float64)
    w = np.clip(raw_w * 2000, 0.5, 5.0)  # 合理缩放
    train_hour = (ctx.ds_ts[train_mask] % 86400) // 3600
    bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
    if bad_hour.any():
        w[bad_hour] *= 2.0
        print(f"  坏时段加权 ({bad_hour.sum()}/{len(w)})", flush=True)
    return w


def predict_rank(models, X, n):
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        if isinstance(m, tuple):
            booster, bi = m; import xgboost as xgb_mod
            raw = booster.predict(xgb_mod.DMatrix(X), iteration_range=(0, bi))
        elif hasattr(m, 'best_iteration'):
            raw = m.predict(X, num_iteration=m.best_iteration)
        else:
            raw = m.predict(X, prediction_type='RawFormulaVal')
        R[i] = np.argsort(np.argsort(raw)).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def train_lgb(ctx, extra_raw):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    t0 = time.time(); symbol = ctx.symbol; h = ctx.horizon
    print(f"\n{'='*70}\n[1/3] 增强 LGBM  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)

    models = []
    for seed in BAGGED_SEEDS:
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx_all.copy()] = True
        Xtr = get_X(ctx, extra_raw, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
        w = make_weights(ctx, train_mask)
        params = dict(objective='binary', metric='auc', learning_rate=0.02,
                      num_leaves=127, max_depth=-1, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
                      lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                      num_threads=config.N_JOBS, verbosity=-1, seed=seed)
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
                      valid_names=['early_stop'],
                      callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        print(f"  LGB seed{seed}: best_iter={m.best_iteration} auc_es={m.best_score['early_stop']['auc']:.4f}", flush=True)
        models.append(m)
        del Xtr, dtr, ytr, w, train_mask; gc.collect()

    del Xes, yes, trm, esm, tr_idx_all; gc.collect()

    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    Xm = get_X(ctx, extra_raw, meta_mask); pm_raw = predict_rank(models, Xm, len(Xm)); del Xm; gc.collect()
    Xt = get_X(ctx, extra_raw, test_mask); pt_raw = predict_rank(models, Xt, len(Xt)); del Xt; gc.collect()

    cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    fin = np.isfinite(pm_raw); cal.fit(np.log(np.clip(pm_raw[fin],1e-7,1-1e-7)).reshape(-1,1), ctx.label[meta_mask][fin])
    pm = cal.predict_proba(np.log(np.clip(pm_raw,1e-7,1-1e-7)).reshape(-1,1))[:,1]
    pt = cal.predict_proba(np.log(np.clip(pt_raw,1e-7,1-1e-7)).reshape(-1,1))[:,1]
    print(f"  LGB Platt coef={cal.coef_[0][0]:.4f} 完成 ({time.time()-t0:.0f}s)", flush=True)

    del models, cal, pm_raw, pt_raw; gc.collect()
    return pm.astype(np.float32), pt.astype(np.float32), ctx.label[meta_mask], ctx.label[test_mask], ctx.retf("test"), ctx.times("test")


def train_xgb(ctx, extra_raw):
    import xgboost as xgb
    t0 = time.time(); symbol = ctx.symbol; h = ctx.horizon
    print(f"\n{'='*70}\n[2/3] XGBoost  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)

    models = []
    for seed in BAGGED_SEEDS:
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx_all.copy()] = True
        Xtr = get_X(ctx, extra_raw, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
        w = make_weights(ctx, train_mask)
        params = dict(objective='binary:logistic', eval_metric='auc',
                      learning_rate=0.03, max_depth=7, min_child_weight=150,
                      subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                      nthread=config.N_JOBS, seed=seed, verbosity=0)
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=w); des = xgb.DMatrix(Xes, label=yes)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(des, "val")], early_stopping_rounds=300, verbose_eval=False)
        print(f"  XGB seed{seed}: best_iter={m.best_iteration} auc_es={m.best_score:.4f}", flush=True)
        models.append((m, m.best_iteration))
        del Xtr, dtr, ytr, w, train_mask; gc.collect()

    del Xes, yes, trm, esm, tr_idx_all; gc.collect()

    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    Xm = get_X(ctx, extra_raw, meta_mask); pm = predict_rank(models, Xm, len(Xm)); del Xm; gc.collect()
    Xt = get_X(ctx, extra_raw, test_mask); pt = predict_rank(models, Xt, len(Xt)); del Xt; gc.collect()

    del models; gc.collect()
    return pm.astype(np.float32), pt.astype(np.float32), ctx.label[meta_mask], ctx.label[test_mask], ctx.retf("test"), ctx.times("test")


def train_cat(ctx, extra_raw):
    from catboost import CatBoostClassifier, Pool
    t0 = time.time(); symbol = ctx.symbol; h = ctx.horizon
    print(f"\n{'='*70}\n[3/3] CatBoost  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)

    models = []
    for seed in BAGGED_SEEDS:
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx_all.copy()] = True
        Xtr = get_X(ctx, extra_raw, train_mask); ytr = ctx.label[train_mask].astype(np.float64)
        w = make_weights(ctx, train_mask)
        params = dict(loss_function='Logloss', eval_metric='AUC',
                      learning_rate=0.03, depth=7, min_data_in_leaf=150,
                      l2_leaf_reg=3, random_seed=seed, verbose=0, thread_count=config.N_JOBS)
        tr = Pool(Xtr, label=ytr, weight=w); va = Pool(Xes, label=yes)
        m = CatBoostClassifier(**params, iterations=5000, early_stopping_rounds=300)
        m.fit(tr, eval_set=va, use_best_model=True, verbose=False)
        bi = m.get_best_iteration() if m.get_best_iteration() is not None else 0
        print(f"  CAT seed{seed}: best_iter={bi}", flush=True)
        models.append(m)
        del Xtr, tr, ytr, w, train_mask; gc.collect()

    del Xes, yes, trm, esm, tr_idx_all; gc.collect()

    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    Xm = get_X(ctx, extra_raw, meta_mask); pm = predict_rank(models, Xm, len(Xm)); del Xm; gc.collect()
    Xt = get_X(ctx, extra_raw, test_mask); pt = predict_rank(models, Xt, len(Xt)); del Xt; gc.collect()

    del models; gc.collect()
    return pm.astype(np.float32), pt.astype(np.float32), ctx.label[meta_mask], ctx.label[test_mask], ctx.retf("test"), ctx.times("test")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("symbol", choices=["ETH","BTC"]); ap.add_argument("horizon", type=int)
    a = ap.parse_args()
    symbol, h = a.symbol, a.horizon; t0 = time.time()
    print(f"\n{'#'*70}\n  串行三模型 + 堆叠融合  {symbol} H={h}\n{'#'*70}", flush=True)

    ctx = AssetContext(symbol, horizon=h)
    extra_raw = compute_extra_raw(ctx); gc.collect()

    p_lgb_m, p_lgb_t, y_meta, y_test, retf_test, ts_test = train_lgb(ctx, extra_raw)
    gc.collect(); print("[gc] LGB done", flush=True)

    p_xgb_m, p_xgb_t, *_ = train_xgb(ctx, extra_raw)
    gc.collect(); print("[gc] XGB done", flush=True)

    p_cat_m, p_cat_t, *_ = train_cat(ctx, extra_raw)
    gc.collect(); print("[gc] CAT done", flush=True)

    # 释放 ctx + extra_raw (大对象)
    del extra_raw, ctx; gc.collect()
    print("[gc] ctx + extra_raw released", flush=True)

    # ---- 堆叠 (线性融合) ----
    from sklearn.linear_model import LogisticRegression
    X_meta = np.column_stack([p_lgb_m, p_xgb_m, p_cat_m]).astype(np.float64)
    X_test = np.column_stack([p_lgb_t, p_xgb_t, p_cat_t]).astype(np.float64)
    stack = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    stack.fit(X_meta, y_meta.astype(np.int8))
    print(f"\n[stacking] 系数: lgb={stack.coef_[0][0]:.3f} xgb={stack.coef_[0][1]:.3f} cat={stack.coef_[0][2]:.3f}", flush=True)
    p_stack_m = stack.predict_proba(X_meta)[:, 1]
    p_stack_t = stack.predict_proba(X_test)[:, 1]

    # ---- 评估各模型 ----
    print(f"\n{'='*70}\n  测试集 top1% 准确率\n{'='*70}", flush=True)
    for name, pt in [("LGB", p_lgb_t), ("XGB", p_xgb_t), ("CAT", p_cat_t), ("STACK", p_stack_t)]:
        r = evaluate_topk(pt, y_test, retf_test, ts_test)
        bar = "█" * int(r['accuracy'] * 60)
        print(f"  [{name:5s}] acc={r['accuracy']:.4f} k={r['k']:5d} trades/day={r['trades_per_day']:5.1f} avg_ret={r['avg_ret_bps']:7.1f}bps {bar}")
        if r['acc_by_month']:
            m_min = min(r['acc_by_month'], key=r['acc_by_month'].get)
            print(f"         月度最低: {r['acc_by_month'][m_min]:.4f} ({m_min[-7:]})")

    # 阈值校准 (meta_val)
    conf_m = np.maximum(p_stack_m, 1 - p_stack_m)
    k_m = max(1, int(len(p_stack_m) * config.COVERAGE))
    thresh = np.partition(conf_m, -k_m)[-k_m]
    print(f"\n[threshold] meta_val top1% 置信度阈值: {thresh:.4f}", flush=True)

    os.makedirs(f"{config.PROJECT_DIR}/results", exist_ok=True)
    out = {
        "symbol": symbol, "horizon": h,
        "threshold_top1pct": float(thresh),
        "stack_test_accuracy": float(evaluate_topk(p_stack_t, y_test, retf_test, ts_test)['accuracy']),
        "lgb_test_accuracy": float(evaluate_topk(p_lgb_t, y_test, retf_test, ts_test)['accuracy']),
        "xgb_test_accuracy": float(evaluate_topk(p_xgb_t, y_test, retf_test, ts_test)['accuracy']),
        "cat_test_accuracy": float(evaluate_topk(p_cat_t, y_test, retf_test, ts_test)['accuracy']),
        "elapsed_s": time.time() - t0,
    }
    with open(f"{config.PROJECT_DIR}/results/{symbol}_h{h}_stacking_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f"{config.DS_DIR}/{symbol}_h{h}_STACK_v2_test_p.npy", p_stack_t)
    print(f"\n结果保存: results/{symbol}_h{h}_stacking_v2.json", flush=True)
    print(f"总耗时: {time.time()-t0:.0f}s", flush=True)

    del p_lgb_m, p_xgb_m, p_cat_m, p_lgb_t, p_xgb_t, p_cat_t, p_stack_m, p_stack_t
    del y_meta, y_test, retf_test, ts_test, stack, X_meta, X_test
    gc.collect()
    print("全部完成.", flush=True)


if __name__ == "__main__":
    main()
