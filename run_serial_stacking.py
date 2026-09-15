#!/usr/bin/env python3
"""串行三模型训练 + 堆叠融合，单进程单币种单horizon。

核心约束: 每个模型训练完毕后仅保留 meta_val / test 预测值,
其余全部释放 (模型对象 / 特征矩阵 / extra_raw / Dataset)。

用法:
    python run_serial_stacking.py <SYMBOL> <HORIZON>   # e.g. ETH 15
"""
import os, sys, gc, time, argparse, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from data_store import AssetContext
from evaluate import evaluate_topk
from validate_eth_quick import (FEATURES, CROSS_FEATURES, EXTRA_FEATURE_NAMES,
                                 BAGGED_SEEDS, topk_acc_eval,
                                 compute_extra_raw, get_extra_for_mask, get_X)


def _predict_rank_avg(models, X, n):
    """多个树模型的排名平均 (LGB/XGB/Cat 统一接口)。"""
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        if isinstance(m, tuple):          # (booster, best_iter) for xgb
            booster, best_iter = m
            import xgboost as xgb_mod
            raw = booster.predict(xgb_mod.DMatrix(X), iteration_range=(0, best_iter))
        elif hasattr(m, 'best_iteration'):  # lgb.Booster
            raw = m.predict(X, num_iteration=m.best_iteration)
        else:                                # catboost
            raw = m.predict(X, prediction_type='RawFormulaVal')
        R[i] = np.argsort(np.argsort(raw)).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def train_lgb_enhanced(ctx, extra_raw):
    """增强 LGBM: 5-seed bagging + 加权采样 + 坏时段加权 + 排名平均 + Platt校准。

    返回: (meta_val_p, test_p, y_meta, y_test, retf_test, ts_test)
    释放: ctx 外的所有中间变量 (Xtr/Xes/models/datasets/... 全部 del + gc)。
    """
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    symbol, h = ctx.symbol, ctx.horizon
    print(f"\n{'='*70}\n[1/3] 增强 LGBM  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]

    # ---- 早停集 (一次加载) ----
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    models = []
    for seed in BAGGED_SEEDS:
        tr_idx = tr_idx_all.copy()  # 全量用
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = get_X(ctx, extra_raw, train_mask)
        ytr = ctx.label[train_mask].astype(np.float64)

        # ---- 加权: 未来收益绝对值 + 坏时段2x ----
        raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)
        train_hour = (ctx.ds_ts[train_mask] % 86400) // 3600
        bad_hour = ((train_hour >= 17) & (train_hour <= 20)) | (train_hour <= 5)
        if bad_hour.any():
            w[bad_hour] *= 2.0

        params = dict(
            objective="binary", metric="auc", learning_rate=0.02,
            num_leaves=127, max_depth=-1, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
            lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
            num_threads=config.N_JOBS, verbosity=-1, seed=seed,
        )
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
                      valid_names=['early_stop'], feval=topk_acc_eval,
                      callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                 lgb.log_evaluation(0)])
        topk_val = m.best_score['early_stop'].get('top1_acc', -1)
        print(f"  LGB seed{seed}: best_iter={m.best_iteration} top1_acc_es={topk_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
        models.append(m)
        # 释放当前 seed 的训练变量
        del Xtr, dtr, ytr, w, train_mask
        gc.collect()

    del Xes, yes, trm, esm, tr_idx_all, train_retf
    gc.collect()

    # ---- 生成 meta_val / test 排名平均概率 ----
    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    n_meta = meta_mask.sum(); n_test = test_mask.sum()

    # meta_val
    Xm = get_X(ctx, extra_raw, meta_mask)
    p_rank_meta = _predict_rank_avg(models, Xm, n_meta)
    del Xm; gc.collect()

    # test
    Xt = get_X(ctx, extra_raw, test_mask)
    p_rank_test = _predict_rank_avg(models, Xt, n_test)
    del Xt; gc.collect()

    # ---- Platt 校准 (在 meta_val 上拟合) ----
    cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    fin = np.isfinite(p_rank_meta)
    cal.fit(np.log(np.clip(p_rank_meta[fin], 1e-7, 1 - 1e-7)).reshape(-1, 1),
            ctx.label[meta_mask][fin])
    p_final_meta = cal.predict_proba(np.log(np.clip(p_rank_meta, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]
    p_final_test = cal.predict_proba(np.log(np.clip(p_rank_test, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]
    print(f"  LGB Platt coef={cal.coef_[0][0]:.4f} 完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 释放所有模型, 仅保留预测 ----
    del models, cal, p_rank_meta, p_rank_test
    gc.collect()

    y_meta = ctx.label[meta_mask]; y_test = ctx.label[test_mask]
    retf_test = ctx.retf("test"); ts_test = ctx.times("test")
    return (p_final_meta.astype(np.float32), p_final_test.astype(np.float32),
            y_meta, y_test, retf_test, ts_test)


def train_xgb(ctx, extra_raw):
    """XGBoost: 5-seed bagging, 全量训练。

    返回: (meta_val_p, test_p, y_meta, y_test, retf_test, ts_test)
    """
    import xgboost as xgb

    t0 = time.time()
    symbol, h = ctx.symbol, ctx.horizon
    print(f"\n{'='*70}\n[2/3] XGBoost  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")

    models = []
    for seed in BAGGED_SEEDS:
        tr_idx = tr_idx_all.copy()
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = get_X(ctx, extra_raw, train_mask)
        ytr = ctx.label[train_mask].astype(np.float64)

        raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)

        params = dict(
            objective="binary:logistic", eval_metric="auc",
            learning_rate=0.03, max_depth=7, min_child_weight=150,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            nthread=config.N_JOBS, seed=seed, verbosity=0,
        )
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=w)
        des = xgb.DMatrix(Xes, label=yes)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(des, "val")], early_stopping_rounds=200, verbose_eval=False)
        print(f"  XGB seed{seed}: best_iter={m.best_iteration} best_score={m.best_score:.4f} ({time.time()-t0:.0f}s)", flush=True)
        models.append((m, m.best_iteration))
        del Xtr, dtr, ytr, w, train_mask
        gc.collect()

    del Xes, yes, trm, esm, tr_idx_all, train_retf
    gc.collect()

    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    n_meta = meta_mask.sum(); n_test = test_mask.sum()

    Xm = get_X(ctx, extra_raw, meta_mask)
    p_meta = _predict_rank_avg(models, Xm, n_meta); del Xm; gc.collect()
    Xt = get_X(ctx, extra_raw, test_mask)
    p_test = _predict_rank_avg(models, Xt, n_test); del Xt; gc.collect()

    del models; gc.collect()

    y_meta = ctx.label[meta_mask]; y_test = ctx.label[test_mask]
    retf_test = ctx.retf("test"); ts_test = ctx.times("test")
    return (p_meta.astype(np.float32), p_test.astype(np.float32),
            y_meta, y_test, retf_test, ts_test)


def train_cat(ctx, extra_raw):
    """CatBoost: 5-seed bagging, 全量训练。"""
    from catboost import CatBoostClassifier, Pool

    t0 = time.time()
    symbol, h = ctx.symbol, ctx.horizon
    print(f"\n{'='*70}\n[3/3] CatBoost  {symbol} H{h}\n{'='*70}", flush=True)

    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")

    models = []
    for seed in BAGGED_SEEDS:
        tr_idx = tr_idx_all.copy()
        train_mask = np.zeros_like(trm, dtype=bool); train_mask[tr_idx] = True
        Xtr = get_X(ctx, extra_raw, train_mask)
        ytr = ctx.label[train_mask].astype(np.float64)

        raw_w = np.abs(train_retf).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)

        params = dict(
            loss_function="Logloss", eval_metric="AUC",
            learning_rate=0.03, depth=7, min_data_in_leaf=150,
            l2_leaf_reg=3, random_seed=seed, verbose=0, thread_count=config.N_JOBS,
        )
        tr = Pool(Xtr, label=ytr, weight=w)
        va = Pool(Xes, label=yes)
        m = CatBoostClassifier(**params, iterations=5000, early_stopping_rounds=200)
        m.fit(tr, eval_set=va, use_best_model=True, verbose=False)
        best_iter = m.get_best_iteration() if m.get_best_iteration() is not None else 0
        print(f"  CAT seed{seed}: best_iter={best_iter} ({time.time()-t0:.0f}s)", flush=True)
        models.append(m)
        del Xtr, tr, ytr, w, train_mask
        gc.collect()

    del Xes, yes, trm, esm, tr_idx_all, train_retf
    gc.collect()

    meta_mask = ctx.split_rows["meta_val"]; test_mask = ctx.split_rows["test"]
    n_meta = meta_mask.sum(); n_test = test_mask.sum()

    Xm = get_X(ctx, extra_raw, meta_mask)
    p_meta = _predict_rank_avg(models, Xm, n_meta); del Xm; gc.collect()
    Xt = get_X(ctx, extra_raw, test_mask)
    p_test = _predict_rank_avg(models, Xt, n_test); del Xt; gc.collect()

    del models; gc.collect()

    y_meta = ctx.label[meta_mask]; y_test = ctx.label[test_mask]
    retf_test = ctx.retf("test"); ts_test = ctx.times("test")
    return (p_meta.astype(np.float32), p_test.astype(np.float32),
            y_meta, y_test, retf_test, ts_test)


def do_stacking(p_lgb_m, p_xgb_m, p_cat_m, p_lgb_t, p_xgb_t, p_cat_t, y_meta):
    """线性堆叠: 3模型概率作为特征, meta_val上拟合逻辑回归。返回 meta_val+test 的融合概率。"""
    from sklearn.linear_model import LogisticRegression
    X_meta = np.column_stack([p_lgb_m, p_xgb_m, p_cat_m]).astype(np.float64)
    X_test = np.column_stack([p_lgb_t, p_xgb_t, p_cat_t]).astype(np.float64)

    stack = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    stack.fit(X_meta, y_meta.astype(np.int8))
    print(f"\n[stacking] 系数: lgb={stack.coef_[0][0]:.3f} xgb={stack.coef_[0][1]:.3f} cat={stack.coef_[0][2]:.3f} 截距={stack.intercept_[0]:.3f}", flush=True)
    p_meta = stack.predict_proba(X_meta)[:, 1]
    p_test = stack.predict_proba(X_test)[:, 1]
    return p_meta.astype(np.float32), p_test.astype(np.float32), stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", choices=["ETH", "BTC"])
    ap.add_argument("horizon", type=int)
    a = ap.parse_args()

    symbol = a.symbol; h = a.horizon
    t0 = time.time()
    print(f"\n{'#'*70}")
    print(f"  串行三模型 + 堆叠融合   {symbol}  H={h}   (进程内严格内存隔离)")
    print(f"{'#'*70}", flush=True)

    # ---- 一次加载 ctx + extra_raw, 三模型共用, 训练完释放 ----
    print(f"\n[0/3] 加载数据 + 计算新增特征...", flush=True)
    ctx = AssetContext(symbol, horizon=h)
    print(f"  ctx: {len(ctx.ds_ts)} rows, {len(ctx.feat_names)} base feats", flush=True)
    extra_raw = compute_extra_raw(ctx)

    # ---- 依次训练三模型, 每个仅保留 (p_meta, p_test, y_meta, y_test, retf_test, ts_test) ----
    # LGB
    p_lgb_m, p_lgb_t, y_meta, y_test, retf_test, ts_test = train_lgb_enhanced(ctx, extra_raw)
    gc.collect()
    print(f"  [内存] LGB 后 gc done", flush=True)

    # XGB
    p_xgb_m, p_xgb_t, *_ = train_xgb(ctx, extra_raw)
    gc.collect()
    print(f"  [内存] XGB 后 gc done", flush=True)

    # Cat
    p_cat_m, p_cat_t, *_ = train_cat(ctx, extra_raw)
    gc.collect()
    print(f"  [内存] CAT 后 gc done", flush=True)

    # ---- 释放 ctx + extra_raw (大对象), 仅保留预测值 ----
    del extra_raw
    # 保留 y_meta / y_test / retf_test / ts_test 用于评估
    del ctx
    gc.collect()
    print(f"  [内存] 释放 ctx + extra_raw, 仅保留 3 组预测 + 评估信息", flush=True)

    # ---- 堆叠 ----
    p_stack_m, p_stack_t, stack = do_stacking(p_lgb_m, p_xgb_m, p_cat_m,
                                              p_lgb_t, p_xgb_t, p_cat_t, y_meta)
    del stack, p_lgb_m, p_xgb_m, p_cat_m, p_lgb_t, p_xgb_t, p_cat_t
    gc.collect()

    # ---- 评估 ----
    print(f"\n{'='*70}\n  测试集 top1% 准确率评估\n{'='*70}", flush=True)
    for name, pt in [("LGB 增强", None), ("XGBoost", None), ("CatBoost", None), ("STACK 融合", p_stack_t)]:
        if pt is None:
            continue
        r = evaluate_topk(pt, y_test, retf_test, ts_test)
        bar = "█" * int(r['accuracy'] * 60)
        print(f"  [{name:11s}] acc={r['accuracy']:.4f}  "
              f"k={r['k']:5d}  trades/day={r['trades_per_day']:5.1f}  "
              f"avg_ret={r['avg_ret_bps']:7.1f}bps  {bar}")
        # 月度最低
        if r['acc_by_month']:
            m_min = min(r['acc_by_month'], key=r['acc_by_month'].get)
            print(f"            月度最低: {r['acc_by_month'][m_min]:.4f} ({m_min[-7:]})")

    # 阈值校准 (meta_val 上选 top1% 对应阈值, 避免 test 泄漏)
    meta_mask_pos = slice(None)  # p_stack_m 就是 meta_val 的全部行
    conf_m = np.maximum(p_stack_m, 1 - p_stack_m)
    k_m = max(1, int(len(p_stack_m) * config.COVERAGE))
    thresh = np.partition(conf_m, -k_m)[-k_m]
    print(f"\n[threshold] meta_val top1% 对应置信度阈值: {thresh:.4f}", flush=True)

    # 保存关键结果
    out = {
        "symbol": symbol, "horizon": h,
        "threshold_top1pct": float(thresh),
        "stack_test_accuracy": float(evaluate_topk(p_stack_t, y_test, retf_test, ts_test)['accuracy']),
        "elapsed": time.time() - t0,
    }
    os.makedirs(f"{config.PROJECT_DIR}/results", exist_ok=True)
    with open(f"{config.PROJECT_DIR}/results/{symbol}_h{h}_stacking_result.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f"{config.DS_DIR}/{symbol}_h{h}_STACK_test_p.npy", p_stack_t)
    print(f"\n  结果保存: results/{symbol}_h{h}_stacking_result.json", flush=True)
    print(f"  预测保存: data/datasets/{symbol}_h{h}_STACK_test_p.npy", flush=True)
    print(f"\n总耗时: {time.time()-t0:.0f}s", flush=True)

    # 释放
    del p_stack_m, p_stack_t, y_meta, y_test, retf_test, ts_test
    gc.collect()
    print(f"全部完成.", flush=True)


if __name__ == "__main__":
    main()
