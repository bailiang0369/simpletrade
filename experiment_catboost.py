#!/usr/bin/env python3
"""实验: CatBoost + 标签过滤 — 两种方法提升 top-1% 准确率。

方法1: CatBoost 默认训练 (ordered boosting + 对称树)
方法2: CatBoost + 标签过滤 (只保留|z_ret|>0.5 的高信号样本训练)
"""
import os, sys, gc, time
import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

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
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]


def train_catboost(ctx, seed=42, use_filter=False, max_train=2_600_000):
    """训练 CatBoost 模型。
    use_filter=True: 只保留 |z_ret| > 0.5 的高信号样本。
    """
    t0 = time.time()
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    
    Xes = ctx.X_subset(FEATURES, esm)
    yes = ctx.label[esm].astype(int)
    
    # 训练集采样
    rng = np.random.default_rng(seed)
    tr_idx = tr_idx_all.copy()
    if len(tr_idx) > max_train:
        keep = rng.choice(len(tr_idx), max_train, replace=False)
        tr_idx = tr_idx[keep]
    
    train_mask = np.zeros_like(trm, dtype=bool)
    train_mask[tr_idx] = True
    
    Xtr = ctx.X_subset(FEATURES, train_mask)
    ytr = ctx.label[train_mask].astype(int)
    retf_tr = ctx.retf("train")[np.where(train_mask[tr_idx_all])[0]] if len(tr_idx) < len(tr_idx_all) else ctx.retf("train")
    
    # 可选: 标签过滤 - 只保留 |z_ret| > 0.5 的高信号样本
    if use_filter:
        ret_z = (retf_tr - retf_tr.mean()) / (retf_tr.std() + 1e-10)
        keep = np.abs(ret_z) > 0.5
        print(f"  [filter] 保留 {keep.sum()}/{len(keep)} 高信号样本 ({keep.mean()*100:.1f}%)", flush=True)
        Xtr = Xtr[keep]
        ytr = ytr[keep]
    
    # 权重: 正样本放大
    w = np.ones(len(ytr), dtype=np.float64)
    w[ytr == 1] = 2.0
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        border_count=128,
        random_seed=seed,
        thread_count=config.N_JOBS,
        verbose=0,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=200,
        early_stopping_rounds=200,
        use_best_model=True,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    
    model.fit(
        Xtr, ytr,
        eval_set=(Xes, yes),
        sample_weight=w,
        verbose=0,
    )
    
    best_iter = model.get_best_iteration() or 0
    best_score = model.get_best_score()["validation"]["AUC"] if "validation" in model.get_best_score() else -1
    print(f"  seed{seed} best_iter={best_iter} auc={best_score:.4f} ({time.time()-t0:.0f}s)", flush=True)
    
    return model


def predict_catboost(model, ctx, split):
    mask = ctx.split_rows[split]
    X = ctx.X_subset(FEATURES, mask)
    return model.predict_proba(X)[:, 1].astype(np.float64)


def main():
    symbol = "BTC"
    print(f"\n{'='*65}")
    print(f"  CatBoost 实验: {symbol} (H30)")
    print(f"  Baseline: test top1% acc ≈ 0.617")
    print(f"{'='*65}\n")
    
    ctx = AssetContext(symbol, horizon=30)
    
    # 方法1: CatBoost 默认
    print("\n--- 方法1: CatBoost 默认 ---")
    m1 = train_catboost(ctx, seed=42)
    p1 = predict_catboost(m1, ctx, "test")
    r1 = evaluate_topk(p1, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >>> CatBoost test: acc={r1['accuracy']:.4f}  tpd={r1['trades_per_day']:.1f}  avg_ret={r1['avg_ret_bps']:.1f}bps")
    del m1; gc.collect()
    
    # 方法2: CatBoost + 标签过滤
    print("\n--- 方法2: CatBoost + 标签过滤(只保留|z_ret|>0.5) ---")
    m2 = train_catboost(ctx, seed=42, use_filter=True)
    p2 = predict_catboost(m2, ctx, "test")
    r2 = evaluate_topk(p2, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >>> CatBoost+filter test: acc={r2['accuracy']:.4f}  tpd={r2['trades_per_day']:.1f}  avg_ret={r2['avg_ret_bps']:.1f}bps")
    del m2; gc.collect()
    
    # 方法3: 多seed CatBoost bagging
    print("\n--- 方法3: CatBoost 5-seed bagging ---")
    models = []
    for seed in [42, 49, 56, 63, 70]:
        m = train_catboost(ctx, seed=seed, use_filter=False)
        models.append(m)
    pt = np.zeros(len(ctx.y("test")), dtype=np.float64)
    for m in models:
        pt += predict_catboost(m, ctx, "test")
    pt /= len(models)
    r3 = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >>> CatBoost bagging test: acc={r3['accuracy']:.4f}  tpd={r3['trades_per_day']:.1f}  avg_ret={r3['avg_ret_bps']:.1f}bps")
    del models; gc.collect()
    
    # 方法4: 多seed + 标签过滤 bagging
    print("\n--- 方法4: CatBoost 5-seed bagging + 标签过滤 ---")
    models = []
    for seed in [42, 49, 56, 63, 70]:
        m = train_catboost(ctx, seed=seed, use_filter=True)
        models.append(m)
    pt = np.zeros(len(ctx.y("test")), dtype=np.float64)
    for m in models:
        pt += predict_catboost(m, ctx, "test")
    pt /= len(models)
    r4 = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >>> CatBoost+filter bagging test: acc={r4['accuracy']:.4f}  tpd={r4['trades_per_day']:.1f}  avg_ret={r4['avg_ret_bps']:.1f}bps")
    
    del ctx; gc.collect()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()