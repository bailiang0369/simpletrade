#!/usr/bin/env python3
"""实验: LightGBM Lambdarank 直接优化 top-k 排序质量。

核心思路:
- 当前 GBDT 优化的是二元分类 (AUC), 但评估的是 top-1% 排序准确率
- 任务目标函数不一致 → 用 lambdarank 直接优化 NDCG@k
- 按天分组, 每 24h 内样本作为一个 query group
- 标签按未来收益分 5 级 relevance, 模型学习把高收益样本排在顶部

输出: 打印 test 上 top-1% 准确率, 与当前 baseline (0.617) 对比。
"""
import os, sys, gc, time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

# 与 gbdt.py 完全一致的特征集
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

# ---- 5 级 relevance 分桶阈值 (基于 ret_future 的 z-score) ----
RELEVANCE_BINS = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]


def compute_relevance(retf):
    """将未来收益映射到 5 级 relevance (0-4)。"""
    fin = np.isfinite(retf)
    z = np.zeros_like(retf, dtype=np.float64)
    if fin.sum() > 10:
        mu = np.nanmean(retf)
        sd = np.nanstd(retf) + 1e-10
        z[fin] = (retf[fin] - mu) / sd
    return np.digitize(z, [-1.0, -0.3, 0.3, 1.0]).astype(np.int32)  # [0,4]


def build_daily_groups(ts_sec):
    """按天构建 query group。ts_sec: int64 epoch 秒。返回 group 边界数组。"""
    days = ts_sec // 86400
    _, counts = np.unique(days, return_counts=True)
    return counts.astype(np.int32)


def train_lambdarank(ctx, seeds=None, max_train=2_600_000, n_rounds=5000):
    """训练多个 seed 的 lambdarank 模型, 返回预测结果。"""
    if seeds is None:
        seeds = [42, 49, 56, 63, 70]
    
    t0 = time.time()
    trm = ctx.split_rows["train"]
    esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    
    # 早停集特征
    Xes = ctx.X_subset(FEATURES, esm)
    retf_es = ctx.retf("early_stop")
    rel_es = compute_relevance(retf_es)
    # 早停集 daily group
    ts_es = ctx.times("early_stop").astype("datetime64[s]").astype(np.int64)
    grp_es = build_daily_groups(ts_es)
    
    # 训练集全局 times
    ts_tr_all = ctx.times("train").astype("datetime64[s]").astype(np.int64)
    retf_tr_all = ctx.retf("train")
    
    models = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        has_sampling = len(tr_idx) > max_train
        if has_sampling:
            keep = rng.choice(len(tr_idx), max_train, replace=False)
            tr_idx = tr_idx[keep]
        
        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True
        
        Xtr = ctx.X_subset(FEATURES, train_mask)
        
        # 正确获取采样后的 retf 和 times
        if has_sampling:
            keep_local = np.where(train_mask[tr_idx_all])[0]
            retf_tr = retf_tr_all[keep_local]
            ts_tr = ts_tr_all[keep_local]
        else:
            retf_tr = retf_tr_all
            ts_tr = ts_tr_all
        
        # relevance 标签
        rel_tr = compute_relevance(retf_tr)
        
        # 训练集 daily group
        grp_tr = build_daily_groups(ts_tr)
        
        # 权重: 强信号放大
        w = np.ones(len(rel_tr), dtype=np.float64)
        w[rel_tr == 0] = 2.0   # 强下跌
        w[rel_tr == 4] = 3.0   # 强上涨 (最重要)
        w[rel_tr == 3] = 1.5   # 弱上涨
        
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 5, 10],
            "label_gain": [0, 1, 2, 3, 4],
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 2,
            "min_data_in_leaf": 100,
            "lambda_l1": 0.05,
            "lambda_l2": 1.0,
            "num_threads": config.N_JOBS,
            "verbosity": -1,
            "seed": seed,
            "min_data": 1,
        }
        
        dtr = lgb.Dataset(Xtr, label=rel_tr, group=grp_tr, weight=w)
        des = lgb.Dataset(Xes, label=rel_es, group=grp_es, reference=dtr)
        
        m = lgb.train(
            params, dtr,
            num_boost_round=n_rounds,
            valid_sets=[des],
            valid_names=["early_stop"],
            callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                       lgb.log_evaluation(0)],
        )
        models.append(m)
        
        ndcg = m.best_score["early_stop"].get("ndcg@1", -1)
        print(f"  seed{seed} best_iter={m.best_iteration} ndcg@1={ndcg:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del Xtr, dtr
        gc.collect()
    
    print(f"  lambdarank {len(models)} seeds done in {time.time()-t0:.0f}s", flush=True)
    return models


def predict_rank(models, ctx, split):
    """排名平均: 每棵树的 raw score 转百分位秩再平均, 返回 float64 分数。"""
    mask = ctx.split_rows[split]
    X = ctx.X_subset(FEATURES, mask)
    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        s = m.predict(X, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(s)).astype(np.float64) / (n - 1)
    return np.asarray(R.mean(axis=0), dtype=np.float64)


def main():
    symbol = "BTC"
    print(f"\n{'='*65}")
    print(f"  Lambdarank 实验: {symbol} (H30)")
    print(f"  Baseline: test top1% acc ≈ 0.617")
    print(f"{'='*65}\n")
    
    ctx = AssetContext(symbol, horizon=30)
    
    # 训练
    models = train_lambdarank(ctx)
    
    # 评估 test
    pt = predict_rank(models, ctx, "test")
    y = ctx.y("test")
    rf = ctx.retf("test")
    ts = ctx.times("test")
    
    r = evaluate_topk(pt, y, rf, ts)
    print(f"\n  >>> Lambdarank test: acc={r['accuracy']:.4f}  "
          f"tpd={r['trades_per_day']:.1f}  k={r['k']}  "
          f"avg_ret={r['avg_ret_bps']:.1f}bps")
    
    np.save(f"{config.DS_DIR}/{symbol}_lambdarank_pt.npy", pt.astype(np.float32))
    print(f"\n  [saved] {config.DS_DIR}/{symbol}_lambdarank_pt.npy")
    
    del ctx, models
    gc.collect()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()