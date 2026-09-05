"""跨族集成池 (LGBM×5 + XGB×5 + CatBoost×5) 在 ETH 与 BTC 上验证 每日局部 top-1%。

内存受限(4GB cgroup): 分阶段执行, 每进程只训练一个 family (5 个模型), 模型与预测落盘,
集成与评估单独进程加载, 避免单进程同时持有 15 个模型 + 训练数据导致 OOM。

用法:
  python experiment_pool20.py train ETH lgb    # 训练一个 family 并保存模型+预测
  python experiment_pool20.py train ETH xgb
  python experiment_pool20.py train ETH cat
  python experiment_pool20.py eval  ETH        # 加载全部 15 组预测, rank_mean 集成 + 评估
  python experiment_pool20.py all  ETH         # 依次执行 train lgb/xgb/cat + eval
"""
import os, sys, gc, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES, BAGGED_SEEDS,
                                compute_extra_raw, get_X)

MAX_TRAIN = 2_600_000
MODEL_ROOT = "/workspace/models_saved/pool20_rerun"
os.makedirs(MODEL_ROOT, exist_ok=True)


def _train_masks_weights(ctx, seed):
    trm = ctx.split_rows["train"]
    tr_idx_all = np.where(trm)[0]
    rng = np.random.default_rng(seed)
    tr_idx = tr_idx_all.copy()
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = rng.choice(len(tr_idx), MAX_TRAIN, replace=False)
    train_mask = np.zeros_like(trm, dtype=bool)
    train_mask[tr_idx] = True
    keep_local = np.where(train_mask[tr_idx_all])[0]
    raw_w = np.abs(ctx.retf("train")[keep_local]).astype(np.float64)
    w = np.clip(raw_w * 50, 0.5, 5.0)
    return train_mask, w


def train_family(symbol, family):
    t0 = time.time()
    print(f"  [{symbol}] {family} 训练开始", flush=True)
    ctx = AssetContext(symbol, horizon=30)
    extra_raw = compute_extra_raw(ctx)
    esm = ctx.split_rows["early_stop"]
    Xes = get_X(ctx, extra_raw, esm); yes = ctx.label[esm].astype(np.float64)
    models = []
    for seed in BAGGED_SEEDS:
        tr_mask, w = _train_masks_weights(ctx, seed)
        Xtr = get_X(ctx, extra_raw, tr_mask); ytr = ctx.label[tr_mask].astype(np.float64)
        if family == "lgb":
            import lightgbm as lgb
            p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                     max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                     min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                     num_threads=config.N_JOBS, verbosity=-1, seed=seed)
            dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                          feval=_topk_acc_eval,
                          callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                     lgb.log_evaluation(0)])
            m.save_model(f"{MODEL_ROOT}/{symbol}_{family}_seed{seed}.txt")
            bi = m.best_iteration
        elif family == "xgb":
            import xgboost as xgb
            des = xgb.DMatrix(Xes, label=yes)
            dtr = xgb.DMatrix(Xtr, label=ytr, weight=w)
            p = dict(objective="binary:logistic", eval_metric="auc", eta=0.02, max_depth=8,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                     reg_alpha=0.05, reg_lambda=1.0, tree_method="hist", nthread=config.N_JOBS,
                     seed=seed)
            m = xgb.train(p, dtr, num_boost_round=5000, evals=[(des, "es")],
                          early_stopping_rounds=200, verbose_eval=False)
            m.save_model(f"{MODEL_ROOT}/{symbol}_{family}_seed{seed}.json")
            bi = m.best_iteration
        else:
            from catboost import CatBoostClassifier, Pool
            eval_pool = Pool(Xes, yes)
            tr_pool = Pool(Xtr, ytr, weight=w)
            m = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=1.0,
                                   random_seed=seed, task_type="CPU", thread_count=config.N_JOBS,
                                   verbose=False, early_stopping_rounds=200, loss_function="Logloss",
                                   allow_writing_files=False)
            m.fit(tr_pool, eval_set=eval_pool, verbose_eval=False)
            m.save_model(f"{MODEL_ROOT}/{symbol}_{family}_seed{seed}.cbm")
            bi = m.best_iteration_
        models.append((family, bi))
        print(f"  [{family}] {symbol} seed{seed} iter={bi} ({time.time()-t0:.0f}s)", flush=True)
        del Xtr, ytr, w; gc.collect()
        if family in ("lgb", "xgb"):
            del dtr; gc.collect()
    del Xes, yes; gc.collect()
    # ---- 预测 meta_val / test 并落盘 (R 矩阵按模型逐行保存, 常驻仅单行) ----
    for split in ("meta_val", "test"):
        mask = ctx.split_rows[split]
        X = get_X(ctx, extra_raw, mask); n = len(X)
        P = np.zeros((len(models), n), dtype=np.float32)
        for i, (_, bi) in enumerate(models):
            if family == "lgb":
                import lightgbm as lgb
                mm = lgb.Booster(model_file=f"{MODEL_ROOT}/{symbol}_{family}_seed{BAGGED_SEEDS[i]}.txt")
                P[i] = mm.predict(X, num_iteration=bi)
            elif family == "xgb":
                import xgboost as xgb
                mm = xgb.Booster(); mm.load_model(f"{MODEL_ROOT}/{symbol}_{family}_seed{BAGGED_SEEDS[i]}.json")
                P[i] = mm.predict(xgb.DMatrix(X), iteration_range=(0, bi))
            else:
                from catboost import CatBoostClassifier
                mm = CatBoostClassifier(); mm.load_model(f"{MODEL_ROOT}/{symbol}_{family}_seed{BAGGED_SEEDS[i]}.cbm")
                P[i] = mm.predict(X, prediction_type="Probability")[:, 1]
            del mm; gc.collect()
        np.save(f"{config.DS_DIR}/{symbol}_{family}_{split}_P.npy", P)
        print(f"  [{family}] {symbol} {split} P saved ({time.time()-t0:.0f}s)", flush=True)
        del X, P; gc.collect()
    print(f"  [{symbol}] {family} 完成 {time.time()-t0:.0f}s", flush=True)


def _topk_acc_eval(preds, train_data):
    """LGBM feval: early_stop 上 top-1% 准确率(与仓库一致)。"""
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def eval_policy(p, y, sec_arr, mode):
    n = len(p); pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    if mode == "global":
        k = max(1, int(round(n * 0.01)))
        sel = np.argsort(-conf)[:k]
    else:
        day = sec_arr // 86400; days = np.unique(day); sm = np.zeros(n, bool)
        for d in days:
            md = day == d
            kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
            sub = np.where(md)[0]
            sm[sub[np.argsort(-conf[sub])[:kd]]] = True
        sel = np.where(sm)[0]
    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq}
    min_k = min(int((mts == u).sum()) for u in uniq)
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    n_days = np.unique(sec_arr // 86400).size
    return (float((pred[sel] == y[sel]).mean()), min_a, min_k, nbad,
            float(sel.size) / n_days, float(sel.size) / n, acc_m)


def evaluate(symbol):
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    print(f"\n{'='*70}\n  {symbol} 跨族池 (LGBM5+XGB5+CAT5) 集成评估\n{'='*70}", flush=True)
    for split in ("meta_val", "test"):
        Ps = [np.load(f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy") for f in ("lgb", "xgb", "cat")]
        P = np.concatenate(Ps, axis=0)
        n = P.shape[1]
        p = rank_mean(P, n)
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        print(f"\n===== {symbol} {split} rows={n} n_days={np.unique(sec_arr//86400).size} =====")
        for mode in ("global", "daily"):
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_policy(p, y, sec_arr, mode)
            print(f"[{mode}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) "
                  f"bad(<55)={nbad} tpd={tpd:.2f} cov={cov:.4f}")
            print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
        del Ps, P, p; gc.collect()
    print(f"  [{symbol}] 评估完成 {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "all"])
    ap.add_argument("symbol", choices=["ETH", "BTC"])
    ap.add_argument("family", nargs="?", choices=["lgb", "xgb", "cat"])
    args = ap.parse_args()
    if args.mode == "train":
        train_family(args.symbol, args.family)
    elif args.mode == "eval":
        evaluate(args.symbol)
    else:
        for f in ("lgb", "xgb", "cat"):
            train_family(args.symbol, f)
        evaluate(args.symbol)


if __name__ == "__main__":
    main()
