#!/usr/bin/env python3
"""多资产联合训练: ETH+BTC 合并数据训练 pool20 (LGBM5+XGB5+CAT5), 分别评估两资产。

动机: ETH 单独训练时 test 存在 ~52% 的坏月 (2025-12), BTC 训练良好但跨资产迁移后
ETH 坏月转移到 2025-10/11。联合训练让模型同时学习两资产共同结构, 期望稳定 ETH。

内存受限: 每进程一个 family, 模型/预测落盘。用法:
  python experiment_pool20_joint.py train lgb
  python experiment_pool20_joint.py train xgb
  python experiment_pool20_joint.py train cat
  python experiment_pool20_joint.py eval
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES, BAGGED_SEEDS,
                                compute_extra_raw, get_X)

MAX_TRAIN = 2_600_000
MODEL_ROOT = os.path.join(config.MODEL_DIR, "pool20_joint")
os.makedirs(MODEL_ROOT, exist_ok=True)
SYMBOLS = ["ETH", "BTC"]


def load_ctxs():
    return {s: AssetContext(s, horizon=30) for s in SYMBOLS}


def compute_extras(ctxs):
    return {s: compute_extra_raw(c) for s, c in ctxs.items()}


def joint_masks_weights(ctxs, seed):
    """从两资产 train 段合并采样, 返回每资产的 (mask, weight)。"""
    outs = {}
    for s in SYMBOLS:
        ctx = ctxs[s]
        trm = ctx.split_rows["train"]
        tr_idx_all = np.where(trm)[0]
        rng = np.random.default_rng(seed)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > MAX_TRAIN // 2:
            tr_idx = rng.choice(len(tr_idx), MAX_TRAIN // 2, replace=False)
        mask = np.zeros_like(trm, dtype=bool); mask[tr_idx] = True
        keep_local = np.where(mask[tr_idx_all])[0]
        raw_w = np.abs(ctx.retf("train")[keep_local]).astype(np.float64)
        w = np.clip(raw_w * 50, 0.5, 5.0)
        outs[s] = (mask, w)
    return outs


def train_family(family):
    t0 = time.time()
    print(f"  [JOINT] {family} 训练开始", flush=True)
    ctxs = load_ctxs()
    extras = compute_extras(ctxs)
    # early_stop 拼接
    Xes_list = [get_X(ctxs[s], extras[s], ctxs[s].split_rows["early_stop"]) for s in SYMBOLS]
    yes_list = [ctxs[s].label[ctxs[s].split_rows["early_stop"]].astype(np.float64) for s in SYMBOLS]
    Xes = np.concatenate(Xes_list, axis=0); yes = np.concatenate(yes_list, axis=0)
    del Xes_list, yes_list; gc.collect()
    bis = []
    for seed in BAGGED_SEEDS:
        mws = joint_masks_weights(ctxs, seed)
        Xtr_list = [get_X(ctxs[s], extras[s], mws[s][0]) for s in SYMBOLS]
        ytr_list = [ctxs[s].label[mws[s][0]].astype(np.float64) for s in SYMBOLS]
        w_list = [mws[s][1] for s in SYMBOLS]
        Xtr = np.concatenate(Xtr_list, axis=0); ytr = np.concatenate(ytr_list, axis=0)
        w = np.concatenate(w_list, axis=0)
        del Xtr_list, ytr_list, w_list; gc.collect()
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
            m.save_model(f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.txt")
            bis.append(m.best_iteration)
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
            m.save_model(f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.json")
            bis.append(m.best_iteration)
        else:
            from catboost import CatBoostClassifier, Pool
            eval_pool = Pool(Xes, yes)
            tr_pool = Pool(Xtr, ytr, weight=w)
            m = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=1.0,
                                   random_seed=seed, task_type="CPU", thread_count=config.N_JOBS,
                                   verbose=False, early_stopping_rounds=200, loss_function="Logloss",
                                   allow_writing_files=False)
            m.fit(tr_pool, eval_set=eval_pool, verbose_eval=False)
            m.save_model(f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.cbm")
            bis.append(m.best_iteration_)
        print(f"  [JOINT {family}] seed{seed} iter={bis[-1]} ({time.time()-t0:.0f}s)", flush=True)
        del Xtr, ytr, w; gc.collect()
        if family in ("lgb", "xgb"):
            del dtr; gc.collect()
    del Xes, yes; gc.collect()
    # ---- 预测两资产 meta_val/test 落盘 ----
    for s in SYMBOLS:
        for split in ("meta_val", "test"):
            mask = ctxs[s].split_rows[split]
            X = get_X(ctxs[s], extras[s], mask); n = len(X)
            P = np.zeros((5, n), dtype=np.float32)
            for i, seed in enumerate(BAGGED_SEEDS):
                bi = bis[i]
                if family == "lgb":
                    import lightgbm as lgb
                    mm = lgb.Booster(model_file=f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.txt")
                    P[i] = mm.predict(X, num_iteration=bi)
                elif family == "xgb":
                    import xgboost as xgb
                    mm = xgb.Booster(); mm.load_model(f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.json")
                    P[i] = mm.predict(xgb.DMatrix(X), iteration_range=(0, bi))
                else:
                    from catboost import CatBoostClassifier
                    mm = CatBoostClassifier(); mm.load_model(f"{MODEL_ROOT}/JOINT_{family}_seed{seed}.cbm")
                    P[i] = mm.predict(X, prediction_type="Probability")[:, 1]
                del mm; gc.collect()
            np.save(f"{config.DS_DIR}/JOINT_{s}_{family}_{split}_P.npy", P)
            print(f"  [JOINT {family}] {s} {split} P saved ({time.time()-t0:.0f}s)", flush=True)
            del X, P; gc.collect()
    print(f"  [JOINT] {family} 完成 {time.time()-t0:.0f}s", flush=True)


def _topk_acc_eval(preds, train_data):
    labels = train_data.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


def eval_daily(p, y, sec_arr):
    n = len(p); pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
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


def evaluate():
    ctxs = load_ctxs()
    for s in SYMBOLS:
        ctx = ctxs[s]
        for split in ("meta_val", "test"):
            Ps = [np.load(f"{config.DS_DIR}/JOINT_{s}_{f}_{split}_P.npy") for f in ("lgb", "xgb", "cat")]
            P = np.concatenate(Ps, axis=0); n = P.shape[1]
            R = np.zeros_like(P, dtype=np.float64)
            for i in range(P.shape[0]):
                R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
            p = R.mean(axis=0)
            y = ctx.y(split)
            sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            print(f"\n===== JOINT {s} {split} rows={n} n_days={np.unique(sec_arr//86400).size} =====")
            for mode in ("global", "daily"):
                if mode == "global":
                    k = max(1, int(round(n * 0.01)))
                    sel = np.argsort(-np.maximum(p, 1 - p))[:k]
                    pred = (p >= 0.5).astype(np.int8)
                    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
                    uniq = np.unique(mts)
                    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq}
                    min_k = min(int((mts == u).sum()) for u in uniq)
                    min_a = min(acc_m.values())
                    nbad = sum(1 for a in acc_m.values() if a < 0.55)
                    acc = float((pred[sel] == y[sel]).mean())
                    tpd = float(sel.size) / np.unique(sec_arr // 86400).size
                    print(f"[global] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f}")
                    print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
                else:
                    acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
                    print(f"[daily ] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f} cov={cov:.4f}")
                    print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
            del Ps, P, R, p; gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("family", nargs="?", choices=["lgb", "xgb", "cat"])
    args = ap.parse_args()
    if args.mode == "train":
        train_family(args.family)
    else:
        evaluate()


if __name__ == "__main__":
    main()
