#!/usr/bin/env python3
"""快版训练: 只训 seed42, 跳过 49/56/63/70. 用于相关性分析 (不需要最准)."""
import os, sys, gc, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES,
                                compute_extra_raw, get_X)

MAX_TRAIN = 2_600_000
SYMBOLS = ["ETH", "BTC"]
SEED = 42  # 只训这一个

def model_root(horizon):
    r = f"/workspace/models_saved/pool_short_h{horizon}"
    os.makedirs(r, exist_ok=True)
    return r

def load_ctxs(horizon):
    return {s: AssetContext(s, horizon=horizon, ds_name=f"ds_{s}_h{horizon}") for s in SYMBOLS}

def compute_extras(ctxs):
    return {s: compute_extra_raw(c) for s, c in ctxs.items()}

def joint_masks_weights(ctxs, seed):
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

def train_one(family, horizon):
    root = model_root(horizon)
    t0 = time.time()
    print(f"[h{horizon}] {family} seed{SEED} 开始", flush=True)
    ctxs = load_ctxs(horizon)
    extras = compute_extras(ctxs)
    Xes_list = [get_X(ctxs[s], extras[s], ctxs[s].split_rows["early_stop"]) for s in SYMBOLS]
    yes_list = [ctxs[s].label[ctxs[s].split_rows["early_stop"]].astype(np.float64) for s in SYMBOLS]
    Xes = np.concatenate(Xes_list, axis=0); yes = np.concatenate(yes_list, axis=0)
    del Xes_list, yes_list; gc.collect()

    mp = f"{root}/JOINT_{family}_seed{SEED}.{'txt' if family=='lgb' else 'json' if family=='xgb' else 'cbm'}"
    if os.path.exists(mp):
        print(f"  [SKIP] 已存在: {mp}", flush=True)
        return mp

    mws = joint_masks_weights(ctxs, SEED)
    Xtr_list = [get_X(ctxs[s], extras[s], mws[s][0]) for s in SYMBOLS]
    ytr_list = [ctxs[s].label[mws[s][0]].astype(np.float64) for s in SYMBOLS]
    wtr_list = [mws[s][1] for s in SYMBOLS]
    Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list); wtr = np.concatenate(wtr_list)
    del Xtr_list, ytr_list, wtr_list; gc.collect()

    Xva_list = [get_X(ctxs[s], extras[s], ctxs[s].split_rows["val"]) for s in SYMBOLS]
    yva_list = [ctxs[s].label[ctxs[s].split_rows["val"]].astype(np.float64) for s in SYMBOLS]
    Xva = np.concatenate(Xva_list); yva = np.concatenate(yva_list)
    del Xva_list, yva_list; gc.collect()

    params = {
        "lgb": {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1},
        "xgb": {"objective": "binary:logistic", "eval_metric": "auc",
                "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 200,
                "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                "nthread": 8},
        "cat": {"loss_function": "Logloss", "eval_metric": "AUC",
                "learning_rate": 0.05, "depth": 8, "min_data_in_leaf": 200,
                "l2_leaf_reg": 3, "random_seed": SEED, "verbose": 0},
    }[family]

    if family == "lgb":
        import lightgbm as lgb
        tr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        va = lgb.Dataset(Xva, label=yva, reference=tr)
        m = lgb.train(params, tr, num_boost_round=3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        m.save_model(mp)
    elif family == "xgb":
        import xgboost as xgb
        tr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        va = xgb.DMatrix(Xva, label=yva)
        m = xgb.train(params, tr, num_boost_round=3000,
                      evals=[(va, "val")], early_stopping_rounds=100, verbose_eval=False)
        m.save_model(mp)
    else:
        from catboost import CatBoostClassifier, Pool
        tr = Pool(Xtr, label=ytr, weight=wtr)
        va = Pool(Xva, label=yva)
        m = CatBoostClassifier(**params, iterations=3000, early_stopping_rounds=100)
        m.fit(tr, eval_set=va, use_best_model=True, verbose=False)
        m.save_model(mp)

    # 保存 test P
    for sym in SYMBOLS:
        ctx = ctxs[sym]
        extras_s = extras[sym]
        for split in ["meta_val", "test"]:
            mask = ctx.split_rows[split]
            Xt = get_X(ctx, extras_s, mask)
            raw = m.predict(Xt)
            p = 1.0 / (1.0 + np.exp(-raw))
            p_path = f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_{family}_{split}_P.npy"
            np.save(p_path, p.astype(np.float32))
            print(f"  [{family}] {sym} {split} P saved ({p.shape})", flush=True)

    print(f"[h{horizon}] {family} 完成 {time.time()-t0:.0f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("horizon", type=int)
    ap.add_argument("family", choices=["lgb", "xgb", "cat"])
    a = ap.parse_args()
    train_one(a.family, a.horizon)

if __name__ == "__main__":
    main()
