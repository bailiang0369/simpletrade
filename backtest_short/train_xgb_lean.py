#!/usr/bin/env python3
"""省内存 XGB: 单币种训练 + 推理保存 P。用法: train_xgb_lean.py {ETH|BTC} {seed}"""
import os, sys, gc, time
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext
from validate_eth_quick import FEATURES, EXTRA_FEATURE_NAMES, compute_extra_raw, get_X
import xgboost as xgb


def train_sym(sym, seed, horizon=15):
    root = f"/workspace/models_saved/pool_short_h{horizon}"
    os.makedirs(root, exist_ok=True)
    t0 = time.time()
    print(f"[{sym} h{horizon}] XGB seed{seed} 开始 (FEATURES={len(FEATURES)})", flush=True)

    ctx = AssetContext(sym, horizon=horizon, ds_name=f"ds_{sym}_h{horizon}")
    extra = compute_extra_raw(ctx)

    Xes = get_X(ctx, extra, ctx.split_rows["early_stop"])
    yes = ctx.label[ctx.split_rows["early_stop"]].astype(np.float64)

    tr_mask = ctx.split_rows["train"]
    Xtr = get_X(ctx, extra, tr_mask)
    ytr = ctx.label[tr_mask].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("train")) * 50, 0.5, 5.0)
    print(f"  TR: {Xtr.shape} ES: {Xes.shape}", flush=True)

    mp = f"{root}/{sym}_xgb_seed{seed}.json"
    if os.path.exists(mp):
        print(f"  [SKIP] exists", flush=True)
    else:
        params = {"objective": "binary:logistic", "eval_metric": "auc",
                  "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 200,
                  "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                  "nthread": 3, "seed": seed}
        tr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        va = xgb.DMatrix(Xes, label=yes)
        m = xgb.train(params, tr, num_boost_round=3000,
                      evals=[(va, "val")], early_stopping_rounds=100, verbose_eval=False)
        m.save_model(mp)
        print(f"  best_iter={m.best_iteration} auc={m.best_score:.4f} ({time.time()-t0:.0f}s)", flush=True)

    m = xgb.Booster(); m.load_model(mp)
    for split in ["meta_val", "test"]:
        mask = ctx.split_rows[split]
        Xt = get_X(ctx, extra, mask)
        raw = m.predict(xgb.DMatrix(Xt))
        np.save(f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_xgb_seed{seed}_{split}_P.npy",
                raw.astype(np.float32))
        print(f"  {split} P saved", flush=True)

    del m, ctx, extra, Xtr, ytr, wtr, Xes, yes
    gc.collect()
    print(f"[{sym}] XGB seed{seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sym", choices=["ETH", "BTC"])
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    train_sym(a.sym, a.seed)
