#!/usr/bin/env python3
"""省内存 LGB: 单币种训练 + 推理保存 P。用法: train_lgb_lean.py {ETH|BTC} {seed}"""
import os, sys, gc, time
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext
from validate_eth_quick import FEATURES, EXTRA_FEATURE_NAMES, compute_extra_raw, get_X
import lightgbm as lgb


def train_sym(sym, seed, horizon=15):
    root = f"/workspace/models_saved/pool_short_h{horizon}"
    os.makedirs(root, exist_ok=True)
    t0 = time.time()
    print(f"[{sym} h{horizon}] LGB seed{seed} 开始 (FEATURES={len(FEATURES)})", flush=True)

    ctx = AssetContext(sym, horizon=horizon, ds_name=f"ds_{sym}_h{horizon}")
    extra = compute_extra_raw(ctx)

    Xes = get_X(ctx, extra, ctx.split_rows["early_stop"])
    yes = ctx.label[ctx.split_rows["early_stop"]].astype(np.float64)
    print(f"  ES: {Xes.shape}", flush=True)

    tr_mask = ctx.split_rows["train"]
    Xtr = get_X(ctx, extra, tr_mask)
    ytr = ctx.label[tr_mask].astype(np.float64)
    wtr = np.clip(np.abs(ctx.retf("train")) * 50, 0.5, 5.0)
    print(f"  TR: {Xtr.shape}", flush=True)

    mp = f"{root}/{sym}_lgb_seed{seed}.txt"
    if os.path.exists(mp):
        print(f"  [SKIP] exists", flush=True)
    else:
        params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                  "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": seed}
        tr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        va = lgb.Dataset(Xes, label=yes, reference=tr)
        m = lgb.train(params, tr, num_boost_round=3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        m.save_model(mp)
        print(f"  best_iter={m.best_iteration} auc={m.best_score['valid_0']['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    m = lgb.Booster(model_file=mp)
    for split in ["meta_val", "test"]:
        mask = ctx.split_rows[split]
        Xt = get_X(ctx, extra, mask)
        raw = m.predict(Xt)
        np.save(f"{config.DS_DIR}/SHORT_{sym}_h{horizon}_lgb_seed{seed}_{split}_P.npy",
                raw.astype(np.float32))
        print(f"  {split} P saved", flush=True)

    del m, ctx, extra, Xtr, ytr, wtr, Xes, yes
    gc.collect()
    print(f"[{sym}] seed{seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sym", choices=["ETH", "BTC"])
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    train_sym(a.sym, a.seed)
