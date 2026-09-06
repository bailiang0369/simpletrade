#!/usr/bin/env python3
"""用已训练的 JOINT lgb decay 模型 (models_saved/pool20_joint/JOINT_lgb_seed*_d{HL}.txt)
对 ETH/BTC 的 meta_val/test 做预测并落盘 (DS_DIR/JOINT_decay{HL}_{s}_lgb_{split}_P.npy)。
用法: python predict_decay.py 365
"""
import os, sys, gc, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import lightgbm as lgb
from data_store import AssetContext
from validate_eth_quick import compute_extra_raw, get_X, BAGGED_SEEDS

MODEL_ROOT = "/workspace/models_saved/pool20_joint"


def main():
    hl = int(sys.argv[1])
    t0 = time.time()
    tag = f"_decay{hl}"
    mt = f"_d{hl}"
    for s in ("ETH", "BTC"):
        ctx = AssetContext(s, horizon=30)
        extras = compute_extra_raw(ctx)
        for split in ("meta_val", "test"):
            mask = ctx.split_rows[split]
            X = get_X(ctx, extras, mask)
            n = len(X)
            P = np.zeros((len(BAGGED_SEEDS), n), dtype=np.float32)
            for i, seed in enumerate(BAGGED_SEEDS):
                m = lgb.Booster(model_file=f"{MODEL_ROOT}/JOINT_lgb_seed{seed}{mt}.txt")
                P[i] = m.predict(X, num_iteration=m.best_iteration)
                del m; gc.collect()
            np.save(f"{config.DS_DIR}/JOINT{tag}_{s}_lgb_{split}_P.npy", P)
            print(f"  [{s} {split}] saved ({time.time()-t0:.0f}s)", flush=True)
            del X, P; gc.collect()
        del ctx, extras; gc.collect()
    print(f"  done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
