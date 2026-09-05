"""BTC 同口径验证: 与 ETH 完全相同的 LGBM 池(同特征/超参/早停/评估), 对比全局 vs 每日局部 top1%。
用于回答: 换 BTC 后准确率如何, 以及是否满足 CROSS_ASSET_MAX_DELTA(<=3pp)。
原则: 仅用 BTC 自身数据训练, 不跨资产; mv 只用于观察, 不调参; test 一次性。
"""
import os, sys, gc, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from experiment_floor_test import (BASE_FEATURES, SEEDS, MAX_TRAIN,
                                   topk_acc_eval, train_pool, rank_mean, eval_policy)


def main():
    sym = "BTC"
    t0 = time.time()
    ctx = AssetContext(sym, horizon=30)
    models = train_pool(BASE_FEATURES, ctx)
    print(f"[{sym}] 训练完成 {time.time()-t0:.0f}s", flush=True)

    for split in ("meta_val", "test"):
        p = rank_mean(models, BASE_FEATURES, ctx, split)
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        print(f"\n===== {sym} {split} rows={len(p)} n_days={np.unique(sec_arr//86400).size} =====")
        for mode in ("global", "daily"):
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_policy(p, y, sec_arr, mode)
            print(f"[{mode}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) "
                  f"bad(<55)={nbad} tpd={tpd:.2f} cov={cov:.4f}")
            print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
    print(f"\n[{sym}] 完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()