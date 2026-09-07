#!/usr/bin/env python3
"""跨资产验证: 用 BTC 训练的 pool20 模型 (LGBM5+XGB5+CAT5) 预测 ETH 的 meta_val/test。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from validate_eth_quick import compute_extra_raw, get_X

MODEL_ROOT = "/workspace/models_saved/pool20_rerun"
FAMS = ["lgb", "xgb", "cat"]


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


def predict_with_btc_models(ctx, extra_raw, split, family):
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    mask = ctx.split_rows[split]
    X = get_X(ctx, extra_raw, mask); n = len(X)
    P = np.zeros((5, n), dtype=np.float32)
    for i, seed in enumerate([42, 49, 56, 63, 70]):
        if family == "lgb":
            mm = lgb.Booster(model_file=f"{MODEL_ROOT}/BTC_{family}_seed{seed}.txt")
            P[i] = mm.predict(X)
        elif family == "xgb":
            mm = xgb.Booster(); mm.load_model(f"{MODEL_ROOT}/BTC_{family}_seed{seed}.json")
            P[i] = mm.predict(xgb.DMatrix(X))
        else:
            mm = CatBoostClassifier(); mm.load_model(f"{MODEL_ROOT}/BTC_{family}_seed{seed}.cbm")
            P[i] = mm.predict(X, prediction_type="Probability")[:, 1]
        del mm; gc.collect()
    return P


def main():
    ctx = AssetContext("ETH", horizon=30)
    extra_raw = compute_extra_raw(ctx)
    for split in ("meta_val", "test"):
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        Ps = []
        for f in FAMS:
            P = predict_with_btc_models(ctx, extra_raw, split, f)
            Ps.append(P)
            n = P.shape[1]
            R = np.zeros_like(P, dtype=np.float64)
            for i in range(P.shape[0]):
                R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
            p = R.mean(axis=0)
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
            print(f"[BTC→ETH {f:3s}] {split} acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f}")
            del R, p; gc.collect()
        P = np.concatenate(Ps, axis=0); n = P.shape[1]
        R = np.zeros_like(P, dtype=np.float64)
        for i in range(P.shape[0]):
            R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
        p = R.mean(axis=0)
        acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_daily(p, y, sec_arr)
        print(f"[BTC→ETH ALL ] {split} acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) bad={nbad} tpd={tpd:.2f}")
        print(f"   逐月: {{k: round(v,3)}}", {k: round(v, 3) for k, v in acc_m.items()})
        del Ps, P, R, p; gc.collect()


if __name__ == "__main__":
    main()
