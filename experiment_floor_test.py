"""test 一次性验证: 与参考脚本同口径的 LGBM 池, 对比 全局 vs 每日局部 top1% 的逐月下限。
训练(参考 optimize_eth.train_seed): train 训练 + early_stop 早停, feval=top1_acc 作为早停目标。
产出并保存 mv / test 预测, 在两者上评估覆盖率策略。
"""
import os, sys, gc, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

BASE_FEATURES = [
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
]
SEEDS = [42, 49, 56, 63, 70]
MAX_TRAIN = 2_600_000


def topk_acc_eval(preds, dtrain):
    labels = dtrain.get_label()
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs); k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return "top1_acc", acc, True


def train_pool(feats, ctx):
    import lightgbm as lgb
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = ctx.X_subset(feats, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    models = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        tr_idx = tr_idx_all.copy()
        if len(tr_idx) > MAX_TRAIN:
            tr_idx = rng.choice(len(tr_idx), MAX_TRAIN, replace=False)
        trm2 = np.zeros_like(trm, dtype=bool); trm2[tr_idx] = True
        Xtr = ctx.X_subset(feats, trm2); ytr = ctx.label[trm2].astype(np.float64)
        keep = np.where(trm2[tr_idx_all])[0]
        raw_w = np.abs(train_retf[keep]).astype(np.float64)
        raw_w[ytr > 0.5] *= 2.0
        w = np.clip(raw_w * 50, 0.5, 5.0)
        p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                 max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                 min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
                 num_threads=config.N_JOBS, verbosity=-1, seed=sd, min_data=1)
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                      feval=topk_acc_eval,
                      callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5), lgb.log_evaluation(0)])
        models.append((m, m.best_iteration))
        print(f"  seed{sd}: iter={m.best_iteration}", flush=True)
        del Xtr, dtr, ytr, w; gc.collect()
    del Xes, yes; gc.collect()
    return models


def rank_mean(models, feats, ctx, split):
    m = ctx.split_rows[split]; n = int(m.sum())
    X = ctx.X_subset(feats, m)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, (mm, bi) in enumerate(models):
        pv = mm.predict(X, num_iteration=bi)
        R[i] = np.argsort(np.argsort(pv)).astype(np.float64) / (n - 1)
        del pv; gc.collect()
    p = R.mean(axis=0)
    del X, R; gc.collect()
    return p


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


def main():
    t0 = time.time()
    ctx = AssetContext("ETH", horizon=30)
    models = train_pool(BASE_FEATURES, ctx)
    print(f"训练完成 {time.time()-t0:.0f}s", flush=True)

    for split in ("meta_val", "test"):
        p = rank_mean(models, BASE_FEATURES, ctx, split)
        np.save(f"{config.DS_DIR}/ETH_floor_{split}_p.npy", p.astype(np.float32))
        y = ctx.y(split)
        sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        print(f"\n===== {split} rows={len(p)} n_days={np.unique(sec_arr//86400).size} =====")
        for mode in ("global", "daily"):
            acc, min_a, min_k, nbad, tpd, cov, acc_m = eval_policy(p, y, sec_arr, mode)
            print(f"[{mode}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) "
                  f"bad(<55)={nbad} tpd={tpd:.2f} cov={cov:.4f}")
            print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
    print(f"\n完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()