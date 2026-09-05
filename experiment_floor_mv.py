"""ETH 月度下限诊断：在 meta_val 上训练 LGBM 集成池并产出逐月 top-1% 结构。

原则(与交接一致):
  - 只用 ETH 数据; train 训练 / early_stop 早停 / meta_val 只用于诊断与选择, test 不碰。
  - top-1% 口径: 全集 argsort(-conf), conf=max(p,1-p); 用全局归一化 rank。
脚本输出(落盘, 供后续 floor-aware 优化复用):
  - {DS_DIR}/ETH_floor_mv_p.npy   : mv 全期每行预测概率 p (长度=mv行数)
  - 打印 mv 逐月: 样本数 / 准确率 / 平均置信 / 集成分歧, 用于判定 46% 性质。
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


def train_one(feats, ctx, seed):
    import lightgbm as lgb
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    tr_idx_all = np.where(trm)[0]
    Xes = ctx.X_subset(feats, esm); yes = ctx.label[esm].astype(np.float64)
    train_retf = ctx.retf("train")
    rng = np.random.default_rng(seed)
    tr_idx = tr_idx_all.copy()
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = rng.choice(len(tr_idx), MAX_TRAIN, replace=False)
    trm2 = np.zeros_like(trm, dtype=bool); trm2[tr_idx] = True
    Xtr = ctx.X_subset(feats, trm2); ytr = ctx.label[trm2].astype(np.float64)
    keep_local = np.where(trm2[tr_idx_all])[0]
    raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
    w = np.clip(raw_w * 50, 0.5, 5.0)
    p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
             max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
             min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,
             num_threads=config.N_JOBS, verbosity=-1, seed=seed, min_data=1)
    dtr = lgb.Dataset(Xtr, ytr, weight=w)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["early_stop"],
                  callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5), lgb.log_evaluation(0)])
    bi = m.best_iteration
    del Xtr, dtr, ytr, w, Xes; gc.collect()
    return m, bi


def main():
    t0 = time.time()
    ctx = AssetContext("ETH", horizon=30)

    # 训练 bagged 池
    models = []
    for sd in SEEDS:
        mm, bi = train_one(BASE_FEATURES, ctx, sd)
        models.append((mm, bi))
        print(f"  seed{sd}: iter={bi}  ({time.time()-t0:.0f}s)", flush=True)

    # mv 预测: 每模型内 rank 归一化到 [0,1], 再等权 -> p
    mvm = ctx.split_rows["meta_val"]
    nmv = int(mvm.sum())
    R = np.zeros((len(SEEDS), nmv), dtype=np.float64)
    # mv 也必须预处理 ret_day 以外的特征与训练一致; X_subset 直接取 ds 列, 一致即可
    Xmv = ctx.X_subset(BASE_FEATURES, mvm)
    for i, (mm, bi) in enumerate(models):
        pv = mm.predict(Xmv, num_iteration=bi)
        R[i] = np.argsort(np.argsort(pv)).astype(np.float64) / (nmv - 1)
        del pv; gc.collect()
    p_raw = R.mean(axis=0)
    std_raw = R.std(axis=0)
    del Xmv; gc.collect()
    # 保存 mv 预测用于 floor-aware 优化
    np.save(f"{config.DS_DIR}/ETH_floor_mv_p.npy", p_raw.astype(np.float32))
    np.save(f"{config.DS_DIR}/ETH_floor_mv_std.npy", std_raw.astype(np.float32))

    y = ctx.y("meta_val"); retf = ctx.retf("meta_val"); times = ctx.times("meta_val")
    r = evaluate_topk(p_raw, y, retf, times)
    print(f"\n=== mv 整体 top-1% ===")
    print(f"acc={r['accuracy']:.4f} k={r['k']} tpd={r['trades_per_day']:.1f} "
          f"conf_mean={r['conf_mean']:.4f} up={r['avg_ret_up_bps']:.1f}bp dn={r['avg_ret_dn_bps']:.1f}bp")

    # 逐月结构: 在全局 top-1% 命中的样本上按月份统计
    conf = np.maximum(p_raw, 1 - p_raw)
    sel = np.argsort(-conf)[:r["k"]]
    import datetime
    sec = np.asarray(times).astype("datetime64[s]").astype(np.int64)
    m_sel = sec[sel].astype("datetime64[s]").astype("datetime64[M]")
    s_pred = (p_raw >= 0.5).astype(np.int8)
    print(f"\n=== mv 逐月结构 (全局 top-1% 命中的样本, n/mon acc/mon conf/disp) ===")
    uniq = np.unique(m_sel)
    min_acc = 1.0; min_n = 1e9; real_bad_months = 0
    for mu in uniq:
        mm = m_sel == mu
        n_m = int(mm.sum())
        a_m = float((s_pred[sel][mm] == y[sel][mm]).mean())
        c_m = float(conf[sel][mm].mean())
        d_m = float(std_raw[sel][mm].mean())
        mstr = str(mu)[:7]
        flag = "  <== 最差" if a_m < min_acc else ""
        if a_m < min_acc:
            min_acc = a_m; min_n = n_m
        if n_m >= 30 and a_m < 0.55:
            real_bad_months += 1
        print(f"  {mstr}: n={n_m:4d} acc={a_m:.4f} conf={c_m:.4f} disp={d_m:.4f}{flag}")
    print(f"\n最差月: acc={min_acc:.4f} n={min_n}   | 高样本(n>=30)且 acc<0.55 的月数={real_bad_months}")
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()