#!/usr/bin/env python3
"""Phase1: 训练 JOINT pool20 权重到 live/bundles/h{horizon}/。

JOINT 池是跨币联合训练: 两币(BTC/ETH)合并 train 段, per family 5 个 bagged seed,
共 15 booster(5 lgb + 5 xgb + 5 cat)。两币*每周期共用同一套权重(按 family+seed),
因此每 (horizon,family,seed) 产出一个模型文件。

与实盘 bundles 的路径约定(见 freeze_bundle): live/bundles/h{horizon}/JOINT_{sym}_{fam}_seed{seed}.{ext}
注: 实盘两币共用一池, 故文件以 sym 命名仅为可读, 实际任何 sym 都加载同一池 (Predictor 只用 family+seed)。

用法(建议后台续跑, 每个 fam 独立可并行):
  python live/train_pool.py --horizon 30 --fam lgb
  python live/train_pool.py --horizon 30 --fam xgb
  python live/train_pool.py --horizon 30 --fam cat
  # 同 60, 但 h60 需先建 horizon=60 的 ds 数据集(build_dataset 的 action 参数)。

关键参数(live/predict.py 与离线 experiment_pool20_joint 对齐):
  MAX_TRAIN 合并上限, lgb 参数/早停作用于两币合并的 early_stop(es) 集。
"""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from data_store import AssetContext
from validate_eth_quick import EXTRA_FEATURE_NAMES, BAGGED_SEEDS, get_X, compute_extra_raw

BUNDLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundles")
MAX_TRAIN = 2_600_000
SYMBOLS = ["ETH", "BTC"]
EXT = {"lgb": "txt", "xgb": "json", "cat": "cbm"}
SKIP_DONE = False


def joint_masks_weights(ctxs, seed):
    """从两资产 train 段按 seed 采样, 返回每资产的 (mask, weight)。"""
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


def load_bars_df(s):
    """在线特征引擎的输入 bars df(同 compute_X 约定), 从 raw parquet 取全量(tail 裁剪)。"""
    from data_store import load_raw_bars
    return load_raw_bars(s)


def build_pool_X(horizon):
    """返回 (ctxs, extras) 供训练: cnt=x.ds_ts 对应 min. X 用于 es/tr 需按 max rows。"""
    ctxs = {s: AssetContext(s, horizon=horizon) for s in SYMBOLS}
    extras = {s: compute_extra_from_raw(ctxs[s]) for s in SYMBOLS}
    return ctxs, extras


def compute_extra_from_raw(ctx):
    """由 raw channels 复刻 compute_extra_raw(在线特征引擎 _extra_features 的裸版)。

    离线 compute_extra_raw(ctx) 已实现(validate_eth_quick), 直接复用即可。
    """
    return compute_extra_raw(ctx)


def family_Xes(ctxs, extras, horizon):
    """两币合并的 early_stop 拼接。"""
    Xs, ys = [], []
    for s in SYMBOLS:
        m = ctxs[s].split_rows["early_stop"]
        Xs.append(pool_X(ctxs[s], extras[s], m))
        ys.append(ctxs[s].label[m].astype(np.float64))
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    del Xs, ys; gc.collect()
    return X, y


def pool_X(ctx, extra, mask):
    """离线 get_X: 直接从预计算 ds 列(38 base + 17 cross) + extra(3) 取 mask 行特征 [n,58]。

    与在线 compute_X 的关系由 replay_online 的等价性闸门(≤1e-5)保证, 故训练侧直接用
    官方 get_X 速度快且与离线 experiment_pool20_joint 语义完全一致。
    """
    return get_X(ctx, extra, mask)


def train_family(horizon, family, only_seeds=None):
    t0 = __import__("time").time()
    ctxs, extras = build_pool_X(horizon)
    Xes, yes = family_Xes(ctxs, extras, horizon)
    out_dir = os.path.join(BUNDLES_DIR, f"h{horizon}")
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    if os.path.exists(os.path.join(out_dir, "best_iteration.json")):
        try:
            import json as _json
            info = _json.load(open(os.path.join(out_dir, "best_iteration.json")))
        except Exception:
            info = {}
    for seed in BAGGED_SEEDS:
        if only_seeds and seed not in only_seeds:
            continue
        model_name = f"JOINT_{family}_seed{seed}.{EXT[family]}"
        if SKIP_DONE and os.path.exists(os.path.join(out_dir, model_name)):
            print(f"[train] h{horizon} {family} seed{seed} 已存在, 跳过", flush=True)
            continue
        mws = joint_masks_weights(ctxs, seed)
        Xtr_list = [pool_X(ctxs[s], extras[s], mws[s][0]) for s in SYMBOLS]
        ytr_list = [ctxs[s].label[mws[s][0]].astype(np.float64) for s in SYMBOLS]
        w_list = [mws[s][1] for s in SYMBOLS]
        Xtr = np.concatenate(Xtr_list, axis=0); ytr = np.concatenate(ytr_list, axis=0)
        w = np.concatenate(w_list, axis=0)
        del Xtr_list, ytr_list, w_list; gc.collect()
        if family == "lgb":
            import lightgbm as lgb
            p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
                     min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0,
                     scale_pos_weight=1.0, num_threads=config.N_JOBS, verbosity=-1, seed=seed)
            dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                          callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                     lgb.log_evaluation(0)])
            name = f"JOINT_lgb_seed{seed}.txt"
            m.save_model(os.path.join(out_dir, name), num_iteration=m.best_iteration)
            info[name] = int(m.best_iteration)
        elif family == "xgb":
            import xgboost as xgb
            dt = xgb.DMatrix(Xtr, ytr, weight=w); ev = xgb.DMatrix(Xes, yes)
            p = dict(objective="binary:logistic", eval_metric="auc", eta=0.02, max_depth=8,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                     reg_alpha=0.05, reg_lambda=1.0, nthread=config.N_JOBS, seed=seed)
            m = xgb.train(p, dt, num_boost_round=5000, evals=[(ev, "es")],
                          early_stopping_rounds=200, verbose_eval=0)
            name = f"JOINT_xgb_seed{seed}.json"
            m.save_model(os.path.join(out_dir, name))
            info[name] = int(m.best_iteration)
        elif family == "cat":
            from catboost import CatBoostClassifier, Pool
            tr = Pool(Xtr, ytr, weight=w)
            ev = Pool(Xes, yes)
            m = CatBoostClassifier(iterations=5000, learning_rate=0.02, depth=9,
                                   loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=1.0,
                                   od_wait=200, random_seed=seed, thread_count=config.N_JOBS,
                                   verbose=0)
            m.fit(tr, eval_set=ev, use_best_model=True)
            name = f"JOINT_cat_seed{seed}.cbm"
            m.save_model(os.path.join(out_dir, name))
            info[name] = int(m.get_best_iteration())
        print(f"[train] h{horizon} {family} seed{seed} done, best={info[name]}", flush=True)
        # 释放本 seed 大对象(直接 del 真实局部, 勿用 del locals()); 防下一 seed 叠加 OOM
        for _n in ("Xtr", "ytr", "w"):
            try:
                exec(f"del {_n}")
            except Exception:
                pass
        gc.collect()
    with open(os.path.join(out_dir, "best_iteration.json"), "w") as _f:
        json.dump(info, _f, indent=2)
    print(f"[train] h{horizon} {family} 完成, {__import__('time').time()-t0:.0f}s", flush=True)


def main():
    global SKIP_DONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[30, 60])
    ap.add_argument("--fam", required=True, choices=["lgb", "xgb", "cat"])
    ap.add_argument("--seed", type=int, default=None, help="只训练该 seed(续跑/分进程用)")
    ap.add_argument("--skip-done", action="store_true",
                    help="若 best_iteration.json 已含某 seed 则跳过(支持断点续跑)")
    a = ap.parse_args()
    SKIP_DONE = a.skip_done
    # 定义 SKIP_DONE 供 train_family 作用域引用
    if a.horizon == 60:
        # h60 需 ds 卷含 horizon=60 的 label(当前 ds 为 h30); 未建则这里报错引导
        import data_store
        _ = data_store.AssetContext  # noqa: F401
        # 具体: 需用 build_dataset 的 horizon 参数重建 ds; 本脚本假定已就绪
    train_family(a.horizon, a.fam, only_seeds=([a.seed] if a.seed else None))


if __name__ == "__main__":
    main()