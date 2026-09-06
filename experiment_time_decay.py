#!/usr/bin/env python3
"""时间衰减权重 A/B: 训练样本按距 train 段末尾的时长指数衰减权重, 让模型更贴近期 regime。

动机: train 段 2020-01~2024-06 横跨牛/熊/震荡多种 regime, 早期(2020-2021)市场结构与当前
差异大, 可能成为噪声。若按样本新旧加权(近期权重大), 模型更贴当前市场状态, 或可提升准确率。

公平对比: 同一 LGB 流程/早停/随机种子, 仅权重函数不同 (A/B 唯一变量):
  W0 基线:  收益绝对值加权 clip(|ret|*50, 0.5, 5)          (与跨资产实验完全一致)
  W1:       W0 × 指数衰减(半衰期 12 月)
  W2:       W0 × 指数衰减(半衰期 24 月)
R2 逐日滚动 Top1% 评估 (mv 段置信度种子 -> test 一次性), 与跨资产/funding 实验同协议。
用法: python experiment_time_decay.py [ETH|BTC]   (默认 ETH,BTC)
"""
import os, sys, gc, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import lightgbm as lgb
from data_store import AssetContext
import experiment_seq_model as ESM

MAX_TRAIN = 2_000_000
LR = 0.03
BAGGED = [42, 49]
HALF_LIVES = {  # tag -> 半衰期天数
    "W0_retw": None,      # 基线: 仅收益加权
    "W1_hl12m": 365,
    "W2_hl24m": 730,
}


def make_weights(ctx, keep, half_life_days=None):
    """训练权重 = 收益绝对值加权 × (可选)时间指数衰减。keep 为 train 段抽样后的相对索引。"""
    retw = np.clip(np.abs(ctx.retf("train")[keep]).astype(np.float64) * 50.0, 0.5, 5.0)
    if half_life_days is None:
        return retw
    tr_idx = np.where(ctx.split_rows["train"])[0]
    ts = ctx.ds_ts[tr_idx[keep]].astype(np.float64)          # 样本时刻(秒)
    tr_end = ctx.ds_ts[tr_idx].max()                          # train 段末尾
    days = (tr_end - ts) / 86400.0
    decay = 0.5 ** (days / half_life_days)
    return retw * decay


def fit(ctx, half_life_days, seed):
    from validate_eth_quick import FEATURES
    feats = list(FEATURES)
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    Xes_all = ctx.X_subset(feats, esm)
    yes = ctx.label[esm].astype(np.float64)

    Xtr_all = ctx.X_subset(feats, trm)
    rng = np.random.default_rng(seed)
    ntr = len(Xtr_all)
    keep = rng.choice(ntr, size=min(MAX_TRAIN, ntr), replace=False) if ntr > MAX_TRAIN else np.arange(ntr)
    Xtr = Xtr_all[keep]
    ytr = ctx.label[trm][keep].astype(np.float64)
    w = make_weights(ctx, keep, half_life_days)
    del Xtr_all
    gc.collect()

    p = dict(objective="binary", metric="auc", learning_rate=LR, num_leaves=127,
             max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
             min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0,
             num_threads=config.N_JOBS, verbosity=-1, seed=seed,
             scale_pos_weight=1.0)
    dtr = lgb.Dataset(Xtr, ytr, weight=w)
    des = lgb.Dataset(Xes_all, yes, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(200, verbose=False),
                             lgb.log_evaluation(0)])
    bi = m.best_iteration
    del Xtr, dtr, ytr, w, Xes_all, des
    gc.collect()
    return m, bi


def ens_predict(ctx, models, split_name):
    from validate_eth_quick import FEATURES
    mask = ctx.split_rows[split_name]
    X = ctx.X_subset(list(FEATURES), mask)
    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        pr = m.predict(X, num_iteration=m.best_iteration).astype(np.float64)
        R[i] = np.argsort(np.argsort(pr)).astype(np.float64) / (n - 1)
    del X
    gc.collect()
    return R.mean(axis=0)


def run(symbol):
    print(f"\n########## {symbol} 时间衰减权重 A/B ##########", flush=True)
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    results = {}
    for tag, hl in HALF_LIVES.items():
        models = []
        for seed in BAGGED:
            m, bi = fit(ctx, hl, seed)
            print(f"  [{tag}/seed{seed}] iter={bi} ({time.time()-t0:.0f}s)", flush=True)
            models.append(m)
        pmv = ens_predict(ctx, models, "meta_val")
        pt = ens_predict(ctx, models, "test")
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8),
                        ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
        ysel = ctx.y("test")[r["sel"]]
        acc, rows = ESM.monthly(r["pred"], ysel, r["mts"])
        worst = min(x[2] for x in rows)
        nbad = sum(1 for x in rows if x[2] < 0.55)
        results[tag] = (acc, worst, nbad, len(ysel))
        print(f"  [{tag}] 总acc={acc:.4f} 最差月={worst:.4f} 坏月={nbad} 信号={len(ysel)} ({time.time()-t0:.0f}s)", flush=True)
        for m in models: del m
        gc.collect()
    base = results["W0_retw"][0]
    print(f"\n  >>> {symbol} 时间衰减对比 (W0 基线 acc={base:.4f}):", flush=True)
    for tag in ("W1_hl12m", "W2_hl24m"):
        acc, worst, nbad, nsig = results[tag]
        print(f"      {tag:<10} acc={acc:.4f} ({acc-base:+.4f})  最差月={worst:.4f}  坏月={nbad}  信号={nsig}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run(s)


if __name__ == "__main__":
    main()
