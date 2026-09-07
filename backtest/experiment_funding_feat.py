#!/usr/bin/env python3
"""验证 funding(资金费率)可靠特征能否在无泄漏R2协议下提升30min方向准确率。

公平对比: 同一骨架特征(BASE_FEATURES)  + 可选 funding 特征
  - 无funding:  X = BASE_FEATURES
  - 有funding:  X = BASE_FEATURES + [funding_capped, sign, 24h滚动z, 极端flag]
同一 LGB流程/早停/随机种子, 仅在特征开关上不同; 用 R2 逐日滚动 Top1% 评估。
用法: python experiment_funding_feat.py [ETH|BTC]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc, argparse, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import lightgbm as lgb
from data_store import AssetContext
import experiment_seq_model as ESM
from optimize_eth_round2 import BASE_FEATURES

MAX_TRAIN = 2_000_000
SEED = 42
LR = 0.03
N_SEED = 2
BAGGED = [42, 49]


def build_funding_extra(ctx):
    """全 ds 长的 funding 特征矩阵 (n_ds, 4), 阈值全部由 train 段拟合(无泄漏)。"""
    t = pq.read_table(f"{config.DS_DIR}/raw_{ctx.symbol}.parquet", columns=["funding"])
    raw_fr = t["funding"].to_numpy().astype(np.float64)
    s = pd.Series(raw_fr)
    rm = s.rolling(1440).mean().to_numpy()
    rs = s.rolling(1440).std().to_numpy()
    fz = np.where(rs > 1e-12, (raw_fr - np.where(rs > 1e-12, rm, 0.0)) / np.where(rs > 1e-12, rs, 1.0), 0.0)
    ds_i = ctx.ds_to_raw
    fr_ds = raw_fr[ds_i]
    fz_ds = fz[ds_i]
    trm = ctx.split_rows["train"]
    p1, p99 = np.percentile(fr_ds[trm], 1), np.percentile(fr_ds[trm], 99)
    q99 = np.percentile(np.abs(fz_ds[trm]), 99)
    capped = np.clip(fr_ds, p1, p99).astype(np.float32)
    F = np.stack([capped, np.sign(fr_ds).astype(np.float32),
                  fz_ds.astype(np.float32), (np.abs(fz_ds) >= q99).astype(np.float32)], axis=1)
    del t, raw_fr, s, rm, rs, fz
    gc.collect()
    return F.astype(np.float32)


def fit(ctx, Fextra, use_funding, seed):
    feats = list(ctx.feat_names)
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    Xes_all = ctx.X_subset(feats, esm)
    yes = ctx.label[esm].astype(np.float64)
    if use_funding:
        Xes_all = np.concatenate([Xes_all, Fextra[esm]], axis=1)

    Xtr_all = ctx.X_subset(feats, trm)
    rng = np.random.default_rng(seed)
    ntr = len(Xtr_all)
    keep = rng.choice(ntr, size=min(MAX_TRAIN, ntr), replace=False) if ntr > MAX_TRAIN else np.arange(ntr)
    Xtr = Xtr_all[keep]
    ytr = ctx.label[trm][keep].astype(np.float64)
    retw = np.clip(np.abs(ctx.retf("train")[keep]).astype(np.float64) * 50.0, 0.5, 5.0)
    if use_funding:
        Xtr = np.concatenate([Xtr, Fextra[np.where(trm)[0]][keep]], axis=1)
    del Xtr_all
    gc.collect()

    p = dict(objective="binary", metric="auc", learning_rate=LR, num_leaves=127,
             max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
             min_data_in_leaf=100, lambda_l1=0.05, lambda_l2=1.0,
             num_threads=config.N_JOBS, verbosity=-1, seed=seed,
             scale_pos_weight=1.0)
    dtr = lgb.Dataset(Xtr, ytr, weight=retw)
    des = lgb.Dataset(Xes_all, yes, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(200, verbose=False),
                             lgb.log_evaluation(0)])
    bi = m.best_iteration
    del Xtr, dtr, ytr, retw
    del Xes_all, des
    gc.collect()
    return m, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run(s)


def run(symbol):
    print(f"\n########## {symbol} funding特征验证 ##########", flush=True)
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    Fextra = build_funding_extra(ctx)
    print(f"  funding特征矩阵 {Fextra.shape}  (训练阈值已固定)", flush=True)

    results = {}
    for tag, use_f in (("无funding", False), ("有funding", True)):
        models = []
        for seed in BAGGED[:N_SEED]:
            m, bi = fit(ctx, Fextra, use_f, seed)
            print(f"  [{tag}/seed{seed}] iter={bi} ({time.time()-t0:.0f}s)", flush=True)
            models.append(m)
        pmv = ens_predict(ctx, models, "meta_val", Fextra, use_f)
        pt = ens_predict(ctx, models, "test", Fextra, use_f)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[- ESM.WIN_DAYS * 1440:]
        r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8),
                        ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
        ysel = ctx.y("test")[r["sel"]]
        acc = ESM.report(tag, r["pred"], ysel, r["mts"])
        results[tag] = acc
        for m in models: del m
        # 用 meta_val 行的软标签做校准(与管线一致), 对 test 概率做 rank 融合即可
        gc.collect()
    print(f"\n  >>> {symbol}: 无funding={results['无funding']:.4f}  有funding={results['有funding']:.4f}  "
          f"增量={results['有funding']-results['无funding']:+.4f}", flush=True)


def ens_predict(ctx, models, split_name, Fextra, use_funding):
    mask = ctx.split_rows[split_name]
    X = ctx.X_subset(list(ctx.feat_names), mask)
    if use_funding:
        X = np.concatenate([X, Fextra[mask]], axis=1)
    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        pr = m.predict(X, num_iteration=m.best_iteration).astype(np.float64)
        R[i] = np.argsort(np.argsort(pr)).astype(np.float64) / (n - 1)
    del X
    gc.collect()
    return R.mean(axis=0)


def predict_set(model, ctx, split_name, Fextra, use_funding):
    mask = ctx.split_rows[split_name]
    X = ctx.X_subset(list(ctx.feat_names), mask)
    if use_funding:
        X = np.concatenate([X, Fextra[mask]], axis=1)
    p = model.predict(X, num_iteration=model.best_iteration).astype(np.float64)
    del X
    return p


if __name__ == "__main__":
    main()