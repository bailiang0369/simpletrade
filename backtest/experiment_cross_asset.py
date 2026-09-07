#!/usr/bin/env python3
"""验证跨资产特征(BTC<->ETH)能否在无泄漏R2协议下提升30min方向准确率。

理论依据: 加密市场 ETH 与 BTC 高度相关, BTC 是 beta 龙头。把源币的
因果派生特征(多尺度动量/z-score/波动率/主动买卖失衡)对齐到目标币每个
样本时刻(只取 <= t 的最近行), 作为目标币的额外输入特征。

公平对比: 同一骨架特征(ctx.feat_names) + 可选跨资产特征, 同一 LGB流程/
早停/随机种子, 仅特征开关不同; 用 R2 逐日滚动 Top1% 评估(与 funding 实验同协议)。
用法: python experiment_cross_asset.py [ETH|BTC]   (默认 ETH,BTC)
  - ETH: 特征来自 BTC (龙头->追随)
  - BTC: 特征来自 ETH
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

MAX_TRAIN = 2_000_000
LR = 0.03
BAGGED = [42, 49]
OTHER = {"ETH": "BTC", "BTC": "ETH"}   # 目标币 -> 源币


def _shift(a, k):
    out = np.empty_like(a)
    out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def build_cross_asset(ctx, other):
    """为 ctx(目标币) 构建 other(源币) 的因果特征矩阵 (n_ds, F)。
    全部只用 t 及以前的源币信息: 按 ds_ts 取源币最近 <= t 的行。"""
    t = pq.read_table(f"{config.DS_DIR}/raw_{other}.parquet",
                      columns=["ts", "open", "high", "low", "close", "buy_vol", "sell_vol"])
    ots = t["ts"].to_numpy().astype(np.int64)
    oc = t["close"].to_numpy().astype(np.float64)
    ob = t["buy_vol"].to_numpy().astype(np.float64)
    os_ = t["sell_vol"].to_numpy().astype(np.float64)
    del t
    gc.collect()

    lc = np.log(np.maximum(oc, 1e-12))
    cols = {}
    for k in (5, 15, 30, 60, 120, 240):
        cols[f"lr_{k}"] = lc - _shift(lc, k)
    s = pd.Series(lc)
    for w in (30, 60, 120):
        mu = s.rolling(w).mean().to_numpy()
        sd = s.rolling(w).std().to_numpy()
        cols[f"z_{w}"] = np.where(sd > 1e-9, (lc - mu) / sd, 0.0)
    lr1 = lc - _shift(lc, 1)
    cols["rvol_60"] = pd.Series(lr1).rolling(60).std().to_numpy() * 100
    d = pd.Series(ob - os_)
    t_ = pd.Series(ob + os_)
    cols["cvd_30"] = (d.rolling(30).sum() / (t_.rolling(30).sum() + 1e-12)).to_numpy()
    cols["cvd_60"] = (d.rolling(60).sum() / (t_.rolling(60).sum() + 1e-12)).to_numpy()
    del s, lr1, d, t_, ob, os_
    gc.collect()

    names = list(cols.keys())
    F_other = np.stack(list(cols.values()), axis=1).astype(np.float32)   # (len(ots), F)
    del cols
    gc.collect()

    idx = np.searchsorted(ots, ctx.ds_ts, side="right") - 1              # 最近 <= t
    bad = idx < 0
    idx = np.clip(idx, 0, len(ots) - 1)
    F = F_other[idx]
    if bad.any():
        F[bad] = 0.0
        print(f"  [warn] {int(bad.sum())} ds 行对齐不到 {other} raw, 置零", flush=True)
    del F_other, ots, oc, idx, bad
    gc.collect()
    print(f"  跨资产特征 {len(names)} 列: {names}", flush=True)
    return F.astype(np.float32), names


def fit(ctx, Fextra, use_cross, seed):
    from validate_eth_quick import FEATURES
    feats = list(FEATURES)                       # 固定基础特征(不含跨资产列, 保证A/B公平)
    trm = ctx.split_rows["train"]; esm = ctx.split_rows["early_stop"]
    Xes_all = ctx.X_subset(feats, esm)
    yes = ctx.label[esm].astype(np.float64)
    if use_cross:
        Xes_all = np.concatenate([Xes_all, Fextra[esm]], axis=1)

    Xtr_all = ctx.X_subset(feats, trm)
    rng = np.random.default_rng(seed)
    ntr = len(Xtr_all)
    keep = rng.choice(ntr, size=min(MAX_TRAIN, ntr), replace=False) if ntr > MAX_TRAIN else np.arange(ntr)
    Xtr = Xtr_all[keep]
    ytr = ctx.label[trm][keep].astype(np.float64)
    retw = np.clip(np.abs(ctx.retf("train")[keep]).astype(np.float64) * 50.0, 0.5, 5.0)
    if use_cross:
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
    del Xtr, dtr, ytr, retw, Xes_all, des
    gc.collect()
    return m, bi


def ens_predict(ctx, models, split_name, Fextra, use_cross):
    from validate_eth_quick import FEATURES
    mask = ctx.split_rows[split_name]
    X = ctx.X_subset(list(FEATURES), mask)
    if use_cross:
        X = np.concatenate([X, Fextra[mask]], axis=1)
    n = len(X)
    R = np.zeros((len(models), n), dtype=np.float64)
    for i, m in enumerate(models):
        pr = m.predict(X, num_iteration=m.best_iteration).astype(np.float64)
        R[i] = np.argsort(np.argsort(pr)).astype(np.float64) / (n - 1)
    del X
    gc.collect()
    return R.mean(axis=0)


def run(symbol):
    other = OTHER[symbol]
    print(f"\n########## {symbol}(+{other}特征) 跨资产验证 ##########", flush=True)
    t0 = time.time()
    ctx = AssetContext(symbol, horizon=30)
    Fextra, _ = build_cross_asset(ctx, other)

    results = {}
    for tag, use_c in (("无跨资产", False), ("有跨资产", True)):
        models = []
        for seed in BAGGED:
            m, bi = fit(ctx, Fextra, use_c, seed)
            print(f"  [{tag}/seed{seed}] iter={bi} ({time.time()-t0:.0f}s)", flush=True)
            models.append(m)
        pmv = ens_predict(ctx, models, "meta_val", Fextra, use_c)
        pt = ens_predict(ctx, models, "test", Fextra, use_c)
        seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
        r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8),
                        ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
        ysel = ctx.y("test")[r["sel"]]
        acc = ESM.report(tag, r["pred"], ysel, r["mts"])
        results[tag] = acc
        for m in models: del m
        gc.collect()
    print(f"\n  >>> {symbol}: 无跨资产={results['无跨资产']:.4f}  有跨资产={results['有跨资产']:.4f}  "
          f"增量={results['有跨资产']-results['无跨资产']:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run(s)


if __name__ == "__main__":
    main()
