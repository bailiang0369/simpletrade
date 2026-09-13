#!/usr/bin/env python3
"""增量特征实验: ETH h15 + VPIN / 多窗口 OFI. 独立运行, 不改任何现有文件.

流程:
  1. 从 1min raw parquet 计算多窗口 OFI + VPIN
  2. 与 h15 dataset 对齐 (ds_to_raw 索引映射)
  3. 用现有 FEATURES + 新增 ~10 个 OFI/VPIN 特征训练 5-seed LGBM
  4. 对比 baseline (不加新特征) 在 meta_val / test 上的准确率

口径: 因果 P99 (前30天), 15模型集成的 baseline 参考 h10 log.
用法: python backtest_short/exp_ofi_vpin.py
"""
import os, sys, gc, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import config
from data_store import AssetContext

# ---- 实验参数 ----
SYMBOL = "ETH"
HORIZON = 15
BAGGED_SEEDS = [42, 49, 56, 63, 70]
# OFI 多窗口: 1min bar 数
OFI_WINDOWS = [15, 30, 60, 120, 240, 480, 960]
# VPIN 桶大小: 30 分钟平均成交量 (bar-level, 后面按固定数量切)
VPIN_BAR_CHUNK = 100  # 每 100 个 1min bar 累积一个桶

# ---- 基线特征 (和现有 train_pool_short 一致) ----
BASE_FEATURES = [
    "lr_15", "lr_120", "lr_240", "mom_60",
    "rvol_30", "rvol_60", "rvol_z_60", "rvol_dir",
    "z_30", "z_60", "z_120",
    "pos_30", "pos_120", "pos_240",
    "dd_240", "ru_240",
    "hh_dd_60", "ll_ru_60",
    "lr_skew_60", "max_range_30",
    "tb_act_60", "ts_act_60", "cvd_30", "cvd_60",
    "mom_align_30_240",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "ret_day",
    "dn_run_len", "dn_run_max_240", "dn_net_240", "dn_accel_240",
    "low_break_cnt_120", "bounce_fail_240", "regime_dn_bear", "dn_rvol_ratio",
]


def compute_ofi_vpin_features(raw_parquet, target_symbol):
    """从 1min raw parquet 计算多窗口 OFI + VPIN, 返回 dict{feat_name: 1d-array} 按原始 1min 索引."""
    t0 = time.time()
    print(f"[ofi] 读取 {raw_parquet} ...")
    df = pl.read_parquet(raw_parquet).sort("ts")
    bv = df["buy_vol"].to_numpy().astype(np.float64)
    sv = df["sell_vol"].to_numpy().astype(np.float64)
    v = bv + sv
    ofi = bv - sv
    n = len(v)
    print(f"[ofi] {target_symbol} 1min bar={n}, V mean={v.mean():.1f}, OFI mean={ofi.mean():.1f}")

    feats = {}

    # --- 1. 多窗口 OFI (简单移动和 / 滚动标准化) ---
    for w in OFI_WINDOWS:
        if n < w:
            continue
        # 累积和 rolling
        cs = np.concatenate([[0.0], np.cumsum(ofi)])
        roll_sum = cs[w:] - cs[:-w]          # 长度 n-w+1
        feats[f"ofi_sum_{w}"] = np.concatenate([[0.0] * (w - 1), roll_sum]).astype(np.float32)

        # 滚动 OFI / 滚动 V (平衡比率)
        csv = np.concatenate([[0.0], np.cumsum(v)])
        roll_v = csv[w:] - csv[:-w]
        ratio = np.zeros(n, dtype=np.float32)
        ratio[w - 1:] = (roll_sum / np.maximum(roll_v, 1e-9)).astype(np.float32)
        feats[f"ofi_ratio_{w}"] = ratio

    # --- 2. VPIN: 等成交量桶内 |OFI|/V 均值 + 桶间 OFI 方向 ---
    # 简化方案: 等 1min bar 数量桶 (100 bar ≈ 100 分钟), 计算桶内指标, 然后映射回 bar 级
    chunk = VPIN_BAR_CHUNK
    vp_n_buckets = n // chunk
    if vp_n_buckets > 10:
        b_ofi = np.zeros(vp_n_buckets, dtype=np.float64)
        b_v   = np.zeros(vp_n_buckets, dtype=np.float64)
        for k in range(vp_n_buckets):
            s = k * chunk; e = s + chunk
            b_ofi[k] = ofi[s:e].sum()
            b_v[k]   = v[s:e].sum()
        b_ratio = b_ofi / np.maximum(b_v, 1e-9)   # 桶级 OFI/V
        b_abs   = np.abs(b_ratio)                  # 桶级不平衡绝对值 (VPIN 核心)
        # 映射回 1min bar, 末尾 padding 到 n
        def _pad(arr):
            if len(arr) == n: return arr
            return np.concatenate([arr, [arr[-1]] * (n - len(arr))])

        feats["vpin_bar_abs"] = _pad(np.repeat(b_abs, chunk)).astype(np.float32)
        feats["vpin_bar_dir"] = _pad(np.repeat(np.sign(b_ratio), chunk)).astype(np.float32)
        for wb in [5, 10]:
            if vp_n_buckets >= wb:
                cs2 = np.concatenate([[0.0], np.cumsum(b_abs)])
                roll = cs2[wb:] - cs2[:-wb]
                feats[f"vpin_ma{wb}"] = _pad(np.repeat(
                    np.concatenate([[0.0] * (wb - 1), roll / wb]), chunk
                )).astype(np.float32)

    # --- 3. 加权 OFI (最近 30 bar 指数加权) ---
    ew = np.exp(np.linspace(-1, 0, 30))
    ew /= ew.sum()
    ofi_ew = np.convolve(ofi, ew, mode='full')[:n]
    feats["ofi_ewma_30"] = ofi_ew.astype(np.float32)
    feats["ofi_ewma_ratio"] = (ofi_ew / np.maximum(
        np.convolve(v, ew, mode='full')[:n], 1e-9
    )).astype(np.float32)

    print(f"[ofi] 特征数={len(feats)}, 耗时 {time.time()-t0:.1f}s")
    for k, arr in feats.items():
        if arr.max() == arr.min() or np.isnan(arr).any():
            print(f"  [WARN] {k}: 全0 / 含NaN, max={arr.max()}, nan={np.isnan(arr).sum()}")
    return feats


def align_to_dataset(feat_dict, ctx, horizon):
    """feat_dict 是 1min bar 级别的特征 (索引=原始 1min bar 下标).
    dataset 是 horizon 分钟级别, ctx.ds_to_raw[mask] 把 ds 行映射回 raw 行.
    返回: {split: np.array(n_rows, n_feats)}."""
    n_ds = len(ctx.ds_to_raw)
    print(f"[align] ds 总行数={n_ds}, ds->raw 长度={len(ctx.ds_to_raw)}")
    results = {}
    for split in ("train", "meta_val", "test", "early_stop"):
        mask = ctx.split_rows[split]
        if not mask.any():
            continue
        ri = ctx.ds_to_raw[mask].astype(int)   # ds 行 -> raw 1min bar 下标
        X = np.column_stack([feat_dict[n][ri] for n in feat_dict])
        results[split] = X.astype(np.float32)
        print(f"  [{split}] shape={X.shape}, "
              f"nan={np.isnan(X).sum()}, "
              f"feat0_mean={X[:,0].mean():.4f}")
    return results


def train_lgbm(X_train, y_train, X_es, y_es, seed, max_iter=2000):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    n = len(X_train); k = min(n, 500_000)
    idx = rng.choice(n, k, replace=False)
    w = np.clip(np.abs(y_train[idx]) * 50, 0.5, 5.0)

    dtr = lgb.Dataset(X_train[idx], label=y_train[idx], weight=w)
    des = lgb.Dataset(X_es, label=y_es, reference=dtr)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
        "min_data_in_leaf": 200, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "verbose": -1, "n_jobs": 1, "seed": seed,
    }
    m = lgb.train(params, dtr, num_boost_round=max_iter,
                  valid_sets=[des], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return m


def rank_ens(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def causal_eval(sec, day_of, conf, pred, days, q, win_days, y):
    """无前视因果 Pq 阈值."""
    day_confs = {int(d): conf[day_of == d] for d in days}
    day_list = days.astype(int).tolist()
    sel = np.zeros(len(sec), dtype=bool)
    for i, d in enumerate(day_list):
        prior = day_list[max(0, i - win_days):i]
        if len(prior) < win_days:
            continue
        hist = np.concatenate([day_confs[d2] for d2 in prior])
        tau = float(np.percentile(hist, q))
        m = day_of == d
        sel[m] = conf[m] >= tau
    n = int(sel.sum())
    acc = float((pred[sel] == y[sel]).mean()) if n else 0
    return n, acc


def run(baseline_only=False):
    t_all = time.time()
    print(f"===== EXP {SYMBOL} h{HORIZON} VPIN/OFI =====")

    ctx = AssetContext(SYMBOL, horizon=HORIZON, ds_name=f"ds_{SYMBOL}_h{HORIZON}")

    # 加载 baseline 特征 (dataset 自带)
    X_base = {}
    for split in ("train", "meta_val", "test", "early_stop"):
        mask = ctx.split_rows[split]
        X_base[split] = ctx.X_subset(BASE_FEATURES, mask)
        print(f"[base] {split}: {X_base[split].shape}")

    # 计算 OFI/VPIN 特征
    raw_path = os.path.join(config.DS_DIR, f"raw_{SYMBOL}.parquet")
    assert os.path.exists(raw_path), f"missing {raw_path}, run fetch first"
    feat_dict = compute_ofi_vpin_features(raw_path, SYMBOL)
    X_ofi = align_to_dataset(feat_dict, ctx, HORIZON)
    NEW_FEAT_NAMES = list(feat_dict.keys())

    # 合并
    X_full = {}
    for split in X_base:
        X_full[split] = np.column_stack([X_base[split], X_ofi[split]])
    print(f"[full] feature dim={X_full['train'].shape[1]} "
          f"(base={X_base['train'].shape[1]} + ofi={len(NEW_FEAT_NAMES)})")

    y = ctx.label.astype(np.float32)
    y_es = y[ctx.split_rows["early_stop"]]
    y_mv = y[ctx.split_rows["meta_val"]]
    y_te = y[ctx.split_rows["test"]]
    sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    results = []

    for label, use_ofi in [("BASE  仅原有特征", False),
                           ("FULL  BASE+OFI+VPIN", True)]:
        print(f"\n--- {label} ---")
        X_tr_ = X_full if use_ofi else X_base
        X_es_ = X_full if use_ofi else X_base
        preds = []
        t_tr = time.time()
        for seed in BAGGED_SEEDS:
            m = train_lgbm(X_tr_["train"], ctx.label[ctx.split_rows["train"]].astype(np.float64),
                           X_es_["early_stop"], y_es.astype(np.float64), seed)
            best_iter = m.best_iteration
            Pmv = m.predict(X_es_["meta_val"], num_iteration=best_iter)
            Pte = m.predict(X_es_["test"], num_iteration=best_iter)
            preds.append((Pmv, Pte))
            print(f"  seed={seed} iter={best_iter}")
        print(f"  train total {time.time()-t_tr:.1f}s")

        # 集成
        Pmv_all = np.stack([p[0] for p in preds], axis=0)
        Pte_all = np.stack([p[1] for p in preds], axis=0)
        pmv = rank_ens(1.0 / (1.0 + np.exp(-Pmv_all)))
        pte = rank_ens(1.0 / (1.0 + np.exp(-Pte_all)))

        for split, p_, y_, sec_ in [("meta_val", pmv, y_mv, None),
                                     ("test", pte, y_te, sec_te)]:
            conf = np.abs(p_ - 0.5) * 2
            pred = (p_ >= 0.5).astype(np.int8)
            # 因果 P99 / 前30天
            if sec_ is None:
                # meta_val 从 ctx.times 拿
                sec_ = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            day_of = sec_ // 86400
            days = np.unique(day_of)
            for q in [99.0, 99.2, 99.5]:
                n, acc = causal_eval(sec_, day_of, conf, pred, days, q, 30, y_)
                top = conf.argsort()[::-1][:max(1, int(0.01 * len(conf)))]
                top_acc = float((pred[top] == y_[top]).mean())
                print(f"  [{split}] win=30 q={q}: n={n} ({n/len(days):.1f}/天) "
                      f"acc={acc*100:.2f}% | top1%有前视={top_acc*100:.2f}%")
                results.append({
                    "label": label, "split": split, "q": q,
                    "n": n, "acc": acc, "top_acc": top_acc,
                })

    # 特征重要性 (FULL)
    import lightgbm as lgb
    imps = []
    for seed in BAGGED_SEEDS:
        m = train_lgbm(X_full["train"], ctx.label[ctx.split_rows["train"]].astype(np.float64),
                       X_full["early_stop"], y_es.astype(np.float64), seed)
        imps.append(m.feature_importance(importance_type="gain"))
    mean_imp = np.mean(imps, axis=0)
    all_feat_names = BASE_FEATURES + NEW_FEAT_NAMES
    order = np.argsort(-mean_imp)
    print("\n[importance top 40]")
    for i in order[:40]:
        tag = " **NEW**" if all_feat_names[i] in NEW_FEAT_NAMES else ""
        print(f"  {all_feat_names[i]:22s} gain={mean_imp[i]:10.1f}{tag}")

    print(f"\n===== DONE {time.time()-t_all:.0f}s =====")
    return results


if __name__ == "__main__":
    run()
