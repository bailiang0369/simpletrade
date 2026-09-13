#!/usr/bin/env python3
"""完整 A/B 对比: ETH h15 加 OFI/VPIN 特征 vs 原来 59 特征 baseline.

完全复现 train_pool_short + validate_eth_quick 的 pipeline:
  - 特征: 38 FEATURES + 3 EXTRA + 18 CROSS = 59 baseline
  - 新增: +20 OFI/VPIN → 79 total
  - 训练: ETH+BTC 联合, MAX_TRAIN=2_600_000, LGBM lr=0.02, 5000 rounds, early_stop=200
  - 评估: causal P99/P99.5/P99.9 + daily top1%

用法: /root/.pyenv/versions/3.12.13/bin/python -u backtest_short/exp_ofi_vpin_final.py
"""
import os, sys, gc, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import polars as pl
import config
from data_store import AssetContext
from validate_eth_quick import (FEATURES, EXTRA_FEATURE_NAMES, CROSS_FEATURES,
                                 BAGGED_SEEDS, compute_extra_raw, get_X)

MAX_TRAIN = 2_600_000
SYMBOLS = ["ETH", "BTC"]
HORIZON = 15
LOG_PATH = "/workspace/results/exp_ofi_vpin_final.log"

# OFI 窗口 (1min bar 数)
OFI_WINDOWS = [15, 30, 60, 120, 240, 480, 960]
VPIN_BAR_CHUNK = 100


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ============================================================
# 1. OFI/VPIN 特征计算 (1min raw bar 级别)
# ============================================================
def compute_ofi_vpin_raw(target_symbol):
    """返回 dict{feat_name: np.array(raw_n)} 按原始 1min bar 索引."""
    raw_path = os.path.join(config.DS_DIR, f"raw_{target_symbol}.parquet")
    t0 = time.time()
    log(f"[ofi] 读取 {raw_path} ...")
    df = pl.read_parquet(raw_path).sort("ts")
    bv = df["buy_vol"].to_numpy().astype(np.float64)
    sv = df["sell_vol"].to_numpy().astype(np.float64)
    v = bv + sv
    ofi = bv - sv
    n = len(v)
    log(f"[ofi] {target_symbol} 1min bar={n}, V mean={v.mean():.1f}, OFI mean={ofi.mean():.1f}")

    feats = {}

    # OFI: sum + ratio, 多窗口
    cs   = np.concatenate([[0.0], np.cumsum(ofi)])
    csv  = np.concatenate([[0.0], np.cumsum(v)])
    for w in OFI_WINDOWS:
        roll_sum = cs[w:] - cs[:-w]
        roll_v   = csv[w:] - csv[:-w]
        feats[f"ofi_sum_{w}"] = np.concatenate([[0.0] * (w - 1), roll_sum]).astype(np.float32)
        ratio = np.zeros(n, dtype=np.float32)
        ratio[w - 1:] = (roll_sum / np.maximum(roll_v, 1e-9)).astype(np.float32)
        feats[f"ofi_ratio_{w}"] = ratio

    # VPIN: 等 100 bar 桶
    chunk = VPIN_BAR_CHUNK
    vp_n = n // chunk
    if vp_n > 10:
        b_ofi = np.zeros(vp_n, dtype=np.float64)
        b_v   = np.zeros(vp_n, dtype=np.float64)
        for k in range(vp_n):
            s = k * chunk; e = s + chunk
            b_ofi[k] = ofi[s:e].sum(); b_v[k] = v[s:e].sum()
        b_ratio = b_ofi / np.maximum(b_v, 1e-9)
        b_abs   = np.abs(b_ratio)

        def _pad(arr):
            if len(arr) == n: return arr
            return np.concatenate([arr, [arr[-1]] * (n - len(arr))])

        feats["vpin_bar_abs"] = _pad(np.repeat(b_abs, chunk)).astype(np.float32)
        feats["vpin_bar_dir"] = _pad(np.repeat(np.sign(b_ratio), chunk)).astype(np.float32)
        for wb in [5, 10]:
            if vp_n >= wb:
                cs2 = np.concatenate([[0.0], np.cumsum(b_abs)])
                roll = cs2[wb:] - cs2[:-wb]
                feats[f"vpin_ma{wb}"] = _pad(np.repeat(
                    np.concatenate([[0.0] * (wb - 1), roll / wb]), chunk
                )).astype(np.float32)

    # EWMA OFI
    ew = np.exp(np.linspace(-1, 0, 30)); ew /= ew.sum()
    ofi_ew = np.convolve(ofi, ew, mode='full')[:n]
    feats["ofi_ewma_30"] = ofi_ew.astype(np.float32)
    feats["ofi_ewma_ratio"] = (ofi_ew / np.maximum(
        np.convolve(v, ew, mode='full')[:n], 1e-9
    )).astype(np.float32)

    log(f"[ofi] 特征数={len(feats)}, 耗时 {time.time()-t0:.1f}s")
    return feats


def align_ofi_to_ds(ofi_dict, ctx):
    """从 raw 1min bar 级特征 → ds 级 (按 mask 选择)."""
    results = {}
    for split in ("train", "meta_val", "test", "early_stop"):
        mask = ctx.split_rows[split]
        ri = ctx.ds_to_raw[mask].astype(int)
        X = np.column_stack([ofi_dict[n][ri] for n in ofi_dict])
        results[split] = X.astype(np.float32)
    return results


def get_X_full(ctx, extra_raw, ofi_dict, mask):
    """完整特征: baseline 59 + OFI/VPIN 20 = 79."""
    X_base = get_X(ctx, extra_raw, mask)   # 59 features
    # OFI/VPIN
    ri = ctx.ds_to_raw[mask].astype(int)
    X_ofi = np.column_stack([ofi_dict[n][ri] for n in ofi_dict])
    return np.column_stack([X_base, X_ofi])


# ============================================================
# 2. 训练 (完全复现 train_pool_short)
# ============================================================
def joint_masks_weights(ctxs, seed):
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


def train_lgbm_family(use_ofi, label):
    """训练 LGBM family. use_ofi=False → 59特征 (baseline), True → 79特征."""
    import lightgbm as lgb
    log(f"\n{'='*62}")
    log(f"[{label}] 训练开始 (ofi={use_ofi}, h{HORIZON})")
    log(f"{'='*62}")

    ctxs = {s: AssetContext(s, horizon=HORIZON, ds_name=f"ds_{s}_h{HORIZON}") for s in SYMBOLS}
    extras = {s: compute_extra_raw(ctxs[s]) for s in SYMBOLS}

    ofi_dicts = {}
    if use_ofi:
        for s in SYMBOLS:
            ofi_dicts[s] = compute_ofi_vpin_raw(s)

    # early_stop set
    if use_ofi:
        Xes_list = [get_X_full(ctxs[s], extras[s], ofi_dicts[s], ctxs[s].split_rows["early_stop"])
                    for s in SYMBOLS]
    else:
        Xes_list = [get_X(ctxs[s], extras[s], ctxs[s].split_rows["early_stop"]) for s in SYMBOLS]
    yes_list = [ctxs[s].label[ctxs[s].split_rows["early_stop"]].astype(np.float64) for s in SYMBOLS]
    Xes = np.concatenate(Xes_list, axis=0); yes = np.concatenate(yes_list, axis=0)
    del Xes_list, yes_list; gc.collect()
    log(f"[early_stop] shape={Xes.shape}")

    bis = []
    for seed in BAGGED_SEEDS:
        mws = joint_masks_weights(ctxs, seed)
        if use_ofi:
            Xtr_list = [get_X_full(ctxs[s], extras[s], ofi_dicts[s], mws[s][0]) for s in SYMBOLS]
        else:
            Xtr_list = [get_X(ctxs[s], extras[s], mws[s][0]) for s in SYMBOLS]
        ytr_list = [ctxs[s].label[mws[s][0]].astype(np.float64) for s in SYMBOLS]
        w_list = [mws[s][1] for s in SYMBOLS]
        Xtr = np.concatenate(Xtr_list, axis=0); ytr = np.concatenate(ytr_list, axis=0)
        w = np.concatenate(w_list, axis=0)
        del Xtr_list, ytr_list, w_list; gc.collect()
        log(f"  seed={seed} Xtr={Xtr.shape}")

        params = dict(
            objective="binary", metric="auc", learning_rate=0.02,
            num_leaves=127, max_depth=-1, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
            lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=1.0,
            num_threads=config.N_JOBS, verbosity=-1, seed=seed,
        )
        dtr = lgb.Dataset(Xtr, ytr, weight=w); des = lgb.Dataset(Xes, yes, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                      callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                                 lgb.log_evaluation(0)])
        bis.append(m.best_iteration)
        log(f"  seed={seed} best_iter={m.best_iteration} ({time.time()-t0:.0f}s)"
            if False else f"  seed={seed} best_iter={m.best_iteration}")

        # 存 best_iter 和模型（临时）
        root = f"/workspace/models_saved/_exp_h{HORIZON}_{label}"
        os.makedirs(root, exist_ok=True)
        m.save_model(f"{root}/JOINT_lgb_seed{seed}.txt")
        del Xtr, ytr, w, dtr, m; gc.collect()

    del Xes, yes; gc.collect()
    log(f"[{label}] LGBM 训练完成, iters={bis}")
    return bis, ctxs, extras, ofi_dicts if use_ofi else None


# ============================================================
# 3. 评估
# ============================================================
def evaluate_lgbm(bis, ctxs, extras, ofi_dicts, use_ofi, label):
    import lightgbm as lgb
    root = f"/workspace/models_saved/_exp_h{HORIZON}_{label}"

    all_results = {}
    for sym in SYMBOLS:
        ctx = ctxs[sym]
        sym_results = {}
        for split in ("meta_val", "test"):
            mask = ctx.split_rows[split]
            if use_ofi:
                X = get_X_full(ctx, extras[sym], ofi_dicts[sym], mask)
            else:
                X = get_X(ctx, extras[sym], mask)
            n = len(X)
            P = np.zeros((5, n), dtype=np.float32)
            for i, seed in enumerate(BAGGED_SEEDS):
                mm = lgb.Booster(model_file=f"{root}/JOINT_lgb_seed{seed}.txt")
                P[i] = mm.predict(X, num_iteration=bis[i])
                del mm; gc.collect()
            sym_results[split] = P
            del X; gc.collect()
        all_results[sym] = sym_results
    return all_results


def full_eval_table(P_lgb, ctx, sym="ETH"):
    """rank-ensemble + 评估, 返回 meta_val/test 所有口径."""
    n = P_lgb.shape[1] // 5   # 5 seeds
    # P_lgb 是 (5, n)
    R = np.zeros_like(P_lgb, dtype=np.float64)
    for i in range(P_lgb.shape[0]):
        R[i] = np.argsort(np.argsort(P_lgb[i])).astype(np.float64) / (P_lgb.shape[1] - 1)
    p = R.mean(axis=0)
    del R; gc.collect()

    results = {}
    for split in ("meta_val", "test"):
        y = ctx.y(split)
        sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        n_split = len(y)
        assert n_split == P_lgb.shape[1], f"{split}: {n_split} vs {P_lgb.shape[1]}"
        pred = (p >= 0.5).astype(np.int8); conf = np.maximum(p, 1 - p)
        day = sec // 86400; days = np.unique(day)
        n_days = len(days)

        # daily top1%
        sm = np.zeros(n_split, bool)
        for d in days:
            md = day == d; kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
            sub = np.where(md)[0]; sm[sub[np.argsort(-conf[sub])[:kd]]] = True
        sel = np.where(sm)[0]
        acc_d = float((pred[sel] == y[sel]).mean())
        tpd_d = float(len(sel)) / n_days

        # causal Pq
        day_confs = {int(d): conf[day == d] for d in days}
        day_list = days.astype(int).tolist()
        row = {"daily_top1_acc": acc_d, "daily_tpd": tpd_d}
        for q in [99.0, 99.2, 99.5, 99.9]:
            sel2 = np.zeros(n_split, bool)
            for i, d in enumerate(day_list):
                prior = day_list[max(0, i - 30):i]
                if len(prior) < 30: continue
                hist = np.concatenate([day_confs[d2] for d2 in prior])
                tau = float(np.percentile(hist, q))
                sel2[day == d] = conf[day == d] >= tau
            n_sel = int(sel2.sum())
            a = float((pred[sel2] == y[sel2]).mean()) if n_sel else 0
            row[f"causal_P{q}"] = (a, float(n_sel) / n_days, n_sel)

        results[split] = row
        del y, sec, pred, conf, day, days, day_confs, sel, sm
        gc.collect()
    return results


def print_table(label, results):
    log(f"\n--- {label} ---")
    for split in ("meta_val", "test"):
        r = results[split]
        log(f"  [{split}] daily_top1%: acc={r['daily_top1_acc']*100:.2f}%, tpd={r['daily_tpd']:.1f}")
        for q in [99.0, 99.2, 99.5, 99.9]:
            a, tpd, n = r[f"causal_P{q}"]
            log(f"  [{split}] causal P{q}: n={n} ({tpd:.1f}/天) acc={a*100:.2f}%")


# ============================================================
# 4. 主流程
# ============================================================
def main():
    t_all = time.time()
    log(f"\n{'#'*62}")
    log(f"# ETH h15 OFI/VPIN 完整 A/B 对比")
    log(f"# 1:1 复现 train_pool_short pipeline")
    log(f"{'#'*62}\n")

    # --- 先跑 FULL (79特征) ---
    log(">>> Step 1: 训练 FULL 模型 (59 + 20 OFI/VPIN = 79 特征) <<<")
    t0 = time.time()
    bis_full, ctxs_full, extras_full, ofi_full = train_lgbm_family(use_ofi=True, label="FULL")
    log(f"[FULL] 训练耗时 {time.time()-t0:.0f}s")

    log("\n>>> Step 2: 评估 FULL 模型 <<<")
    P_all_full = evaluate_lgbm(bis_full, ctxs_full, extras_full, ofi_full, True, "FULL")
    del ofi_full; gc.collect()

    for sym in SYMBOLS:
        ctx = AssetContext(sym, horizon=HORIZON, ds_name=f"ds_{sym}_h{HORIZON}")
        P_sym = P_all_full[sym]  # dict: {split: (5, n)}
        results = {}
        for split in ("meta_val", "test"):
            ctx_eval = AssetContext(sym, horizon=HORIZON, ds_name=f"ds_{sym}_h{HORIZON}")
            results[split] = full_eval_table_single(P_sym[split], ctx_eval, split, sym)
        print_table(f"{sym} h{HORIZON} FULL (79特征, LGBM5)", results)
        del ctx, results, P_sym; gc.collect()
    del P_all_full, ctxs_full, extras_full; gc.collect()

    # --- 再跑 BASE (59特征) ---
    log("\n>>> Step 3: 训练 BASE 模型 (59 原始特征) <<<")
    t0 = time.time()
    bis_base, ctxs_base, extras_base, _ = train_lgbm_family(use_ofi=False, label="BASE")
    log(f"[BASE] 训练耗时 {time.time()-t0:.0f}s")

    log("\n>>> Step 4: 评估 BASE 模型 <<<")
    P_all_base = evaluate_lgbm(bis_base, ctxs_base, extras_base, None, False, "BASE")

    for sym in SYMBOLS:
        ctx = AssetContext(sym, horizon=HORIZON, ds_name=f"ds_{sym}_h{HORIZON}")
        P_sym = P_all_base[sym]
        results = {}
        for split in ("meta_val", "test"):
            results[split] = full_eval_table_single(P_sym[split], ctx, split, sym)
        print_table(f"{sym} h{HORIZON} BASE (59特征, LGBM5)", results)
        del results, P_sym; gc.collect()
    del P_all_base, ctxs_base, extras_base; gc.collect()

    log(f"\n{'='*62}")
    log(f"  总耗时 {time.time()-t_all:.0f}s")
    log(f"{'='*62}")


def full_eval_table_single(P_split, ctx, split, sym):
    """P_split: (5, n) array for one split."""
    R = np.zeros_like(P_split, dtype=np.float64)
    for i in range(P_split.shape[0]):
        R[i] = np.argsort(np.argsort(P_split[i])).astype(np.float64) / (P_split.shape[1] - 1)
    p = R.mean(axis=0); del R; gc.collect()

    y = ctx.y(split); n = len(y)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    assert n == P_split.shape[1], f"{sym} {split}: {n} vs {P_split.shape[1]}"

    pred = (p >= 0.5).astype(np.int8); conf = np.maximum(p, 1 - p)
    day = sec // 86400; days = np.unique(day); n_days = len(days)

    # daily top1%
    sm = np.zeros(n, bool)
    for d in days:
        md = day == d; kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
        sub = np.where(md)[0]; sm[sub[np.argsort(-conf[sub])[:kd]]] = True
    sel = np.where(sm)[0]
    acc_d = float((pred[sel] == y[sel]).mean()); tpd_d = float(len(sel)) / n_days

    # causal Pq
    day_confs = {int(d): conf[day == d] for d in days}
    day_list = days.astype(int).tolist()
    row = {"daily_top1_acc": acc_d, "daily_tpd": tpd_d}
    for q in [99.0, 99.2, 99.5, 99.9]:
        sel2 = np.zeros(n, bool)
        for i, d in enumerate(day_list):
            prior = day_list[max(0, i - 30):i]
            if len(prior) < 30: continue
            hist = np.concatenate([day_confs[d2] for d2 in prior])
            tau = float(np.percentile(hist, q))
            sel2[day == d] = conf[day == d] >= tau
        n_sel = int(sel2.sum())
        a = float((pred[sel2] == y[sel2]).mean()) if n_sel else 0
        row[f"causal_P{q}"] = (a, float(n_sel) / n_days, n_sel)

    return row


if __name__ == "__main__":
    open(LOG_PATH, "w").close()   # 清空日志
    main()
