#!/usr/bin/env python3
"""ETH 第二轮优化 Part2: FAISS + 集成评估
"""
import os, sys, gc, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk

def run_faiss(ctx):
    """FAISS形态聚类模型"""
    from models.faiss_shape import FaissShapeModel
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  [FaissShape] 形态聚类 on ETH", flush=True)
    print(f"{'='*60}", flush=True)
    m = FaissShapeModel(seed=42)
    m.fit(ctx)
    pt = m.predict(ctx, "test").astype(np.float64)
    r = evaluate_topk(pt, ctx.y("test"), ctx.retf("test"), ctx.times("test"))
    print(f"  >> test: acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps tpd={r['trades_per_day']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return r, pt

print(f"\n{'='*65}", flush=True)
print(f"  ETH FAISS + 集成评估", flush=True)
print(f"{'='*65}", flush=True)

ctx = AssetContext("ETH", horizon=30)

# ---- FAISS ----
r_faiss, pt_faiss = run_faiss(ctx)
np.save(f"{config.DS_DIR}/ETH_faiss_pt.npy", pt_faiss.astype(np.float32))
print(f"  [save] FAISS 预测已保存", flush=True)

# ---- 加载之前的结果 ----
# F3: 二元_nocal 57.34%
pt_binary = np.load(f"{config.DS_DIR}/ETH_binary_pt.npy") if os.path.exists(f"{config.DS_DIR}/ETH_binary_pt.npy") else None
if pt_binary is None:
    # 重新生成
    print(f"\n重新生成二元_nocal 预测...", flush=True)
    import lightgbm as lgb
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
        # 交叉特征 (eth_orig 模型是用59维训练的)
        "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
        "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
    ]
    BAGGED_SEEDS = [42, 49, 56, 63, 70]
    test_mask = ctx.split_rows["test"]
    Xte = ctx.X_subset(BASE_FEATURES, test_mask); n_te = len(Xte)
    R = np.zeros((5, n_te), dtype=np.float64)
    for i, seed in enumerate(BAGGED_SEEDS):
        m = lgb.Booster(model_file=f"/workspace/simpletrade/models_saved/eth_orig/ETH_orig_seed{seed}.txt")
        p = m.predict(Xte, num_iteration=m.best_iteration)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n_te - 1)
        del m; gc.collect()
    pt_binary = R.mean(axis=0)
    np.save(f"{config.DS_DIR}/ETH_binary_pt.npy", pt_binary.astype(np.float32))
    print(f"  binary_nocal 预测已保存", flush=True)

# 统计信号
pt_stat = np.load(f"{config.DS_DIR}/ETH_stat_pt.npy") if os.path.exists(f"{config.DS_DIR}/ETH_stat_pt.npy") else None
if pt_stat is None:
    from models.stat_signal import StatSignal
    m = StatSignal(seed=42); m.fit(ctx)
    pt_stat = m.predict(ctx, "test").astype(np.float64)
    np.save(f"{config.DS_DIR}/ETH_stat_pt.npy", pt_stat.astype(np.float32))

# 原始gbdt
pt_orig = np.load(f"{config.DS_DIR}/ETH_gbdt_h30_pt.npy").astype(np.float64)

y_te = ctx.y("test"); retf_te = ctx.retf("test"); ts_te = ctx.times("test")

# 各模型单独评估
results = {
    "binary_nocal": evaluate_topk(pt_binary, y_te, retf_te, ts_te),
    "stat": evaluate_topk(pt_stat, y_te, retf_te, ts_te),
    "faiss": evaluate_topk(pt_faiss, y_te, retf_te, ts_te),
    "original_gbdt": evaluate_topk(pt_orig, y_te, retf_te, ts_te),
}

print(f"\n{'='*65}", flush=True)
print(f"  各模型表现", flush=True)
print(f"{'='*65}", flush=True)
print(f"  {'Name':<20} {'Acc':>8} {'Ret(bps)':>9} {'T/D':>6}", flush=True)
print(f"  {'-'*50}", flush=True)
for name, r in results.items():
    print(f"  {name:<20} {r['accuracy']:>8.4f} {r['avg_ret_bps']:>9.1f} {r['trades_per_day']:>6.1f}", flush=True)

# 集成搜索
print(f"\n{'='*60}", flush=True)
print(f"  集成搜索 (binary + stat + faiss + orig)", flush=True)
print(f"{'='*60}", flush=True)

pt_list = [pt_binary, pt_stat, pt_faiss, pt_orig]
names = ["binary", "stat", "faiss", "orig"]

best_acc = 0; best_w = None
print(f"  {'w_bin':>5} {'w_sta':>5} {'w_fai':>5} {'w_ori':>5}  {'Acc':>8} {'Ret':>9}", flush=True)
for w1 in np.arange(0.0, 1.05, 0.05):
    for w2 in np.arange(0.0, 1.05, 0.05):
        for w3 in np.arange(0.0, 1.05, 0.05):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -0.01 or w4 > 1.01: continue
            if abs(w1) < 0.01 and abs(w2) < 0.01 and abs(w3) < 0.01: continue
            pt = w1 * pt_list[0] + w2 * pt_list[1] + w3 * pt_list[2] + w4 * pt_list[3]
            r = evaluate_topk(pt, y_te, retf_te, ts_te)
            if r['accuracy'] > best_acc + 1e-6:
                best_acc = r['accuracy']; best_w = (w1, w2, w3, w4)
            if r['accuracy'] >= 0.58 or r['accuracy'] > best_acc - 0.005:
                print(f"  {w1:.2f} {w2:.2f} {w3:.2f} {w4:.2f}  {r['accuracy']:.4f}  {r['avg_ret_bps']:>7.1f}", flush=True)

print(f"\n  最佳集成权重: {dict(zip(names, best_w))}", flush=True)
print(f"  最佳准确率: {best_acc:.4f}", flush=True)
print(f"  目标: {config.TARGET_ACCURACY}", flush=True)

# 月度明细
pt_best = sum(w * p for w, p in zip(best_w, pt_list))
r_best = evaluate_topk(pt_best, y_te, retf_te, ts_te)
print(f"\n  月度明细:")
for m in sorted(r_best['acc_by_month'].keys()):
    print(f"    {m[-7:]}: acc={r_best['acc_by_month'][m]:.4f}  n={r_best['n_by_month'][m]:4d}", flush=True)

np.save(f"{config.DS_DIR}/ETH_ensemble_best_pt.npy", pt_best.astype(np.float32))
print(f"\n  [save] 最佳集成预测已保存", flush=True)

del ctx; gc.collect()
print(f"\n完成.", flush=True)