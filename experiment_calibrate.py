#!/usr/bin/env python3
"""逐小时校准: 用 meta_val 的预测偏差, 对 test 预测做小时级偏置校正。

思路:
- 当前模型在特定小时(17,20,0,5)准确率极低
- 用 meta_val 计算每个小时的平均预测偏差, 在 test 上校正
"""
import os, sys, gc, time
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk, _to_epoch_sec

def main():
    print(f"\n{'='*65}")
    print(f"  逐小时校准: 基于增强模型预测")
    print(f"{'='*65}\n")
    
    ctx = AssetContext("BTC", horizon=30)
    
    # 加载增强模型在 meta_val 和 test 上的预测
    # 增强模型应该已经保存了预测
    enhanced_path = f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy"
    enhanced_pv_path = f"{config.DS_DIR}/BTC_gbdt_enhanced_pv.npy"
    
    # 如果不存在, 重新运行增强模型并保存
    if not os.path.exists(enhanced_path):
        print("[WARN] 增强模型预测不存在, 跳过...")
        print("      请先运行 experiment_enhanced.py")
        return
    
    pt = np.load(enhanced_path).astype(np.float64)
    pv = np.load(enhanced_pv_path).astype(np.float64)
    
    # meta_val 和 test 的标签
    ymv = ctx.y("meta_val").astype(np.int8)
    yte = ctx.y("test").astype(np.int8)
    rfte = ctx.retf("test")
    tste = ctx.times("test")
    
    # 时间戳
    ts_mv = ctx.times("meta_val").astype("datetime64[s]").astype(np.int64)
    ts_te = tste.astype("datetime64[s]").astype(np.int64)
    
    # 小时
    hour_mv = (ts_mv % 86400) // 3600
    hour_te = (ts_te % 86400) // 3600
    
    # ---- 方法1: 逐小时偏置校正 ----
    # 计算每个小时在 meta_val 上的平均预测偏差
    correction = np.zeros(24, dtype=np.float64)
    for h in range(24):
        hm = hour_mv == h
        if hm.sum() > 50:
            # 偏差 = 预测均值 - 标签均值 (正数表示过于看多)
            correction[h] = pv[hm].mean() - ymv[hm].mean()
        else:
            correction[h] = 0.0
    
    print("  逐小时校正量:")
    for h in range(24):
        print(f"    Hour {h:2d}: {correction[h]:+.4f}")
    
    # 应用校正
    pt_calibrated = np.clip(pt - correction[hour_te], 0.001, 0.999)
    r = evaluate_topk(pt_calibrated, yte, rfte, tste)
    print(f"\n  >>> 逐小时校正: acc={r['accuracy']:.4f}  tpd={r['trades_per_day']:.1f}  avg_ret={r['avg_ret_bps']:.1f}bps")
    
    # 对比原始
    r0 = evaluate_topk(pt, yte, rfte, tste)
    print(f"  >>> 原始:        acc={r0['accuracy']:.4f}  tpd={r0['trades_per_day']:.1f}  avg_ret={r0['avg_ret_bps']:.1f}bps")
    print(f"  >>> 提升: {r['accuracy'] - r0['accuracy']:+.4f}")
    
    # ---- 方法2: 按小时训练 Logistic 校准 ----
    print("\n  --- 方法2: 逐小时 Logistic 校准 ---")
    pt_cal2 = pt.copy()
    for h in range(24):
        hmv = hour_mv == h
        hte = hour_te == h
        if hmv.sum() > 100 and hte.sum() > 50:
            # 用 meta_val 训练 logistic 校准
            cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            logit_p = np.clip(pv[hmv], 1e-7, 1-1e-7)
            logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
            cal.fit(logit_p, ymv[hmv])
            # 应用
            logit_te = np.clip(pt[hte], 1e-7, 1-1e-7)
            logit_te = np.log(logit_te / (1 - logit_te)).reshape(-1, 1)
            pt_cal2[hte] = cal.predict_proba(logit_te)[:, 1]
    
    r2 = evaluate_topk(pt_cal2, yte, rfte, tste)
    print(f"  >>> 逐小时Logistic: acc={r2['accuracy']:.4f}  tpd={r2['trades_per_day']:.1f}  avg_ret={r2['avg_ret_bps']:.1f}bps")
    
    # ---- 方法3: 小时权重 - 在评估时按小时做二次选择 ----
    # 只在"好小时"选择 top-1%
    print("\n  --- 方法3: 排除最差小时 ---")
    conf = np.maximum(pt, 1 - pt)
    n = len(pt)
    k = max(1, int(n * config.COVERAGE))
    
    # 找出 meta_val 上每个小时的准确率
    hour_acc_mv = {}
    for h in range(24):
        hm = hour_mv == h
        if hm.sum() > 50:
            pred_h = (pv[hm] >= 0.5).astype(np.int8)
            hour_acc_mv[h] = (pred_h == ymv[hm]).mean()
    
    # 按 meta_val 准确率排序小时
    sorted_hours = sorted(hour_acc_mv.items(), key=lambda x: -x[1])
    print("  小时排名 (按meta_val准确率):")
    for h, acc in sorted_hours:
        print(f"    Hour {h:2d}: {acc:.4f}")
    
    # 排除最差的小时, 看准确率变化
    bad_hours = set(h for h, acc in sorted_hours[-5:])  # 最差5小时
    good_mask = ~np.isin(hour_te, list(bad_hours))
    conf_good = conf.copy()
    conf_good[~good_mask] = -1  # 确保不选这些小时
    sel_good = np.argsort(-conf_good)[:k]
    pred_good = (pt >= 0.5).astype(np.int8)
    acc_good = (pred_good[sel_good] == yte[sel_good]).mean()
    print(f"\n  >>> 排除最差5小时: n={good_mask.sum()}/{n}  acc={acc_good:.4f}")
    
    del ctx; gc.collect()
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()