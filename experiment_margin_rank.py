#!/usr/bin/env python3
"""实验: 基于 raw margin (log-odds) 的 top-1% 选择。

当前评估使用 conf = max(p, 1-p) 选择 top-1%。
但模型输出 p = sigmoid(f) 是概率, 置信度 max(p,1-p) 可能不是最优排序指标。

本实验尝试:
  方法1: 直接用 raw margin |f| (log-odds 绝对值) 选择 top-1%
  方法2: 用 p 本身选择 top-1% (仅看涨概率, 不看置信度)
  方法3: 用 p - 0.5 选择 top-1% (偏差幅度)
  方法4: 用 p * (1-p) 的倒数选择 top-1% (信息熵最小)

在已有预测上做后处理, 无需重新训练。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from evaluate import evaluate_topk


def evaluate_topk_custom(p, y, retf, times, rank_by="conf"):
    """按自定义排序指标选择 top-1%。
    
    rank_by:
        "conf"    - max(p, 1-p)  (默认)
        "margin"  - |log(p/(1-p))|  (raw margin 绝对值)
        "p_raw"   - p 本身 (仅看涨概率)
        "p_dev"   - p - 0.5  (偏差幅度, 有方向)
        "entropy" - 1 / (p * (1-p))  (信息熵倒数, 越确定越好)
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    retf = np.asarray(retf, dtype=np.float64)
    n = len(p)
    k = max(1, int(round(n * config.COVERAGE)))
    
    if rank_by == "conf":
        score = np.maximum(p, 1 - p)
    elif rank_by == "margin":
        # log-odds = log(p/(1-p)), |log-odds| 越大越极端
        p_clip = np.clip(p, 1e-10, 1 - 1e-10)
        log_odds = np.log(p_clip / (1 - p_clip))
        score = np.abs(log_odds)
    elif rank_by == "p_raw":
        score = p
    elif rank_by == "p_dev":
        score = np.abs(p - 0.5)
    elif rank_by == "entropy":
        # 信息熵倒数: 越确定熵越小, 倒数越大
        p_clip = np.clip(p, 1e-10, 1 - 1e-10)
        entropy = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))
        score = 1.0 / (entropy + 1e-10)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")
    
    sel = np.argsort(-score)[:k]
    pred = (p >= 0.5).astype(np.int8)
    acc = (pred[sel] == y[sel]).mean()
    avg_ret = float(retf[sel].mean()) * 1e4
    
    return {
        "rank_by": rank_by,
        "accuracy": float(acc),
        "avg_ret_bps": float(avg_ret),
        "conf_mean": float(score[sel].mean()),
    }


def main():
    symbol = "BTC"
    print(f"\n{'=' * 65}")
    print(f"  Raw Margin 排序实验: {symbol} H30")
    print(f"  比较不同排序指标对 top-1% 准确率的影响")
    print(f"{'=' * 65}\n")

    # 加载已有预测
    pred_paths = [
        ("enhanced_gbdt", f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy"),
        ("original_gbdt", f"{config.DS_DIR}/BTC_gbdt_h30_pt.npy"),
    ]
    
    # 加载测试集标签
    from data_store import AssetContext
    ctx = AssetContext(symbol, horizon=30)
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")
    
    rank_methods = ["conf", "margin", "p_raw", "p_dev", "entropy"]
    
    for model_name, pred_path in pred_paths:
        if not os.path.exists(pred_path):
            print(f"  [skip] {model_name}: 预测文件不存在 {pred_path}")
            continue
        
        pt = np.load(pred_path).astype(np.float64)
        print(f"\n  {'=' * 60}")
        print(f"  {model_name}")
        print(f"  {'=' * 60}")
        print(f"  {'Rank Method':<15} {'Acc':>8} {'Ret(bps)':>9}")
        print(f"  {'-' * 35}")
        
        best_acc = 0
        best_method = "conf"
        for method in rank_methods:
            r = evaluate_topk_custom(pt, y_te, retf_te, ts_te, rank_by=method)
            marker = " <<<" if r['accuracy'] > best_acc else ""
            if r['accuracy'] > best_acc:
                best_acc = r['accuracy']
                best_method = method
            print(f"  {method:<15} {r['accuracy']:>8.4f} {r['avg_ret_bps']:>9.1f}{marker}")
        
        print(f"  {'-' * 35}")
        print(f"  最佳: {best_method} (acc={best_acc:.4f})")
    
    # 对比原始 evaluate_topk
    print(f"\n  {'=' * 60}")
    print(f"  原始 evaluate_topk 对比 (conf=default)")
    print(f"  {'=' * 60}")
    for model_name, pred_path in pred_paths:
        if not os.path.exists(pred_path):
            continue
        pt = np.load(pred_path).astype(np.float64)
        r = evaluate_topk(pt, y_te, retf_te, ts_te)
        print(f"  {model_name:<20} acc={r['accuracy']:.4f} ret={r['avg_ret_bps']:.1f}bps")
    
    del ctx
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()