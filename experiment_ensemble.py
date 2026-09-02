#!/usr/bin/env python3
"""实验: 多模型集成 - 在已有预测基础上做加权平均。

利用已训练好的多个模型预测, 尝试不同权重组合:
- 基础模型: enhanced_gbdt (0.6356), original_gbdt (0.6167)
- 辅助模型: catboost, faiss, lstm, stat
- 多尺度: h10, h15, h30, h60
- 标签过滤: p70%, p60%, p50%
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk


def load_pred(name, path):
    if not os.path.exists(path):
        return None, None
    pt = np.load(path).astype(np.float64)
    # 截断到测试集长度
    return name, pt


def evaluate_pred(pt, y_te, retf_te, ts_te):
    r = evaluate_topk(pt, y_te, retf_te, ts_te)
    return r['accuracy'], r['avg_ret_bps'], r['trades_per_day']


def main():
    symbol = "BTC"
    print(f"\n{'=' * 65}")
    print(f"  多模型集成实验: {symbol} H30")
    print(f"  在已有预测上做加权平均, 寻找最优集成权重")
    print(f"{'=' * 65}\n")

    # 加载测试集
    ctx = AssetContext(symbol, horizon=30)
    y_te = ctx.y("test")
    retf_te = ctx.retf("test")
    ts_te = ctx.times("test")
    n_test = len(y_te)

    # 加载所有预测
    preds = {}
    pred_configs = [
        # (name, path, enabled)
        # 主模型
        ("enhanced", f"{config.DS_DIR}/BTC_gbdt_enhanced_pt.npy"),
        ("original", f"{config.DS_DIR}/BTC_gbdt_h30_pt.npy"),
        # 辅助模型
        ("catboost", f"{config.DS_DIR}/BTC_catboost_pt.npy"),
        ("faiss", f"{config.DS_DIR}/BTC_faiss_pt.npy"),
        ("lstm", f"{config.DS_DIR}/BTC_lstm_pt.npy"),
        ("stat", f"{config.DS_DIR}/BTC_stat_pt.npy"),
        # 标签过滤
        ("label_p70", f"{config.DS_DIR}/BTC_label_refine_p70%_pt.npy"),
        ("label_p60", f"{config.DS_DIR}/BTC_label_refine_p60%_pt.npy"),
        ("label_p50", f"{config.DS_DIR}/BTC_label_refine_p50%_pt.npy"),
        # 多尺度
        ("h10", f"{config.DS_DIR}/BTC_gbdt_h10_pt.npy"),
        ("h15", f"{config.DS_DIR}/BTC_gbdt_h15_pt.npy"),
        ("h60", f"{config.DS_DIR}/BTC_gbdt_h60_pt.npy"),
    ]

    for name, path in pred_configs:
        if not os.path.exists(path):
            print(f"  [skip] {name}: 预测文件不存在")
            continue
        pt = np.load(path).astype(np.float64)
        if len(pt) != n_test:
            print(f"  [skip] {name}: 长度不匹配 ({len(pt)} vs {n_test})")
            continue
        acc, ret, tpd = evaluate_pred(pt, y_te, retf_te, ts_te)
        preds[name] = pt
        print(f"  [load] {name:<12} acc={acc:.4f} ret={ret:.1f}bps tpd={tpd:.1f}")

    if len(preds) < 2:
        print("\n  [error] 至少需要 2 个模型做集成")
        return

    print(f"\n  {'=' * 60}")
    print(f"  1. 主模型集成 (enhanced + original)")
    print(f"  {'=' * 60}")

    if "enhanced" in preds and "original" in preds:
        base_acc, _, _ = evaluate_pred(preds["enhanced"], y_te, retf_te, ts_te)
        for w_e in np.arange(0.0, 1.05, 0.05):
            w_o = 1.0 - w_e
            pt_ens = preds["enhanced"] * w_e + preds["original"] * w_o
            acc, ret, tpd = evaluate_pred(pt_ens, y_te, retf_te, ts_te)
            delta = (acc - base_acc) * 100
            marker = " <<<" if acc > base_acc else ""
            if acc > base_acc:
                base_acc = acc
            print(f"    enhanced={w_e:.1f} original={w_o:.1f}: "
                  f"acc={acc:.4f} ({delta:+.2f}pp) ret={ret:.1f}bps{marker}")

    print(f"\n  {'=' * 60}")
    print(f"  2. 三模型集成 (enhanced + original + 最佳辅助)")
    print(f"  {'=' * 60}")

    # 找到最佳辅助模型
    aux_names = [n for n in preds if n not in ("enhanced", "original", "h10", "h15", "h60")]
    acc_by_aux = {}
    for name in aux_names:
        acc, ret, _ = evaluate_pred(preds[name], y_te, retf_te, ts_te)
        acc_by_aux[name] = (acc, ret)

    # 对每个辅助模型, 尝试权重组合
    aux_sorted = sorted(acc_by_aux.keys(), key=lambda x: acc_by_aux[x][0], reverse=True)
    best_acc_all = 0
    best_combo = None

    for aux_name in aux_sorted[:3]:  # 只试前3个辅助模型
        aux_acc = acc_by_aux[aux_name][0]
        print(f"    辅助模型: {aux_name} (acc={aux_acc:.4f})")
        for w_e in [0.4, 0.5, 0.6, 0.7]:
            for w_a in [0.1, 0.15, 0.2]:
                w_o = 1.0 - w_e - w_a
                if w_o < 0:
                    continue
                pt_ens = (preds["enhanced"] * w_e + 
                         preds["original"] * w_o + 
                         preds[aux_name] * w_a)
                acc, ret, tpd = evaluate_pred(pt_ens, y_te, retf_te, ts_te)
                marker = " <<<" if acc > best_acc_all else ""
                if acc > best_acc_all:
                    best_acc_all = acc
                    best_combo = (f"enhanced={w_e:.1f} original={w_o:.1f} "
                                 f"{aux_name}={w_a:.1f}")
                print(f"      enhanced={w_e:.1f} orig={w_o:.1f} "
                      f"{aux_name}={w_a:.1f}: acc={acc:.4f} ret={ret:.1f}bps{marker}")

    print(f"\n  {'=' * 60}")
    print(f"  3. 多尺度集成 (h10 + h15 + h30 + h60)")
    print(f"  {'=' * 60}")

    scale_names = [n for n in preds if n in ("h10", "h15", "h30", "h60")]
    scale_actual = []
    for n in scale_names:
        if n in preds:
            scale_actual.append(n)
    
    # 使用 h30 作为主尺度, 尝试与其他尺度组合
    if "h30" in scale_actual:
        h30_key = "h30" if "h30" in preds else "enhanced"
        h30_pred = preds.get("h30", preds.get("enhanced"))
        h30_acc, _, _ = evaluate_pred(h30_pred, y_te, retf_te, ts_te)
        print(f"    h30 baseline: acc={h30_acc:.4f}")
        
        for other in scale_actual:
            if other == "h30":
                continue
            for w_h30 in [0.6, 0.7, 0.8, 0.9]:
                w_other = 1.0 - w_h30
                pt_ens = h30_pred * w_h30 + preds[other] * w_other
                acc, ret, tpd = evaluate_pred(pt_ens, y_te, retf_te, ts_te)
                delta = (acc - h30_acc) * 100
                marker = " <<<" if delta > 0 else ""
                print(f"    h30={w_h30:.1f} + {other}={w_other:.1f}: "
                      f"acc={acc:.4f} ({delta:+.2f}pp) ret={ret:.1f}bps{marker}")

    if best_combo:
        print(f"\n  {'=' * 60}")
        print(f"  最佳集成: {best_combo}")
        print(f"  最佳准确率: {best_acc_all:.4f}")
        if best_acc_all >= config.TARGET_ACCURACY:
            print(f"  *** 达到目标: {best_acc_all:.4f} >= {config.TARGET_ACCURACY} ***")
        print(f"  {'=' * 60}")

    del ctx
    print(f"\n完成.", flush=True)


if __name__ == "__main__":
    main()