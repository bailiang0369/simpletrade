#!/usr/bin/env python3
"""诊断分析: 当前 GBDT 模型在 test 上的错误模式。"""
import os, sys, gc, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from evaluate import evaluate_topk, _to_epoch_sec

FEATURES = [
    "lr_5","lr_15","lr_30","lr_120","lr_240","mom_60",
    "z_10","z_30","z_60","z_120",
    "rvol_30","rvol_60","rvol_ratio_60_5","rvol_z_60","rvol_dir",
    "pos_30","pos_60","pos_120","pos_240",
    "dd_240","ru_240",
    "hh_dd_60","ll_ru_60","body_pos_60",
    "body_ratio","up_wick","lo_wick","ngreen_10","gap","max_range_30",
    "tbr_z_30","cvd_30","cvd_60",
    "buyvol_strength_30","tb_act_60","ts_act_60","tb_acc_30",
    "cvd_dir_30","tbr_hi_60","lr_skew_60","up_body_ratio_30","mom_align_30_240",
    "hour_sin","hour_cos","dow_sin","dow_cos","is_us","is_eu","ret_day",
    "pos_tbr_interact","vol_mom_interact","pos_cvd_interact",
    "di_spread","di_uptrend","mom_vol_confirm","z_divergence","cvd_accel","vol_cvd_interact","di_plus",
]

def main():
    symbol = "BTC"
    print(f"\n{'='*65}")
    print(f"  诊断分析: {symbol} 当前 GBDT 模型错误模式")
    print(f"{'='*65}\n")
    
    ctx = AssetContext(symbol, horizon=30)
    pt_path = f"{config.DS_DIR}/{symbol}_gbdt_h30_pt.npy"
    if not os.path.exists(pt_path):
        pt_path = f"{config.DS_DIR}/{symbol}_gbdt_pt.npy"
    if not os.path.exists(pt_path):
        print(f"[ERROR] 找不到预测文件")
        return
    
    pt = np.load(pt_path).astype(np.float64)
    y = ctx.y("test").astype(np.int8)
    rf = ctx.retf("test")
    ts = ctx.times("test")
    print(f"  [loaded] {pt_path} ({len(pt)} samples)")
    
    # top-1% 选择
    conf = np.maximum(pt, 1 - pt)
    pred = (pt >= 0.5).astype(np.int8)
    n = len(pt)
    k = max(1, int(n * config.COVERAGE))
    sel = np.argsort(-conf)[:k]
    
    correct = pred[sel] == y[sel]
    wrong = ~correct
    print(f"\n  top-1% 样本: {k}")
    print(f"    正确: {correct.sum()} ({correct.mean()*100:.1f}%)")
    print(f"    错误: {wrong.sum()} ({wrong.mean()*100:.1f}%)")
    
    # 获取特征
    te_sel_mask = np.zeros(n, dtype=bool)
    te_sel_mask[sel] = True
    X_sel = ctx.X_subset(FEATURES, te_sel_mask)
    X_correct = X_sel[correct]
    X_wrong = X_sel[wrong]
    
    # 特征差异分析
    print(f"\n  --- 特征差异 (正确 vs 错误, top-20) ---")
    print(f"  {'特征':<25} {'正确均值':>10} {'错误均值':>10} {'Cohen-d':>8}")
    print(f"  {'-'*55}")
    diffs = []
    for j, fname in enumerate(FEATURES):
        if X_correct.shape[0] > 5 and X_wrong.shape[0] > 5:
            mu_c = X_correct[:, j].mean()
            mu_w = X_wrong[:, j].mean()
            std_c = X_correct[:, j].std() + 1e-10
            std_w = X_wrong[:, j].std() + 1e-10
            ps = np.sqrt((std_c**2 + std_w**2) / 2)
            d = (mu_c - mu_w) / ps if ps > 0 else 0
            diffs.append((abs(d), d, fname, mu_c, mu_w))
    diffs.sort(key=lambda x: -x[0])
    for abs_d, d, name, mu_c, mu_w in diffs[:20]:
        print(f"  {name:<25} {mu_c:>10.4f} {mu_w:>10.4f} {d:>8.3f}")
    
    # 错误的时间分布 (按小时)
    print(f"\n  --- 错误按小时分布 ---")
    te_sec = _to_epoch_sec(ts)
    sel_hour = (te_sec[sel] % 86400) // 3600
    for h in range(24):
        hm = sel_hour == h
        if hm.sum() > 5:
            ha = (pred[sel][hm] == y[sel][hm]).mean()
            print(f"    Hour {h:2d}: n={hm.sum():4d} acc={ha:.4f}")
    
    # 正确 vs 错误样本的收益分布
    print(f"\n  --- 收益分析 ---")
    print(f"  正确样本平均收益: {rf[sel][correct].mean()*1e4:.1f}bps")
    print(f"  错误样本平均收益: {rf[sel][wrong].mean()*1e4:.1f}bps")
    print(f"  错误中实际涨: {(y[sel][wrong]==1).mean()*100:.1f}% (预测跌)")
    print(f"  错误中实际跌: {(y[sel][wrong]==0).mean()*100:.1f}% (预测涨)")
    
    # 按波动率分析
    rvol_idx = FEATURES.index("rvol_60")
    vol_c = X_sel[:, rvol_idx]
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        hm = vol_c > thresh
        if hm.sum() > 5:
            ha = (pred[sel][hm] == y[sel][hm]).mean()
            print(f"  波动率>{thresh:.1f}: n={hm.sum():4d} acc={ha:.4f}")
    
    # 错误样本的置信度分布
    print(f"\n  --- 置信度分析 ---")
    conf_correct = conf[sel][correct]
    conf_wrong = conf[sel][wrong]
    print(f"  正确样本平均置信度: {conf_correct.mean():.4f}")
    print(f"  错误样本平均置信度: {conf_wrong.mean():.4f}")
    for thresh in [0.6, 0.7, 0.8, 0.9]:
        hm = conf[sel] > thresh
        if hm.sum() > 5:
            ha = (pred[sel][hm] == y[sel][hm]).mean()
            print(f"  置信度>{thresh:.1f}: n={hm.sum():4d} acc={ha:.4f}")
    
    del ctx; gc.collect()
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()