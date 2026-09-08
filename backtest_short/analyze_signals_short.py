#!/usr/bin/env python3
"""短周期信号分析: 分析 3/5min 预测信号(每日 top1% 置信度样本)的共同特征。

分析内容:
  1. 时段分布: 信号按 小时(北京时间)/星期/交易时段 的分布 vs 全体样本, 找集中时段
  2. 特征偏离: 信号样本的每个特征均值 vs 全体样本均值(标准化差), 找最显著偏离特征
  3. 方向对比: 看涨信号 vs 看跌信号 的特征差异

用法:
  python -m backtest_short.analyze_signals_short 5 ETH
  python -m backtest_short.analyze_signals_short 3 ETH
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config
from data_store import AssetContext

FAMILIES = ["lgb", "xgb", "cat"]
SEEDS = [42, 49, 56, 63, 70]
BEIJING_OFFSET = 8 * 3600  # UTC+8


def load_P(symbol, horizon, split):
    P = np.concatenate([
        np.load(f"{config.DS_DIR}/SHORT_{symbol}_h{horizon}_{f}_{split}_P.npy")
        for f in FAMILIES
    ], axis=0).astype(np.float64)
    return P


def family_ranks(P):
    n = P.shape[1]
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R


def daily_topk_sel(p, sec_arr, frac=0.01):
    """每日 top1% 高置信度样本索引(与 backtest 一致)。"""
    conf = np.maximum(p, 1 - p)
    day = sec_arr // 86400
    days = np.unique(day)
    n = len(p); sel = np.zeros(n, bool)
    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * frac)))
        sub = np.where(md)[0]
        sel[sub[np.argsort(-conf[sub])[:kd]]] = True
    return np.where(sel)[0]


def select_signals(symbol, horizon, split, w):
    """用 family 权重 w 组合出最终置信度, 选出每日 top1% 信号。返回 idx 和方向。"""
    P = load_P(symbol, horizon, split)
    R = family_ranks(P)
    wf = np.repeat(np.asarray(w, np.float64), len(SEEDS))
    p = (wf[:, None] * R).sum(axis=0)
    idx = daily_topk_sel(p, None, 0.01)  # 占位, 下面重算
    return P, R, p


def hour_dist(ts_sec, sel_idx, all_idx):
    """北京时间小时分布: 信号 vs 全体。返回 {hour: (sig_share, all_share, n_sig)}。"""
    def _dist(idx):
        h = ((ts_sec[idx] + BEIJING_OFFSET) % 86400) // 3600
        vals, cnts = np.unique(h, return_counts=True)
        d = dict(zip(vals.astype(int), cnts))
        return d
    sig = _dist(sel_idx); all_ = _dist(all_idx)
    n_sig = len(sel_idx); n_all = len(all_idx)
    out = {}
    for h in range(24):
        out[h] = (sig.get(h, 0) / n_sig, all_.get(h, 0) / n_all, sig.get(h, 0))
    return out


def dow_dist(ts_sec, sel_idx, all_idx):
    def _dist(idx):
        dow = ((ts_sec[idx] + BEIJING_OFFSET) // 86400) % 7  # 0=Thu (1970-01-01)
        vals, cnts = np.unique(dow, return_counts=True)
        return dict(zip(vals.astype(int), cnts))
    sig = _dist(sel_idx); all_ = _dist(all_idx)
    n_sig = len(sel_idx); n_all = len(all_idx)
    return {d: (sig.get(d, 0) / n_sig, all_.get(d, 0) / n_all, sig.get(d, 0)) for d in range(7)}


def feature_deviation(ctx, sel_idx, all_idx, topk=25):
    """信号 vs 全体 的特征均值差(标准化为全体 std 单位)。"""
    nf = len(ctx.feat_names)
    dev = []
    for j, nm in enumerate(ctx.feat_names):
        col = ctx.Xall[:, j]
        mu_all = col[all_idx].mean()
        sd_all = col[all_idx].std() + 1e-9
        mu_sig = col[sel_idx].mean()
        dev.append((nm, (mu_sig - mu_all) / sd_all, mu_sig, mu_all, sd_all))
    dev.sort(key=lambda x: -abs(x[1]))
    return dev[:topk]


def main():
    horizon = int(sys.argv[1])
    symbol = sys.argv[2].upper()
    # 用之前调优保存的 family 权重(若存在), 否则等权
    import json
    wp = f"/workspace/models_saved/pool_short_h{horizon}/ens_family_w.json"
    if os.path.exists(wp):
        d = json.load(open(wp))
        w = d.get(symbol, {}).get("w", [1/3]*3)
    else:
        w = [1/3]*3
    w = np.asarray(w, np.float64)
    print(f"# SHORT h{horizon} {symbol} 信号分析 (family权重={np.round(w,2)})")

    ctx = AssetContext(symbol, horizon=horizon, ds_name=f"ds_{symbol}_h{horizon}")
    split = "test"
    P = load_P(symbol, horizon, split)
    R = family_ranks(P)
    wf = np.repeat(w, len(SEEDS)); wf = wf / wf.sum()
    p = (wf[:, None] * R).sum(axis=0)
    y = ctx.y(split)
    sec_arr = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    idx = daily_topk_sel(p, sec_arr, 0.01)
    all_idx = np.arange(len(p))
    n_sig = len(idx)
    acc = ( (p[idx] >= 0.5).astype(np.int8) == y[idx] ).mean()
    up_sig = idx[p[idx] >= 0.5]; dn_sig = idx[p[idx] < 0.5]
    print(f"  信号数={n_sig} 覆盖={n_sig/len(p):.4f} 准确率={acc:.4f} 看涨={len(up_sig)} 看跌={len(dn_sig)}")

    # ---- 1. 时段分布 ----
    print("\n== 时段分布 (北京时间) ==")
    print("  hour |  信号占比 |  全体占比 | 偏离(pp) | 信号数")
    hd = hour_dist(sec_arr, idx, all_idx)
    for h in range(24):
        s, a, n = hd[h]
        pp = (s - a) * 100
        flag = " <== 集中" if abs(pp) >= 1.0 else ""
        print(f"  {h:02d}:00 |  {s*100:6.2f}% | {a*100:6.2f}% | {pp:+6.2f} | {n:5d}{flag}")

    print("\n  weekday |  信号占比 |  全体占比 | 偏离(pp)")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dd = dow_dist(sec_arr, idx, all_idx)
    for d in range(7):
        s, a, n = dd[d]
        pp = (s - a) * 100
        print(f"  {names[d]:6s} |  {s*100:6.2f}% | {a*100:6.2f}% | {pp:+6.2f}")

    # 交易时段(北京时间): 亚盘 9-15, 欧盘 15-21, 美盘 21-3, 夜盘 3-9
    print("\n  交易时段(北京时间) | 信号占比 | 全体占比 | 偏离(pp)")
    def _ses(mask):
        return mask.sum()
    h_arr = ((sec_arr + BEIJING_OFFSET) % 86400) // 3600
    for name, lo, hi in [("亚盘9-15", 9, 15), ("欧盘15-21", 15, 21), ("美盘21-03", 21, 27), ("夜盘03-09", 3, 9)]:
        m_sig = ((h_arr[idx] >= lo) & (h_arr[idx] < hi)).mean()
        m_all = ((h_arr[all_idx] >= lo) & (h_arr[all_idx] < hi)).mean()
        print(f"  {name:14s} | {m_sig*100:6.2f}% | {m_all*100:6.2f}% | {(m_sig-m_all)*100:+6.2f}")

    # ---- 2. 特征偏离 ----
    print("\n== 特征偏离 (信号 vs 全体, 单位=全体标准差) ==")
    print("  特征 | 偏离(z) | 信号均值 | 全体均值 | 全体std")
    for nm, z, ms, ma, sd in feature_deviation(ctx, idx, all_idx, 20):
        print(f"  {nm:24s} | {z:+6.2f} | {ms:10.4f} | {ma:10.4f} | {sd:9.4f}")

    # ---- 3. 看涨 vs 看跌信号特征 ----
    print("\n== 看涨 vs 看跌 信号特征差异 ==")
    print("  特征 | 看涨均值 | 看跌均值 | 差(z)")
    nf = len(ctx.feat_names)
    devs = []
    for j, nm in enumerate(ctx.feat_names):
        col = ctx.Xall[:, j]
        mu_up = col[up_sig].mean(); mu_dn = col[dn_sig].mean()
        sd_all = col[all_idx].std() + 1e-9
        devs.append((nm, mu_up, mu_dn, (mu_up - mu_dn) / sd_all))
    devs.sort(key=lambda x: -abs(x[3]))
    for nm, mu_up, mu_dn, z in devs[:15]:
        print(f"  {nm:24s} | {mu_up:10.4f} | {mu_dn:10.4f} | {z:+6.2f}")

    print("\n# done")


if __name__ == "__main__":
    main()