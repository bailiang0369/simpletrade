#!/usr/bin/env python3
"""诊断: 为什么无前视(因果阈值)后准确率下降。

1. conf 分箱准确率曲线: 确认 conf 与准确率是否单调(模型是否校准)
2. 每日top1% vs 因果阈值: 选中的样本 conf 分布差异
3. 因果阈值每日信号数分布 vs 当日准确率
4. ens_p: rank平均 vs 概率平均 对 conf 语义的影响
"""
import os, sys, json
sys.path.insert(0, "/workspace")
import numpy as np
import config
from data_store import AssetContext

HORIZONS = [3, 5, 10, 15]
FAMILIES = ["lgb", "xgb", "cat"]


def load_P(sym, h, split="test"):
    Ps = [np.load(f"{config.DS_DIR}/SHORT_{sym}_h{h}_{f}_{split}_P.npy")
          for f in FAMILIES]
    return np.stack(Ps, axis=0).astype(np.float64)


def rank_ens(P):
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)


def prob_ens(P):
    return P.mean(axis=0)


def analyze(sym, h, ens_fn, label):
    ctx = AssetContext(sym, horizon=h, ds_name=f"ds_{sym}_h{h}")
    split = "test"
    P = load_P(sym, h, split)
    p = ens_fn(P)
    y = ctx.y(split)
    sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
    conf = np.abs(p - 0.5) * 2
    pred = (p >= 0.5).astype(np.int8)
    o = np.argsort(sec); sec, conf, pred, y = sec[o], conf[o], pred[o], y[o]
    days = np.unique(sec // 86400)
    day_of = sec // 86400
    return sec, day_of, conf, pred, y, days


def conf_bins(conf, y):
    """conf 分箱准确率 (10 bins)"""
    qs = np.percentile(conf, np.linspace(0, 100, 11))
    rows = []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (conf >= lo) & (conf <= hi)
        if m.sum() < 50:
            continue
        rows.append((lo, hi, m.sum(), float(y[m].mean())))
    return rows


def main():
    out = {}
    for h in HORIZONS:
        print(f"\n{'='*65}\n周期 h{h}\n{'='*65}")
        for sym in ["ETH"]:
            for ens_fn, label in [(rank_ens, "rank平均"), (prob_ens, "概率平均")]:
                sec, doy, conf, pred, y, days = analyze(sym, h, ens_fn, label)
                print(f"\n--- {sym} h{h} [{label}] ---")
                print("conf 分箱准确率 (lo, hi, n, acc):")
                for lo, hi, n, acc in conf_bins(conf, y):
                    bar = "#" * int(acc * 40)
                    print(f"  conf[{lo:.3f},{hi:.3f}) n={n:6d} acc={acc*100:5.2f}% {bar}")

                # 每日top1% 选中的 conf 范围
                sel = np.zeros(len(sec), dtype=bool)
                for d in days:
                    m = doy == d
                    k = max(1, int(np.ceil(int(m.sum()) * 0.01)))
                    sub = np.where(m)[0]
                    sel[sub[np.argsort(-conf[sub])[:k]]] = True
                print(f"top1%(有前视): n={sel.sum()}, acc={y[sel].mean()*100:.2f}%, "
                      f"conf均值={conf[sel].mean():.3f}, conf最小={conf[sel].min():.3f}")

                # 因果阈值(前30天P99) 选中的 conf 范围
                from backtest_short.causal_signals_fixed import causal_signals_fixed
                ts, pr, gt = causal_signals_fixed(sym, h, win_days=30)
                if len(ts):
                    # 找到对应 conf
                    pos = np.searchsorted(sec, ts)
                    sel_c = np.zeros(len(sec), dtype=bool)
                    sel_c[pos] = True
                    print(f"causal(前30天P99): n={len(ts)}, acc={(pr==gt).mean()*100:.2f}%, "
                          f"conf均值={conf[sel_c].mean():.3f}, conf最小={conf[sel_c].min():.3f}")
                    # 每日信号数分布
                    sday = ts // 86400
                    uniq, cnt = np.unique(sday, return_counts=True)
                    print(f"  每日信号数: 均值={cnt.mean():.1f}, 中位={np.median(cnt):.0f}, "
                          f"min={cnt.min()}, max={cnt.max()}, "
                          f">40个的天数={int((cnt>40).sum())}/{len(cnt)}")
                else:
                    print("causal: 0 信号!")
    json.dump({"note": "diagnostics"}, open("/workspace/results/diag_acc_drop.json", "w"))


if __name__ == "__main__":
    main()
