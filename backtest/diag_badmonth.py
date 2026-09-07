#!/usr/bin/env python3
"""诊断 BTC test R2 坏月选中样本的根因: 特征分布对/错 + 目标月vs好月。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

SYMBOL = "BTC"
TARGET_MONTH = "2026-05"
P99 = 99.0
WIN_SAMPLES = 130_000
FAMS = ["lgb", "xgb", "cat"]
FNAMES = ["lr_120", "z_120", "z_60", "rvol_60", "pos_120", "dd_240",
          "ru_240", "mom_60", "ret_day", "dn_run_len", "regime_dn_bear"]


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def main():
    ctx = AssetContext(SYMBOL, horizon=30)
    mv = load_fused(SYMBOL, "meta_val")
    te = load_fused(SYMBOL, "test")
    mv_y = ctx.y("meta_val"); te_y = ctx.y("test")
    mv_sec = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    te_sec = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)

    mv_conf = np.maximum(mv, 1 - mv)
    te_conf = np.maximum(te, 1 - te)
    te_pred = (te >= 0.5).astype(np.int8)

    # ---- R2 逐日滚动阈值选样 ----
    day = te_sec // 86400
    days = np.unique(day)
    hist = list(mv_conf[-WIN_SAMPLES:])
    sel_list, day_of = [], []
    for dd in days:
        md = day == dd
        tau_d = np.percentile(np.asarray(hist), P99)
        idx = np.where(md & (te_conf >= tau_d))[0]
        sel_list.append(idx); day_of.append(np.full(len(idx), dd))
        hist.extend(te_conf[md])
        if len(hist) > WIN_SAMPLES * 2:
            del hist[:len(hist) - WIN_SAMPLES * 2]
    sel = np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)
    sel_day = np.concatenate(day_of) if day_of else np.array([], dtype=np.int64)
    sel_month = te_sec[sel].astype("datetime64[s]").astype("datetime64[M]")

    print(f"总选中 {len(sel)} 样本")
    by_month = {}
    for u in np.unique(sel_month):
        m = sel_month == u
        acc = float((te_pred[sel][m] == te_y[sel][m]).mean())
        n = int(m.sum())
        by_month[str(u)[:7]] = (n, acc)
        print(f"  {str(u)[:7]}: n={n} acc={acc:.4f}")
    good = max(by_month, key=lambda k: by_month[k][1])

    test_gl = np.where(ctx.split_rows["test"])[0]

    def group_vals(gmask):
        s = sel[gmask]
        gl = test_gl[s]
        out = {}
        for f in FNAMES:
            arr = pq.read_table(f"{config.DS_DIR}/ds_{SYMBOL}.parquet", columns=[f])[f]
            out[f] = np.asarray(arr, dtype=np.float32)[gl]
        out["_pred"] = te_pred[s]; out["_y"] = te_y[s]
        return out

    target = sel_month.astype(str) == np.datetime64(TARGET_MONTH, "M").astype("datetime64[M]").astype(str)
    ctrl = sel_month.astype(str) == good
    tg = group_vals(target); cg = group_vals(ctrl)

    print(f"\n=== 目标月 {TARGET_MONTH} (n={len(tg['_pred'])}) vs 好月 {good} (n={len(cg['_pred'])}) ===")
    print(f"{'feat':<16}{'目标P25':>9}{'目标P50':>9}{'目标P75':>9}{'好月P25':>9}{'好月P50':>9}")
    for f in FNAMES:
        a = np.percentile(tg[f], [25, 50, 75]); b = np.percentile(cg[f], [25, 50])
        print(f"{f:<16}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>9.4f}{b[0]:>9.4f}{b[1]:>9.4f}")
    print(f"\n目标月  做多={int((tg['_pred']==1).sum()):>4} 做空={int((tg['_pred']==0).sum()):>4} | label涨={int((tg['_y']==1).sum()):>4} label跌={int((tg['_y']==0).sum()):>4}")
    print(f"好月    做多={int((cg['_pred']==1).sum()):>4} 做空={int((cg['_pred']==0).sum()):>4}")

    okm = tg["_pred"] == tg["_y"]
    print(f"\n=== {TARGET_MONTH} 对(n={int(okm.sum())})/错(n={int((~okm).sum())}) 特征 ===")
    print(f"{'feat':<16}{'对P25':>9}{'对P50':>9}{'错P50':>9}{'错P75':>9}")
    for f in FNAMES:
        o = np.percentile(tg[f][okm], [25, 50]); e = np.percentile(tg[f][~okm], [50, 75])
        print(f"{f:<16}{o[0]:>9.4f}{o[1]:>9.4f}{e[0]:>9.4f}{e[1]:>9.4f}")


if __name__ == "__main__":
    main()