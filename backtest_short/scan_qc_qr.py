#!/usr/bin/env python3
"""锁参数评估: meta_val 扫 (q_conf, q_rvol) → 锁最优 → test 仅一次无前视评估。

过滤逻辑: conf >= τ_hist(q_conf)  AND  rvol5 >= η_hist(q_rvol)
阈值 τ_hist / η_hist 均为同向历史分布的滚动分位数, 只依赖 <d 天数据。
多空独立。
"""
import os, sys, gc
sys.path.insert(0, "/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config
from data_store import AssetContext

H = 15
SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]


def rank1d(p):
    return np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1)


def load_lgb(sym, split):
    P = np.stack([np.load(f"{config.DS_DIR}/SHORT_{sym}_h{H}_lgb_seed{s}_{split}_P.npy").astype(np.float64)
                  for s in SEEDS], axis=0)
    return np.mean([rank1d(r) for r in P], axis=0)


def load_rvol(sym, split):
    """构建 ds_ts 对齐的 rvol5 (1-min log return rolling std 5)。"""
    raw = pd.read_parquet(f"{config.DS_DIR}/raw_{sym}.parquet", columns=["close", "ts"])
    c = raw["close"].to_numpy(np.float64)
    lr = np.full(len(c), np.nan); lr[1:] = np.log(c[1:] / c[:-1])
    lr = np.nan_to_num(lr, nan=0.0)
    m = np.convolve(lr, np.ones(5) / 5, mode="full")[:len(lr)]
    m2 = np.convolve(lr ** 2, np.ones(5) / 5, mode="full")[:len(lr)]
    rv = np.sqrt(np.clip(m2 - m ** 2, 0, None))
    # 对齐 ds_ts
    ctx = AssetContext(sym, H, ds_name=f"ds_{sym}_h{H}")
    ts2i = {int(t): i for i, t in enumerate(raw["ts"].to_numpy())}
    idx = np.array([ts2i[int(t)] for t in ctx.ds_ts], dtype=np.int64)
    return rv[idx][ctx.split_rows[split]]


def evaluate(pred, conf, rvol, y, sec, q_conf, q_rvol, win=30, cold=30):
    """无前视 + 过滤: conf分位 AND rvol分位。多空独立。"""
    day_of = sec // 86400
    days = np.unique(day_of)
    n = len(conf)
    sel = np.zeros(n, dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di - win):di]
        if len(prior) < cold:
            continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hc = np.isin(day_of, prior) & (pred == side)
            if hc.sum() < cold or m.sum() == 0:
                continue
            tau = float(np.percentile(conf[hc], q_conf))
            eta = float(np.percentile(rvol[hc], q_rvol))
            sel[m & (conf >= tau) & (rvol >= eta)] = True
    acc = float((pred[sel] == y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / len(days)
    return acc * 100, tpd, int(sel.sum())


def scan(sym):
    ctx = AssetContext(sym, H, ds_name=f"ds_{sym}_h{H}")
    print(f"\n{'='*60}\n{sym} | 10seed LGB集成 + rvol5过滤\n{'='*60}")

    # meta_val 网格搜索
    print("\n--- meta_val 网格搜索 (锁参数) ---")
    p_mv = load_lgb(sym, "meta_val")
    y_mv = ctx.y("meta_val")
    sec_mv = np.asarray(ctx.times("meta_val")).astype("datetime64[s]").astype(np.int64)
    conf_mv = np.abs(p_mv - 0.5) * 2
    pred_mv = (p_mv >= 0.5).astype(np.int8)
    rv_mv = load_rvol(sym, "meta_val")

    best = None
    best_score = 0  # score = acc * log(tpd) 平衡质量与数量
    for qc in [98, 98.5, 99, 99.2, 99.5]:
        for qr in [0, 50, 60, 70, 80, 90]:
            acc, tpd, n = evaluate(pred_mv, conf_mv, rv_mv, y_mv, sec_mv, qc, qr, 30, 30)
            if tpd < 8 or tpd > 40:
                continue
            # score: 65% 起线性加分
            score = acc * np.log(tpd + 1)
            if score > best_score:
                best_score = score
                best = (qc, qr, acc, tpd)
            print(f"  qc={qc:5} qr={qr:3}: acc={acc:5.1f}% tpd={tpd:5.1f} n={n:6}", flush=True)

    qc_star, qr_star, acc_mv, tpd_mv = best
    print(f"\n锁最优: qc={qc_star} qr={qr_star} (meta_val acc={acc_mv:.2f}% tpd={tpd_mv:.1f})", flush=True)

    # test 仅一次评估
    print("\n--- test 仅一次无前视评估 ---")
    p_te = load_lgb(sym, "test")
    y_te = ctx.y("test")
    sec_te = np.asarray(ctx.times("test")).astype("datetime64[s]").astype(np.int64)
    conf_te = np.abs(p_te - 0.5) * 2
    pred_te = (p_te >= 0.5).astype(np.int8)
    rv_te = load_rvol(sym, "test")
    acc_te, tpd_te, n_te = evaluate(pred_te, conf_te, rv_te, y_te, sec_te, qc_star, qr_star, 30, 30)
    print(f"test acc={acc_te:.2f}% tpd={tpd_te:.1f} n={n_te} | 最优点 qc={qc_star} qr={qr_star}", flush=True)
    print(f"meta_val acc={acc_mv:.2f}% tpd={tpd_mv:.1f}", flush=True)


if __name__ == "__main__":
    for s in ["ETH", "BTC"]:
        scan(s)