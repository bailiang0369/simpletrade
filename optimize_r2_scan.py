#!/usr/bin/env python3
"""无泄露 R2(每日滚动阈值) 分位数扫描: 在 meta_val 上合法选择分位数, 应用到 test。

严格样本外:
  - 分位数 P ∈ PS 的选择只在 meta_val 上做每日滚动评估 (历史=meta_val自身前段)。
  - test 上: 历史 = meta_val 尾部 WIN 天 + 已见 test 天, 盘前定当天阈值。
  - 同时评估 solo / joint / solo+joint 二次融合 三种融合方式。

选择规则 (meta_val): bad=0 优先 -> 覆盖率∈[0.8%,1.2%] -> 单月下限最高 -> 总acc最高。
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
PS = [98.0, 98.25, 98.5, 98.75, 99.0, 99.25, 99.5]
WIN_DAYS = 90             # 滚动历史窗口(天)
SAMPLES_PER_DAY = 1440    # 1分钟K线/天


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_pool(symbol, tag, split):
    """返回 (n_models, n_samples) 原始预测矩阵。"""
    Ps = []
    for f in FAMS:
        path = f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy" if tag == "JOINT" \
            else f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy"
        Ps.append(np.load(path))
    return np.concatenate(Ps, axis=0)


def eval_monthly(sel, pred, y, mts, sec):
    n_all = len(sec)
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0.0)
    ps, ys, ms = pred[sel], y[sel], mts[sel]
    uniq = np.unique(ms)
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in uniq}
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    cov = float(len(ps)) / n_all
    return (min_a, nbad, cov, acc_m)


def daily_roll(conf, pred, y, mts, sec, pctl, hist_seed):
    """每日滚动: τ_d = hist(含meta_val尾部+已见天) 的 pctl 分位, 盘前定当天阈值。
    hist_seed: 列表, 作为起始历史。返回 (min_a, nbad, cov, acc_m, coverage_per_day)。
    """
    day = sec // 86400
    days = np.unique(day)
    hist = list(hist_seed)
    sel_idx = []
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), pctl)
        s = np.where(md & (conf >= tau))[0]
        sel_idx.append(s)
        hist.extend(conf[md])
        if len(hist) > WIN_DAYS * SAMPLES_PER_DAY * 2:
            del hist[:len(hist) - WIN_DAYS * SAMPLES_PER_DAY * 2]
    sel = np.concatenate(sel_idx) if sel_idx else np.array([], dtype=np.int64)
    return eval_monthly(sel, pred, y, mts, sec)


def build_ctx_data(symbol, tag, fused=True):
    ctx = AssetContext(symbol, horizon=30)
    data = {}
    for split in ("meta_val", "test"):
        P = load_pool(symbol, tag, split)
        if fused:
            p = rank_mean(P, P.shape[1])
        else:
            p = None
        y = ctx.y(split)
        sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
        mts = sec.astype("datetime64[s]").astype("datetime64[M]")
        pred = (p >= 0.5).astype(np.int8) if p is not None else None
        conf = np.maximum(p, 1 - p) if p is not None else None
        data[split] = dict(P=P, p=p, y=y, sec=sec, mts=mts, pred=pred, conf=conf)
        del P
        gc.collect()
    return ctx, data


def run_scan(symbol, tag, fused=True):
    label = tag or "SOLO"
    ctx, data = build_ctx_data(symbol, tag, fused)
    mv, te = data["meta_val"], data["test"]

    # ---- meta_val 上扫描: 历史 = 自身前段(滚动), 模拟 test 协议 ----
    mv_conf, mv_pred, mv_y, mv_mts, mv_sec = mv["conf"], mv["pred"], mv["y"], mv["mts"], mv["sec"]
    n_mv = len(mv_sec)
    mv_hist_seed = list(mv_conf[:WIN_DAYS * SAMPLES_PER_DAY])  # 前90天作seed, 之后开始选
    mv_sel_mask = np.zeros(n_mv, bool)
    mv_sel_mask[:WIN_DAYS * SAMPLES_PER_DAY] = True  # 前90天不算选择期(无历史)

    results = []
    for pp in PS:
        min_a, nbad, cov, acc_m = daily_roll(mv_conf, mv_pred, mv_y, mv_mts, mv_sec, pp,
                                             list(mv_conf[:WIN_DAYS * SAMPLES_PER_DAY]))
        results.append((pp, min_a, nbad, cov, acc_m))
    # 选择: bad=0 优先, cov∈[0.8,1.2]%, min_a 最高, 再总acc
    good = [r for r in results if r[2] == 0 and 0.008 <= r[3] <= 0.012]
    pool = good if good else [r for r in results if r[2] == 0]
    if not pool:
        pool = results
    pool.sort(key=lambda r: r[3], reverse=True)  # cov 接近上限的先比? 下面重新排序
    # 重新严格排序: (cov∈[0.8,1.2]) -> min_a desc -> acc desc
    def key(r):
        ok_cov = 0.008 <= r[3] <= 0.012
        return (ok_cov, r[1] if r[1] == r[1] else -1)
    pool.sort(key=key, reverse=True)
    p_best = pool[0][0]

    print(f"\n===== {label} {symbol} (R2 每日滚动扫描) =====")
    print("  P  ->  mv: min_month/bad/cov")
    for r in results:
        print(f"  P={r[0]:5.2f}  mv(min={r[1]:.4f}, bad={r[2]}, cov={r[3]:.4f})")
    print(f"  >> 选择 P={p_best}")

    # ---- test 应用: 历史 = meta_val 尾部90天 ----
    mv_tail = list(mv_conf[-WIN_DAYS * SAMPLES_PER_DAY:])
    min_a, nbad, cov, acc_m = daily_roll(te["conf"], te["pred"], te["y"], te["mts"], te["sec"],
                                         p_best, list(mv_tail))
    # 总 acc
    sel = None
    day = te["sec"] // 86400
    days = np.unique(day)
    hist = list(mv_tail)
    sel_list = []
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), p_best)
        s = np.where(md & (te["conf"] >= tau))[0]
        sel_list.append(s)
        hist.extend(te["conf"][md])
        if len(hist) > WIN_DAYS * SAMPLES_PER_DAY * 2:
            del hist[:len(hist) - WIN_DAYS * SAMPLES_PER_DAY * 2]
    sel = np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)
    acc = float((te["pred"][sel] == te["y"][sel]).mean()) if sel.size else np.nan
    n_days = np.unique(day).size
    print(f"  [test P={p_best}] acc={acc:.4f} min_month={min_a:.4f} bad={nbad} cov={cov:.4f} tpd={len(sel)/n_days:.2f}")
    print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
    del ctx
    gc.collect()
    return p_best, acc, min_a, nbad


def main():
    # 1) solo / joint 各自
    for symbol, tag in (("ETH", ""), ("BTC", ""), ("ETH", "JOINT"), ("BTC", "JOINT")):
        run_scan(symbol, tag, fused=True)
    # 2) solo+joint 二次融合 (rank 平均)
    for symbol in ("ETH", "BTC"):
        ctx = AssetContext(symbol, horizon=30)
        for split in ("meta_val", "test"):
            P_s = load_pool(symbol, "", split)
            P_j = load_pool(symbol, "JOINT", split)
            n = P_s.shape[1]
            p = (rank_mean(P_s, n) + rank_mean(P_j, n)) / 2.0
            y = ctx.y(split)
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            # 保存到全局 dict
            _GLOB.setdefault(symbol, {})[split] = dict(
                p=p, y=y, sec=sec, mts=mts,
                pred=(p >= 0.5).astype(np.int8), conf=np.maximum(p, 1 - p))
            del P_s, P_j, p
            gc.collect()
        mv = _GLOB[symbol]["meta_val"]; te = _GLOB[symbol]["test"]
        print(f"\n===== 2ND {symbol} (solo+joint 二次融合) =====")
        mv_conf = mv["conf"]
        results = []
        for pp in PS:
            min_a, nbad, cov, acc_m = daily_roll(mv_conf, mv["pred"], mv["y"], mv["mts"], mv["sec"],
                                                 pp, list(mv_conf[:WIN_DAYS * SAMPLES_PER_DAY]))
            results.append((pp, min_a, nbad, cov, acc_m))
        for r in results:
            print(f"  P={r[0]:5.2f}  mv(min={r[1]:.4f}, bad={r[2]}, cov={r[3]:.4f})")
        good = [r for r in results if r[2] == 0 and 0.008 <= r[3] <= 0.012]
        pool = good if good else [r for r in results if r[2] == 0]
        if not pool:
            pool = results
        pool.sort(key=lambda r: (0.008 <= r[3] <= 0.012, r[1] if r[1] == r[1] else -1), reverse=True)
        p_best = pool[0][0]
        print(f"  >> 选择 P={p_best}")
        mv_tail = list(mv_conf[-WIN_DAYS * SAMPLES_PER_DAY:])
        min_a, nbad, cov, acc_m = daily_roll(te["conf"], te["pred"], te["y"], te["mts"], te["sec"],
                                             p_best, list(mv_tail))
        day = te["sec"] // 86400
        days = np.unique(day)
        hist = list(mv_tail)
        sel_list = []
        for dd in days:
            md = day == dd
            tau = np.percentile(np.asarray(hist), p_best)
            s = np.where(md & (te["conf"] >= tau))[0]
            sel_list.append(s)
            hist.extend(te["conf"][md])
            if len(hist) > WIN_DAYS * SAMPLES_PER_DAY * 2:
                del hist[:len(hist) - WIN_DAYS * SAMPLES_PER_DAY * 2]
        sel = np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)
        acc = float((te["pred"][sel] == te["y"][sel]).mean()) if sel.size else np.nan
        n_days = np.unique(day).size
        print(f"  [test P={p_best}] acc={acc:.4f} min_month={min_a:.4f} bad={nbad} cov={cov:.4f} tpd={len(sel)/n_days:.2f}")
        print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})
        del ctx
        gc.collect()


_GLOB = {}


if __name__ == "__main__":
    main()
