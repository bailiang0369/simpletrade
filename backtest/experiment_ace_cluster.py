#!/usr/bin/env python3
"""移植用户 Colab 的 ACE FAISS 聚类-edge 方案到本项目口径，做同类可比实验。

数据口径(与 JOINT pool20 基线完全一致):
- 目标: 本项目 label(未来30根1min收盘>锚价, 方向平衡), 见 data_store.y()
- 样本: ds 的 train/meta_val/test 严格因果切分(config.SPLITS)
- 覆盖: 固定 Top1%, 用 R2 逐日滚动阈值(盘前历史99分位)选信号, 与 show_monthly.py 同协议
- 聚类中心只在 train 拟合; edge 表只在 train 统计; mv/test 仅查表, 完全无泄露
- 不做 notebook 的"test 全局 top X%"选择(那是用户已否掉的 lookahead 泄露)

信号生成:
- 对每个 ds 样本, 在其 raw close / 预计算的 stoch %K 序列上取多尺度窗口
  (10/15/30/60/120, 含"末端强化"重复末几根), 各自 FAISS KMeans 落簇
- 每簇在 train 统计 edge = (簇内 target 均值 - 0.5); 样本 ace_score = 各尺度 edge 之和
- pred = ace_score>0; conf = |ace_score|; R2 每日取 conf 的99分位做当天阈值

用法:
  python experiment_ace_cluster.py            # 跑 ETH + BTC
  python experiment_ace_cluster.py ETH        # 只跑 ETH
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc, argparse
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
P99 = 99.0
WIN_DAYS = 90
SAMP = 1440

# ---- 与 Colab 一致的 多尺度聚类配置 ----
HYBRID_CONFIG = {
    "v_short_10": {"window": 10, "n_clusters": 100, "repeat": 0},
    "short_15":   {"window": 15, "n_clusters": 120, "repeat": 10},
    "short_30":   {"window": 30, "n_clusters": 150, "repeat": 15},
    "mid_60":     {"window": 60, "n_clusters": 300, "repeat": 15},
    "long_120":   {"window": 120, "n_clusters": 600, "repeat": 15},
}
END_K = 3                 # 末端强化取"末几根"
NITER = 25                # KMeans 迭代(CPU 上折中)
FIT_SUBSAMPLE = 400_000   # 为控内存/时间, 中心只在 train 的这一个子样本上拟合(仍为 train 内)
SEED = 42


def build_window_feats(series, pos, w, r_count, is_indicator, chunk=200_000):
    """对 pos 指定的每行, 截取 series 上以该行为末端的 w 根窗口, 归一化+末端强化。
    返回 (m, dims) float32; 位置不足 w-1 时返回整行 NaN 由调用方置 -1。"""
    m = len(pos)
    dims = w + r_count * END_K
    out = np.full((m, dims), np.nan, dtype=np.float32)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        p = pos[s:e]
        idx = p >= w - 1
        if not idx.all():
            p_ok = p[idx]
        else:
            p_ok = p
        if len(p_ok) == 0:
            continue
        # 窗口视图
        shape = (len(p_ok), w)
        strides = (series.strides[0], series.strides[0])
        # 手动取窗口(避免 as_strided 越界于 pos 对应起始)
        start = p_ok - (w - 1)
        # 用次条: 直接 squeeze 出 (len, w) 的读取
        win = np.lib.stride_tricks.as_strided(series, shape=(len(series) - w + 1, w),
                                              strides=(series.strides[0], series.strides[0]))
        windows = win[start]  # (len_ok, w)
        if is_indicator:
            norm = windows / 100.0
        else:
            w_min = windows.min(1, keepdims=True)
            w_max = windows.max(1, keepdims=True)
            denom = np.where(w_max - w_min == 0, 1e-9, w_max - w_min)
            norm = (windows - w_min) / denom
        if r_count > 0:
            feat = np.hstack([norm] + [norm[:, -END_K:]] * r_count)
        else:
            feat = norm
        if idx.all():
            out[s:e] = feat
        else:
            out[s:e][idx] = feat
        del win, windows, norm, feat
        gc.collect()
    return out


def sloppy_free(a):
    del a
    gc.collect()


def compute_stoch_k(high, low, close, period=29, smooth=3):
    """标准 Stochastic %K(周期 period, 再对 K 做 smooth 期 SMA)。"""
    import pandas as pd
    hh = pd.Series(high).rolling(period).max().to_numpy()
    ll = pd.Series(low).rolling(period).min().to_numpy()
    rng = np.where(hh - ll == 0, np.nan, hh - ll)
    k = 100.0 * (close - ll) / rng
    if smooth > 1:
        k = pd.Series(k).rolling(smooth).mean().to_numpy()
    return np.nan_to_num(k.astype(np.float32), nan=50.0)


def _fit(cluster_feat, n_clusters):
    d = cluster_feat.shape[1]
    km = faiss.Kmeans(d, n_clusters, niter=NITER, seed=SEED, verbose=False)
    km.train(cluster_feat)
    return km.index


def cluster_ids_for_scale(index, pos, w, r_count, is_indicator, series):
    feats = build_window_feats(series, pos, w, r_count, is_indicator)
    m = feats.shape[0]
    ids = np.full(m, -1, dtype=np.int32)
    valid = ~np.isnan(feats[:, 0])
    if valid.any():
        _, q = index.search(np.ascontiguousarray(feats[valid], dtype="float32"), 1)
        ids[valid] = q[:, 0]
    sloppy_free(feats)
    return ids


# ---------- 基线: 读取 JOINT pool20 融合概率 ----------
def load_fused_rank(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    n = P.shape[1]
    R = np.stack([np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1) for i in range(P.shape[0])], axis=0)
    return R.mean(axis=0)


def run_symbol(symbol):
    print(f"\n############ {symbol} ############", flush=True)
    ctx = AssetContext(symbol, horizon=30)
    train_mask = ctx.split_rows["train"]
    train_pos = ctx.ds_to_raw[train_mask]
    y_train = ctx.y("train")

    # 预计算 stoch %K (raw 全序列)
    stoch_k = compute_stoch_k(ctx.h, ctx.l, ctx.c, period=29, smooth=3)
    close = ctx.c

    # ---- 1. 拟合聚类中心(仅 train 内的子样本) + 统计每簇 edge(全部 train) ----
    rng = np.random.default_rng(SEED)
    fit_sel = rng.choice(len(train_pos), size=min(FIT_SUBSAMPLE, len(train_pos)), replace=False)
    scales_model = {}        # name -> (index, w, rc, is_indicator, series, edge_cum, edge_cnt)
    # 价格尺度
    for name, cfg in HYBRID_CONFIG.items():
        w, nc, rc = cfg["window"], cfg["n_clusters"], cfg["repeat"]
        print(f"  [价格] 拟合聚类 {name} (w={w}, k={nc})", flush=True)
        sub = build_window_feats(close, train_pos[fit_sel], w, rc, is_indicator=False)
        # 去掉不足窗口行
        ok = ~np.isnan(sub[:, 0])
        index = _fit(np.ascontiguousarray(sub[ok], dtype="float32"), nc)
        # 统计 edge: 全 train
        ids = cluster_ids_for_scale(index, train_pos, w, rc, is_indicator=False, series=close)
        valid = ids >= 0
        sum_cnt = np.bincount(ids[valid], minlength=nc).astype(np.float64)
        sum_y = np.bincount(ids[valid], weights=y_train[valid], minlength=nc).astype(np.float64)
        edge = np.where(sum_cnt > 0, sum_y / np.maximum(sum_cnt, 1) - 0.5, 0.0).astype(np.float32)
        sloppy_free(sub)
        # 重新以全部点位得到 mv/test 会再查表, 这里把 index 存下复用, 记录元数据
        scales_model[f"p_{name}"] = dict(index=index, w=w, rc=rc, is_ind=False, series=close, edge=edge)
        sloppy_free(ids)
        gc.collect()
    # 指标尺度(stoch)
    for name, cfg in HYBRID_CONFIG.items():
        w, nc, rc = cfg["window"], cfg["n_clusters"], cfg["repeat"]
        print(f"  [stoch] 拟合聚类 {name} (w={w}, k={nc})", flush=True)
        sub = build_window_feats(stoch_k, train_pos[fit_sel], w, rc, is_indicator=True)
        ok = ~np.isnan(sub[:, 0])
        index = _fit(np.ascontiguousarray(sub[ok], dtype="float32"), nc)
        ids = cluster_ids_for_scale(index, train_pos, w, rc, is_indicator=True, series=stoch_k)
        valid = ids >= 0
        sum_cnt = np.bincount(ids[valid], minlength=nc).astype(np.float64)
        sum_y = np.bincount(ids[valid], weights=y_train[valid], minlength=nc).astype(np.float64)
        edge = np.where(sum_cnt > 0, sum_y / np.maximum(sum_cnt, 1) - 0.5, 0.0).astype(np.float32)
        sloppy_free(sub)
        scales_model[f"s_{name}"] = dict(index=index, w=w, rc=rc, is_ind=True, series=stoch_k, edge=edge)
        sloppy_free(ids)
        gc.collect()
    sloppy_free(stoch_k)

    # ---- 2. 逐 split 打分 ----
    data = {}
    for split in ("meta_val", "test"):
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        score = np.zeros(len(pos), dtype=np.float32)
        for name, sm in scales_model.items():
            ids = cluster_ids_for_scale(sm["index"], pos, sm["w"], sm["rc"],
                                        sm["is_ind"], series=sm["series"])
            valid = ids >= 0
            contrib = np.zeros(len(pos), dtype=np.float32)
            contrib[valid] = sm["edge"][ids[valid]]
            score += contrib
            sloppy_free(ids)
            gc.collect()
        sec = ctx.ds_ts[ctx.split_rows[split]].astype(np.int64)
        data[split] = dict(conf=np.abs(score), pred=(score > 0).astype(np.int8),
                           sec=sec, y=ctx.y(split), score=score)
        sloppy_free(score)
        gc.collect()

    mv, te = data["meta_val"], data["test"]
    # ---- 3. R2 逐日滚动阈值, 与 show_monthly.py 同协议 ----
    seed_conf = list(mv["conf"][-WIN_DAYS * SAMP:])
    day = te["sec"] // 86400
    days = np.unique(day)
    keep = np.zeros(len(te["sec"]), bool)
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(seed_conf), P99)
        keep[md & (te["conf"] >= tau)] = True
        seed_conf.extend(te["conf"][md])
        if len(seed_conf) > WIN_DAYS * SAMP * 2:
            del seed_conf[:len(seed_conf) - WIN_DAYS * SAMP * 2]
    sel = np.where(keep)[0]
    ps, ys, secs = te["pred"][sel], te["y"][sel], te["sec"][sel]
    mts = secs.astype("datetime64[s]").astype("datetime64[M]")
    n_days = days.size

    # ---- 4. 输出 ----
    print(f"\n===== {symbol}  ACE聚类edge  R2每日滚动阈值(P99), 总选中{len(sel)}/日均{len(sel)/n_days:.2f} =====")
    print(f"{'月份':<10}{'信号数':>8}{'准确率':>8}{'是否<55':>8}")
    for u in np.unique(mts):
        m = mts == u
        n = int(m.sum())
        acc = float((ps == ys)[m].mean())
        flag = "  <-- 坏月" if acc < 0.55 else ""
        print(f"{str(u)[:7]:<10}{n:>8}{acc:>8.4f}{flag:>8}")
    acc_all = float((ps == ys).mean())
    print(f"{'总 acc':<10}{'':>8}{acc_all:>8.4f}")

    # ---- 5. JOINT pool20 基线(同协议, 供对比) ----
    bl = {}
    for split in ("meta_val", "test"):
        p = load_fused_rank(symbol, split)
        y = ctx.y(split)
        sec = ctx.ds_ts[ctx.split_rows[split]].astype(np.int64)
        bl[split] = dict(conf=np.maximum(p, 1 - p), pred=(p >= 0.5).astype(np.int8),
                         sec=sec, y=y)
        sloppy_free(p)
        gc.collect()
    seedc = list(bl["meta_val"]["conf"][-WIN_DAYS * SAMP:])
    d2 = bl["test"]["sec"] // 86400
    keep2 = np.zeros(len(bl["test"]["sec"]), bool)
    for dd in np.unique(d2):
        md = d2 == dd
        tau = np.percentile(np.asarray(seedc), P99)
        keep2[md & (bl["test"]["conf"] >= tau)] = True
        seedc.extend(bl["test"]["conf"][md])
        if len(seedc) > WIN_DAYS * SAMP * 2:
            del seedc[:len(seedc) - WIN_DAYS * SAMP * 2]
    s2 = np.where(keep2)[0]
    p2 = bl["test"]["pred"][s2]; y2 = bl["test"]["y"][s2]
    m2 = (bl["test"]["sec"][s2].astype("datetime64[s]").astype("datetime64[M]"))
    print(f"\n===== {symbol}  JOINT pool20 基线  同协议, 总选中{len(s2)}/日均{len(s2)/len(np.unique(d2)):.2f} =====")
    print(f"{'月份':<10}{'信号数':>8}{'准确率':>8}{'是否<55':>8}")
    for u in np.unique(m2):
        m = m2 == u
        n = int(m.sum()); acc = float((p2 == y2)[m].mean())
        flag = "  <-- 坏月" if acc < 0.55 else ""
        print(f"{str(u)[:7]:<10}{n:>8}{acc:>8.4f}{flag:>8}")
    acc_b = float((p2 == y2).mean())
    print(f"{'总 acc':<10}{'':>8}{acc_b:>8.4f}")
    print(f"\n  [对比] ACE 总{acc_all:.4f}/最差{min(float((ps==ys)[m].mean()) for m in (mts==u for u in np.unique(mts))):.4f}   "
          f"JOINT 总{acc_b:.4f}/最差{min(float((p2==y2)[m].mean()) for m in (m2==u for u in np.unique(m2))):.4f}")
    del ctx
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    syms = [a.symbol] if a.symbol else a.symbols.split(",")
    for s in syms:
        run_symbol(s)


if __name__ == "__main__":
    main()