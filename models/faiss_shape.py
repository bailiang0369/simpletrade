"""FAISS 多尺度形态聚类基模型 (以 raw 1分钟 close 为输入, 独立于特征工程)。

思路(来自用户 Colab 实验 faissv6 的参考):
  - 对每个样本取"当前时刻往前 w 分钟的 close 窗口"作为原始价格形态。
  - 用 FAISS KMeans 在 train 段把形态聚类成 K 簇, 并用 train 段符号标签统计每簇的
    Edge = (簇内30分钟上涨占比) - 0.5  (簇的预测优势)。
  - 对新切分: 把每个样本的形态分配到最近簇, 该簇的 Edge 作为该样本的"形态信号"。
  - 多尺度共振: 对 w in [15,30,60], 各尺度独立聚类打分, 最终得分=各尺度Edge之和(加权重)。

与标量特征 GBDT 正交: 这里只"看"原始价格路径的形态, 不手工造截面特征,
是 ensemble 里有价值的低相关第二信号源。

切分纪律/接口遵循 models/base.py:
  - fit 只用 ctx 的 train 切分 (拟合 KMeans 质心 + 簇 Edge)
  - predict(ctx, split) 返回该切分每行的"形态上涨倾向得分" [0,1] 化
  - 无未来函数: 窗口只到当前 raw 位置, Edge 只在 train 段统计
"""
import os
import threading
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

import config

# 形态尺度(分钟)
SCALES_W = [15, 30, 60]
# 每尺度簇数
N_CLUSTERS = {"15": 240, "30": 300, "60": 360}
# train 段用于 KMeans 的形态子样本数
FIT_SUBSAMPLE = 500_000
EDGE_EPS = 1e-9


def _normalize_raw(raw, pos, wm):
    """对 raw 中的 pos 位置构造 wm 长度滑窗并归一化为比例形态。返回 [n,wm] float32。"""
    n = len(pos)
    win = np.empty((n, wm), dtype=np.float64)
    for q in range(wm):
        win[:, q] = raw[pos - (wm - 1 - q)]
    first = win[:, :1].copy()
    first[first == 0] = 1.0
    X = (win / first).astype(np.float32) - 1.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class FaissShape:
    name = "faiss"
    NAME = "faiss"

    def __init__(self, seed=None, scales=SCALES_W, n_clusters=None, fit_subsample=FIT_SUBSAMPLE):
        if not _HAS_FAISS:
            raise RuntimeError("faiss 未安装: pip install faiss-cpu")
        self.seed = seed or config.SEED
        self.scales = scales
        self.n_clusters = n_clusters or N_CLUSTERS
        self.fit_subsample = fit_subsample
        self.kmeans = {}      # w(str) -> faiss.Kmeans
        self.edges = {}       # w(str) -> [K] 簇 Edge
        self.fitted = False

    # ---------- fit: 每尺度 KMeans 聚类 + 簇 Edge ----------
    def fit(self, ctx):
        raw = ctx.c.astype(np.float64)
        ds_raw = ctx.ds_to_raw
        tr_pos = ds_raw[ctx.split_rows["train"]]
        if self.fit_subsample and len(tr_pos) > self.fit_subsample:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(len(tr_pos), self.fit_subsample, replace=False)
            tr_pos = tr_pos[keep]

        for w in self.scales:
            key = str(w)
            wm = int(w)
            pos = tr_pos[tr_pos >= wm - 1]
            if len(pos) == 0:
                continue
            X = _normalize_raw(raw, pos, wm)
            K = self.n_clusters[key]
            kmeans = faiss.Kmeans(wm, K, niter=20, seed=self.seed, verbose=False, gpu=False,
                                  min_points_per_centroid=1)
            kmeans.train(X)
            D, I = kmeans.index.search(X, 1)
            I = I[:, 0]
            # 对应 ds 行标签(未来30min)
            ds_idx = np.searchsorted(ds_raw, pos)
            ds_idx = np.clip(ds_idx, 0, len(ctx.label) - 1)
            lbl = ctx.label[ds_idx].astype(np.float64)
            edges = np.zeros(K, dtype=np.float64)
            cnts = np.zeros(K, dtype=np.float64)
            np.add.at(edges, I, lbl)
            np.add.at(cnts, I, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_lbl = np.where(cnts > 0, edges / (cnts + EDGE_EPS), 0.5)
            self.edges[key] = (mean_lbl - 0.5).astype(np.float32)
            self.kmeans[key] = kmeans
            print(f"[faiss w={wm}] K={K} 训练完成,  |Edge|均值={np.abs(self.edges[key]).mean():.4f}", flush=True)
        self.fitted = True

    # ---------- predict: 分配最近簇 -> Edge 得分 ----------
    def predict(self, ctx, split):
        if not self.fitted:
            raise RuntimeError("fit first")
        raw = ctx.c.astype(np.float64)
        ds_idx = ctx.split_positions(split)
        scores = np.zeros(len(ds_idx), dtype=np.float64)
        n_s = 0
        for w in self.scales:
            key = str(w)
            if key not in self.kmeans:
                continue
            wm = int(w)
            valid = ds_idx >= wm - 1
            if valid.sum() == 0:
                continue
            pos = ds_idx[valid]
            X = _normalize_raw(raw, pos, wm)
            D, I = self.kmeans[key].index.search(X, 1)
            I = I[:, 0]
            acc = np.zeros(len(ds_idx), dtype=np.float64)
            acc[valid] = self.edges[key][I].astype(np.float64)
            scores += acc
            n_s += 1
        if n_s == 0:
            return np.full(len(ds_idx), 0.5, dtype=np.float32)
        scores /= n_s
        return (scores + 0.5).astype(np.float32)