"""图形/模式识别模型组:
1) CNNImageModel - 把最近60分钟K线渲染成RGB图像, 用小型CNN做"看图"分类
2) FAISSNN      - 把窗口嵌入向量化, 用FAISS建索引做k近邻模式匹配
3) DTWKNN       - 收盘价路径降采样后用FAISS预筛 + DTW精确距离, 加权投票
全部只使用价格 + 主动买卖量。
"""
import time

import faiss
import numpy as np
import torch
import torch.nn as nn

import config
from dtaidistance import dtw
from .base import BaseModel

torch.set_num_threads(config.N_JOBS)


# ---------------- CNN K线图像模型 ----------------
class SmallCNN(nn.Module):
    def __init__(self, H=32, W=60, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.3), nn.Linear(48, 1))

    def forward(self, x):
        z = self.conv(x).flatten(1)
        return self.head(z).squeeze(-1)


class CNNImageModel(BaseModel):
    name = "cnn_img"

    def __init__(self, seed=None, n_samples=config.CNN_SAMPLES, epochs=config.DL_EPOCHS,
                 H=32, W=None):
        super().__init__(seed)
        self.n_samples = n_samples
        self.epochs = epochs
        self.H = H
        self.W = W or config.LOOKBACK_MIN
        self.net = None

    def fit(self, ctx):
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tr_pos = ctx.split_positions("train")
        tr_y = ctx.y("train")
        es_pos = ctx.split_positions("early_stop")
        es_y = ctx.y("early_stop")
        rng = np.random.default_rng(self.seed)
        n_tr = min(self.n_samples, len(tr_pos))
        idx_tr = rng.choice(len(tr_pos), n_tr, replace=False)
        idx_es = rng.choice(len(es_pos), min(config.CNN_ES_SAMPLES, len(es_pos)), replace=False)

        Xtr = torch.from_numpy(ctx.candle_images(tr_pos[idx_tr], W=self.W, H=self.H))
        ytr = torch.tensor(tr_y[idx_tr], dtype=torch.float32)
        Xes = torch.from_numpy(ctx.candle_images(es_pos[idx_es], W=self.W, H=self.H))
        yes = torch.tensor(es_y[idx_es], dtype=torch.float32)

        self.net = SmallCNN(H=self.H, W=self.W)
        opt = torch.optim.Adam(self.net.parameters(), lr=1.5e-3, weight_decay=1e-5)
        crit = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(Xtr, ytr)
        dl = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
        best_acc, best_state = 0.0, None
        for ep in range(self.epochs):
            self.net.train()
            tot, corr = 0, 0
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(self.net(xb), yb)
                loss.backward()
                opt.step()
                pred = (torch.sigmoid(self.net(xb)) > 0.5).float()
                corr += (pred == yb).sum().item()
                tot += len(yb)
            self.net.eval()
            with torch.no_grad():
                val_acc = 0.0
                for s in range(0, len(Xes), config.BATCH_SIZE):
                    xb, yb = Xes[s:s + config.BATCH_SIZE], yes[s:s + config.BATCH_SIZE]
                    pred = (torch.sigmoid(self.net(xb)) > 0.5).float()
                    val_acc += (pred == yb).sum().item()
                val_acc /= len(Xes)
            print(f"[cnn_img] ep{ep} train_acc={corr/tot:.4f} val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fitted = True
        # 显式释放大张量(4GB cgroup): Xtr/Xes 各占几百MB
        del Xtr, ytr, Xes, yes, ds, dl, best_state
        import gc
        gc.collect()
        print(f"[cnn_img] {ctx.symbol} fit done {time.time()-t0:.0f}s, best_val_acc={best_acc:.4f}", flush=True)
        return self

    def predict(self, ctx, split):
        pos = ctx.split_positions(split)
        self.net.eval()
        out = np.empty(len(pos), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(pos), 2048):
                e = min(s + 2048, len(pos))
                X = torch.from_numpy(ctx.candle_images(pos[s:e], W=self.W, H=self.H))
                out[s:e] = torch.sigmoid(self.net(X)).numpy()
        return out

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, ctx, path, C=3):
        self.net = SmallCNN(H=self.H, W=self.W)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.fitted = True


# ---------------- FAISS kNN 模式匹配 ----------------
class FAISSNN(BaseModel):
    name = "faiss"

    def __init__(self, seed=None, templates=150_000, k=101, nlist=128, nprobe=24, W=None):
        super().__init__(seed)
        self.templates = templates
        self.k = k
        self.nlist = nlist
        self.nprobe = nprobe
        self.W = W or config.LOOKBACK_MIN
        self.index = None
        self.tpl_y = None

    def fit(self, ctx):
        t0 = time.time()
        tr_pos = ctx.split_positions("train")
        tr_y = ctx.y("train")
        rng = np.random.default_rng(self.seed)
        n = min(self.templates, len(tr_pos))
        idx = rng.choice(len(tr_pos), n, replace=False)
        emb = ctx.embed_vectors(tr_pos[idx], W=self.W)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        self.tpl_y = tr_y[idx].astype(np.float32)
        quant = faiss.IndexFlatIP(emb.shape[1])
        self.index = faiss.IndexIVFFlat(quant, emb.shape[1], self.nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        self.index.train(emb)
        self.index.add(emb)
        self.index.nprobe = self.nprobe
        self.fitted = True
        print(f"[faiss] {ctx.symbol} index {len(emb)} templates built in {time.time()-t0:.0f}s", flush=True)
        return self

    def predict(self, ctx, split):
        pos = ctx.split_positions(split)
        kk = min(self.k, len(self.tpl_y))
        out = np.empty(len(pos), dtype=np.float32)
        chunk = 200_000
        for s in range(0, len(pos), chunk):
            e = min(s + chunk, len(pos))
            emb = ctx.embed_vectors(pos[s:e], W=self.W)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            D, I = self.index.search(emb, kk)
            lab = self.tpl_y[I]
            out[s:e] = lab.mean(axis=1)
        return out

    def save(self, path):
        faiss.write_index(self.index, path + ".idx")
        np.save(path + "_y.npy", self.tpl_y)

    def load(self, ctx, path):
        self.index = faiss.read_index(path + ".idx")
        self.tpl_y = np.load(path + "_y.npy")
        self.fitted = True


# ---------------- 多尺度时间归一化图形匹配 (变长图形核心方法) ----------------
class TimeNormKNN(BaseModel):
    """多尺度时间归一化图形匹配 —— 回答"变长图形如何匹配"。

    用户问题: 一段上升趋势, 一个用30根K走完, 一个用40根K走完, 内部转折/幅度
    缩放后一致, 但固定窗口 FAISS/DTW 匹配不到。

    解法:
    1) 时间归一化: 任意长度的路径(30根/40根)线性重采样到固定 N 点,
       使"时间拉伸但形状一致"的图形在固定维空间里变成近邻。
    2) 平移/缩放不变: 每个重采样曲线再做 z-score(去均值/除标准差),
       使幅度/起点不同但形态相同(趋势、双底、三角等)的图形可比较。
    3) 多尺度搜索: 查询端不知道图形真实长度, 因此在多个窗口尺度 Ws 上分别
       重采样+搜索, 跨尺度聚合投票 —— 覆盖各种实际长度, 避免子序列匹配。
    4) FAISS 余弦近邻: 归一化后 L2 正则, 内积=余弦相似度, 加权投票预测。
    """
    name = "timenorm"

    def __init__(self, seed=None, Ws=None, N=None, templates=None, k=None,
                 nlist=None, nprobe=None):
        super().__init__(seed)
        self.Ws = tuple(Ws or config.TN_WINDOWS)
        self.N = N or config.TN_N
        self.templates = templates or config.TN_TEMPLATES
        self.k = k or config.TN_K
        self.nlist = nlist or config.TN_NLIST
        self.nprobe = nprobe or config.TN_NPROBE
        self.indexes = None
        self.tpl_y = None
        self._tabs = {}

    def _table(self, W):
        """长度为 W 的路径重采样到 N 点的线性插值表。"""
        t = np.linspace(0, W - 1, self.N)
        lo = np.floor(t).astype(np.int64)
        hi = np.minimum(lo + 1, W - 1)
        w0 = (1.0 - (t - lo)).astype(np.float32)
        w1 = (t - lo).astype(np.float32)
        return lo, hi, w0, w1

    @staticmethod
    def _logclose(ctx):
        return np.log(np.maximum(ctx.c, 1e-9)).astype(np.float32)

    @staticmethod
    def _windows(logc, positions, W):
        """提取 (M, W) 的 log-close 窗口。positions 为 raw 行号。"""
        positions = np.asarray(positions, dtype=np.int64)
        starts = positions - W + 1
        idx = starts[:, None] + np.arange(W, dtype=np.int64)[None, :]
        return logc[idx].astype(np.float32)

    def _resample(self, win, W):
        lo, hi, w0, w1 = self._tabs[W]
        return win[:, lo] * w0[None, :] + win[:, hi] * w1[None, :]

    @staticmethod
    def _normalize(r):
        mu = r.mean(axis=1, keepdims=True)
        sd = r.std(axis=1, keepdims=True) + 1e-6
        z = (r - mu) / sd
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def fit(self, ctx):
        t0 = time.time()
        logc = self._logclose(ctx)
        self._tabs = {W: self._table(W) for W in self.Ws}
        tr_pos = ctx.split_positions("train")
        tr_y = ctx.y("train")
        rng = np.random.default_rng(self.seed)
        n = min(self.templates, len(tr_pos))
        idx = rng.choice(len(tr_pos), n, replace=False)
        tpos = tr_pos[idx]
        self.tpl_y = tr_y[idx].astype(np.float32)
        self.indexes = {}
        for W in self.Ws:
            win = self._windows(logc, tpos, W)
            z = self._normalize(self._resample(win, W))
            dim = self.N
            quant = faiss.IndexFlatIP(dim)
            ivf = faiss.IndexIVFFlat(quant, dim, min(self.nlist, max(1, n // 100)),
                                     faiss.METRIC_INNER_PRODUCT)
            tr = z[np.random.default_rng(0).choice(len(z), min(20000, len(z)), replace=False)]
            ivf.train(tr)
            ivf.add(z)
            ivf.nprobe = self.nprobe
            self.indexes[W] = ivf
            del win, z
            print(f"[timenorm] {ctx.symbol} scale W={W}: {n} templates built", flush=True)
        self.fitted = True
        print(f"[timenorm] {ctx.symbol} {len(self.Ws)} scales fit in {time.time() - t0:.0f}s", flush=True)
        return self

    def predict(self, ctx, split):
        pos = ctx.split_positions(split)
        logc = self._logclose(ctx)
        kk = min(self.k, len(self.tpl_y))
        out = np.empty(len(pos), dtype=np.float32)
        chunk = 100_000
        nW = len(self.Ws)
        for s in range(0, len(pos), chunk):
            e = min(s + chunk, len(pos))
            p = pos[s:e]
            votes = np.empty((e - s, nW), dtype=np.float32)
            for wi, W in enumerate(self.Ws):
                win = self._windows(logc, p, W)
                z = self._normalize(self._resample(win, W))
                D, I = self.indexes[W].search(z, kk)
                lab = self.tpl_y[I]
                wgt = D ** 4                     # 余弦相似度^4 锐化, 突出强匹配
                denom = wgt.sum(axis=1, keepdims=True) + 1e-8
                votes[:, wi] = (wgt * lab).sum(axis=1) / denom[:, 0]
                del win, z, D, I
            out[s:e] = votes.mean(axis=1)        # 跨尺度等权平均
        return out

    def save(self, path):
        import os
        for W, idx in self.indexes.items():
            faiss.write_index(idx, f"{path}_W{W}.idx")
        np.save(path + "_y.npy", self.tpl_y)

    def load(self, ctx, path):
        self._tabs = {W: self._table(W) for W in self.Ws}
        self.tpl_y = np.load(path + "_y.npy")
        self.indexes = {}
        for W in self.Ws:
            self.indexes[W] = faiss.read_index(f"{path}_W{W}.idx")
        self.fitted = True


# ---------------- DTW kNN 时序相似度 (变长路径) ----------------
class DTWKNN(BaseModel):
    """DTW kNN: 变长时序相似度匹配。

    与 TimeNormKNN 配合解决"时间扭曲":
    - 模板在多个窗口尺度上抽取(变长), 先重采样到固定 T 点做 FAISS 预筛,
      把"形状一致但长度不同"的候选找出来;
    - 再对候选在各自原始长度上用 DTW 精确重排(非线形对齐, 容忍局部快慢变化)。
    """
    name = "dtw"

    def __init__(self, seed=None, templates=config.DTW_TEMPLATES, cand=config.DTW_CAND,
                 Ws=None, T=config.DTW_WINDOW):
        super().__init__(seed)
        self.templates = templates
        self.cand = cand
        self.Ws = tuple(Ws or config.DTW_WINDOWS)
        self.T = T
        self.tpl_paths = {}
        self.tpl_y = None
        self.faiss_idx = {}
        self._tabs = {}

    def _table(self, W):
        t = np.linspace(0, W - 1, self.T)
        lo = np.floor(t).astype(np.int64)
        hi = np.minimum(lo + 1, W - 1)
        w0 = (1.0 - (t - lo)).astype(np.float32)
        w1 = (t - lo).astype(np.float32)
        return lo, hi, w0, w1

    @staticmethod
    def _logclose(ctx):
        return np.log(np.maximum(ctx.c, 1e-9)).astype(np.float32)

    @staticmethod
    def _windows(logc, positions, W):
        positions = np.asarray(positions, dtype=np.int64)
        starts = positions - W + 1
        idx = starts[:, None] + np.arange(W, dtype=np.int64)[None, :]
        return logc[idx].astype(np.float32)

    def _resample(self, win, W):
        lo, hi, w0, w1 = self._tabs[W]
        return win[:, lo] * w0[None, :] + win[:, hi] * w1[None, :]

    def fit(self, ctx):
        t0 = time.time()
        logc = self._logclose(ctx)
        self._tabs = {W: self._table(W) for W in self.Ws}
        tr_pos = ctx.split_positions("train")
        tr_y = ctx.y("train")
        rng = np.random.default_rng(self.seed)
        n = min(self.templates, len(tr_pos))
        idx = rng.choice(len(tr_pos), n, replace=False)
        tpos = tr_pos[idx]
        self.tpl_y = tr_y[idx].astype(np.float32)
        for W in self.Ws:
            win = self._windows(logc, tpos, W)
            r = self._resample(win, W)
            nrm = np.linalg.norm(r, axis=1, keepdims=True) + 1e-8
            self.tpl_paths[W] = win              # 保留原始长度用于DTW
            f = faiss.IndexFlatIP(self.T)
            f.add(r / nrm)
            self.faiss_idx[W] = f
            del r, win
            print(f"[dtw] {ctx.symbol} scale W={W}: {n} templates built", flush=True)
        self.fitted = True
        print(f"[dtw] {ctx.symbol} {len(self.Ws)} scales fit in {time.time() - t0:.0f}s", flush=True)
        return self

    def predict(self, ctx, split):
        pos = ctx.split_positions(split)
        logc = self._logclose(ctx)
        cand = min(self.cand, len(self.tpl_y))
        out = np.empty(len(pos), dtype=np.float32)
        chunk = 10_000
        for s in range(0, len(pos), chunk):
            e = min(s + chunk, len(pos))
            p = pos[s:e]
            for i in range(e - s):
                q = p[i]
                # 每个尺度预筛, 汇总候选
                best = []
                for W in self.Ws:
                    if q < W - 1:
                        continue
                    qwin = logc[q - W + 1:q + 1].astype(np.float32)
                    qr = self._resample(qwin[None, :], W)[0]
                    qr_n = qr / (np.linalg.norm(qr) + 1e-8)
                    _, I = self.faiss_idx[W].search(qr_n[None, :], cand)
                    for t in I[0]:
                        best.append((t, W))
                if not best:
                    out[s + i] = 0.5
                    continue
                # DTW 精确距离(在各自原始长度上, 变长由DTW自然处理)
                ds = np.empty(len(best), dtype=np.float64)
                for j, (t, W) in enumerate(best):
                    tpath = self.tpl_paths[W][t].astype(np.float64)
                    # 查询窗口用与模板相同长度以公平比较: 用模板长度Wt
                    Wt = len(tpath)
                    if q < Wt - 1:
                        ds[j] = 1e6
                        continue
                    qwin = logc[q - Wt + 1:q + 1].astype(np.float64)
                    ds[j] = dtw.distance_fast(qwin, tpath, use_pruning=True)
                w = 1.0 / (ds + 1e-3)
                out[s + i] = np.sum(w * self.tpl_y[[b[0] for b in best]]) / np.sum(w)
        return out

    def save(self, path):
        for W, arr in self.tpl_paths.items():
            np.save(f"{path}_paths_W{W}.npy", arr)
        np.save(path + "_y.npy", self.tpl_y)

    def load(self, ctx, path):
        self._tabs = {W: self._table(W) for W in self.Ws}
        self.tpl_y = np.load(path + "_y.npy")
        self.tpl_paths = {}
        self.faiss_idx = {}
        for W in self.Ws:
            arr = np.load(f"{path}_paths_W{W}.npy")
            self.tpl_paths[W] = arr
            r = self._resample(arr, W)
            nrm = np.linalg.norm(r, axis=1, keepdims=True) + 1e-8
            f = faiss.IndexFlatIP(self.T)
            f.add(r / nrm)
            self.faiss_idx[W] = f
        self.fitted = True
