"""统一数据访问层: 加载某交易对的 原始K线 + 特征数据集, 提供时间切分、样本位置映射。

内存高效(4GB cgroup环境):
- 用 pyarrow 按列逐个读取(不整表加载), 边读边写进预分配的 float32 矩阵, 峰值只占1列
- 全程 float32(价格/特征) + int64/int8(时间戳/标签), 不用 float64
- 切分掩码直接用 int64 时间戳比较, 不构造 pandas DatetimeIndex
- raw 只保留 numpy 数组(O/H/L/C/主动买/总量/ts)
"""
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

# 时间切分边界 -> epoch 秒(避免 pandas 开销)
_SPLIT_EPOCH = {
    k: (int(datetime.fromisoformat(a).replace(tzinfo=timezone.utc).timestamp()),
        int(datetime.fromisoformat(b).replace(tzinfo=timezone.utc).timestamp()))
    for k, (a, b) in config.SPLITS.items()
}


class AssetContext:
    def __init__(self, symbol, horizon=None):
        self.symbol = symbol
        self.horizon = horizon or config.HORIZON_MIN
        # ---- 原始K线: 按列读成 numpy (float32) ----
        raw_p = f"{config.DS_DIR}/raw_{symbol}.parquet"
        self.o = pq.read_table(raw_p, columns=["open"])["open"].to_numpy().astype(np.float32)
        self.h = pq.read_table(raw_p, columns=["high"])["high"].to_numpy().astype(np.float32)
        self.l = pq.read_table(raw_p, columns=["low"])["low"].to_numpy().astype(np.float32)
        self.c = pq.read_table(raw_p, columns=["close"])["close"].to_numpy().astype(np.float32)
        self.tb = pq.read_table(raw_p, columns=["taker_buy_base"])["taker_buy_base"].to_numpy().astype(np.float32)
        self.vol = pq.read_table(raw_p, columns=["volume"])["volume"].to_numpy().astype(np.float32)
        self.raw_ts = pq.read_table(raw_p, columns=["ts"])["ts"].to_numpy().astype(np.int64)
        gc.collect()

        # ---- 特征数据集: 按列逐个读入预分配矩阵(峰值=1列) ----
        ds_p = f"{config.DS_DIR}/ds_{symbol}.parquet"
        cols = pq.read_schema(ds_p).names
        self.feat_names = [c for c in cols if c not in ("label", "ret_future", "ts")]
        self.ds_ts = pq.read_table(ds_p, columns=["ts"])["ts"].to_numpy().astype(np.int64)  # 秒
        self.label = pq.read_table(ds_p, columns=["label"])["label"].to_numpy().astype(np.int8)
        self.ret_future = pq.read_table(ds_p, columns=["ret_future"])["ret_future"].to_numpy().astype(np.float32)
        # Xall 惰性加载: 仅表型模型(GBDT/stat)需要, 序列/图形模型只用 raw 通道,
        # 避免 4GB cgroup 下无条件加载 1.2GB 特征矩阵拖垮 CNN/GRU/FAISS/DTW。
        n, nf = len(self.ds_ts), len(self.feat_names)
        self._Xall = None
        self._Xall_shape = (n, nf)
        gc.collect()

        # ds行 -> raw行 的位置映射
        self.ds_to_raw = np.searchsorted(self.raw_ts, self.ds_ts).astype(np.int64)

        # 切分掩码: 直接比较 epoch 秒
        ts_s = self.ds_ts
        self.split_rows = {}
        for name, (a, b) in _SPLIT_EPOCH.items():
            self.split_rows[name] = (ts_s >= a) & (ts_s <= b)

    # ---------- 切分访问 ----------
    @property
    def Xall(self):
        """惰性加载全部特征矩阵(仅表型模型需要)。"""
        if self._Xall is None:
            n, nf = self._Xall_shape
            ds_p = f"{config.DS_DIR}/ds_{self.symbol}.parquet"
            X = np.empty((n, nf), dtype=np.float32)
            for j, c in enumerate(self.feat_names):
                X[:, j] = pq.read_table(ds_p, columns=[c])[c].to_numpy().astype(np.float32, copy=False)
                if j % 20 == 0:
                    gc.collect()
            self._Xall = X
            gc.collect()
        return self._Xall

    def split_idx(self, name):
        return np.where(self.split_rows[name])[0]

    def split_positions(self, name):
        return self.ds_to_raw[self.split_rows[name]]

    def X(self, name):
        return self.Xall[self.split_rows[name]]

    def X_subset(self, names, mask):
        """只载入指定特征列并返回 mask 对应行的矩阵 (float32)。
        逐列流式读取+索引, 峰值仅为单列+输出, 供表型模型省内存使用。"""
        ds_p = f"{config.DS_DIR}/ds_{self.symbol}.parquet"
        m = np.where(mask)[0]
        out = np.empty((len(m), len(names)), dtype=np.float32)
        for j, nm in enumerate(names):
            col = pq.read_table(ds_p, columns=[nm])[nm].to_numpy().astype(np.float32, copy=False)
            out[:, j] = col[m]
        return out

    def y(self, name):
        return self.label[self.split_rows[name]]

    def retf(self, name):
        return self.ret_future[self.split_rows[name]]

    def times(self, name):
        return pd.to_datetime(self.ds_ts[self.split_rows[name]], unit="s")

    # ---------- 序列/图形模型的原始通道 ----------
    def raw_channels(self):
        """逐分钟通道 (N_raw, C): lr, body_ratio, up_wick, lo_wick, tbr, cvd。"""
        if hasattr(self, "_channels"):
            return self._channels
        o, h, l, c = self.o, self.h, self.l, self.c
        tb = self.tb
        ts = np.clip(self.vol - tb, 0, None)
        rng = h - l
        rng_safe = np.where(rng > 0, rng, 1.0)
        lr = np.zeros_like(c)
        lr[1:] = np.log(c[1:] / c[:-1])
        body = (c - o) / rng_safe
        uw = (h - np.maximum(c, o)) / rng_safe
        lw = (np.minimum(c, o) - l) / rng_safe
        tot = tb + ts
        tbr = tb / np.where(tot > 0, tot, 1.0)
        cvd = (tb - ts) / np.where(tot > 0, tot, 1.0)
        self._channels = np.stack([lr, body, uw, lw, tbr, cvd], axis=1).astype(np.float32)
        return self._channels

    def window_batch(self, positions, W=None, chunk=20000):
        W = W or config.LOOKBACK_MIN
        ch = self.raw_channels()
        C = ch.shape[1]
        out = np.empty((len(positions), W, C), dtype=np.float32)
        for s in range(0, len(positions), chunk):
            e = min(s + chunk, len(positions))
            pos = positions[s:e]
            for i, p in enumerate(pos):
                if p >= W - 1:
                    w = ch[p - W + 1:p + 1]
                    mu = w.mean(axis=0, keepdims=True)
                    sd = w.std(axis=0, keepdims=True) + 1e-6
                    out[s + i] = (w - mu) / sd
        return out

    def candle_images(self, positions, W=None, H=48, chunk=20000):
        W = W or config.LOOKBACK_MIN
        o, h, l, c = self.o, self.h, self.l, self.c
        tb = self.tb
        ts = np.clip(self.vol - tb, 0, None)
        tot = tb + ts
        tbr = tb / np.where(tot > 0, tot, 1.0)
        # 通道优先 (N, 3, H, W): ch0=红(阴线) ch1=绿(阳线) ch2=主动买占比
        out = np.zeros((len(positions), 3, H, W), dtype=np.float32)
        for s in range(0, len(positions), chunk):
            e = min(s + chunk, len(positions))
            pos = positions[s:e]
            for i, p in enumerate(pos):
                if p < W - 1:
                    continue
                seg = slice(p - W + 1, p + 1)
                lo_ = l[seg].min()
                hi_ = h[seg].max()
                span = hi_ - lo_
                if span <= 0:
                    span = 1.0
                oo = o[seg]; cc = c[seg]; hh = h[seg]; ll = l[seg]
                y_o = (H - 1 - (oo - lo_) / span * (H - 1)).astype(int)
                y_c = (H - 1 - (cc - lo_) / span * (H - 1)).astype(int)
                y_h = (H - 1 - (hh - lo_) / span * (H - 1)).astype(int)
                y_l = (H - 1 - (ll - lo_) / span * (H - 1)).astype(int)
                tbw = tbr[seg]
                img = out[s + i]
                for j in range(W):
                    top = min(y_h[j], y_l[j]); bot = max(y_h[j], y_l[j])
                    btop = min(y_o[j], y_c[j]); bbot = max(y_o[j], y_c[j])
                    up = cc[j] >= oo[j]
                    if up:
                        img[1, top:bot + 1, j] = 1.0
                        img[1, btop:bbot + 1, j] = 1.0
                    else:
                        img[0, top:bot + 1, j] = 1.0
                        img[0, btop:bbot + 1, j] = 1.0
                img[2, :, j] = tbw[j]
        return out

    def embed_vectors(self, positions, W=None, chunk=50000):
        W = W or config.LOOKBACK_MIN
        ch = self.raw_channels()
        C = ch.shape[1]
        out = np.empty((len(positions), W * C), dtype=np.float32)
        for s in range(0, len(positions), chunk):
            e = min(s + chunk, len(positions))
            pos = positions[s:e]
            for i, p in enumerate(pos):
                if p >= W - 1:
                    w = ch[p - W + 1:p + 1].reshape(-1)
                    mu = w.mean(); sd = w.std() + 1e-6
                    out[s + i] = (w - mu) / sd
        return out

    def close_path(self, positions, W=None):
        W = W or config.LOOKBACK_MIN
        c = self.c
        out = np.zeros((len(positions), W), dtype=np.float32)
        for i, p in enumerate(positions):
            if p >= W - 1:
                seg = c[p - W + 1:p + 1]
                out[i] = np.log(seg / seg[0])
        return out


def load(symbol):
    return AssetContext(symbol)
