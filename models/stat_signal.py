"""统计信号模型 (StatSignal): 原始K线统计特征 + 逻辑回归, 输出未来上涨概率。

思路: 不依赖 137 列宽特征矩阵, 而是直接从原始 OHLCV/买卖量计算一组"统计信号"
(动量、波动、均值回归、量能微观结构、时间周期), 在 train 上拟合一逻辑回归,
作为集成中廉价、可解释、低相关的基线模型。

约束遵守:
- fit 只用 train 段 (遵守"基模型只用 train 训练"的切分纪律)
- 特征仅用 t 时刻及以前信息, 无未来泄漏
- 滚动统计用 cumsum O(n) 实现, 不整表加载特征矩阵, 内存友好

用法(与 train_one/reproduce_core 接口一致):
    m = StatSignal(seed=config.SEED)
    m.fit(ctx)
    p = m.predict(ctx, "meta_val")   # 或 "test", 返回 float32 概率
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import config


def _roll_mean_sd(x, w):
    """滚动均值/标准差 (窗口 w, 含当前根)。返回与 x 等长, 头部 warmup 为 NaN。O(n)。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    if w <= 1:
        mu[:] = x
        sd[:] = 0.0
        return mu, sd
    if w >= n:
        return mu, sd
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    s = cs[w - 1:] - np.concatenate([[0.0], cs[:n - w]])
    s2 = cs2[w - 1:] - np.concatenate([[0.0], cs2[:n - w]])
    m = s / w
    v = s2 / w - m * m
    v = np.maximum(v, 0.0)
    mu[w - 1:] = m
    sd[w - 1:] = np.sqrt(v)
    return mu, sd


def _roll_mean(x, w):
    return _roll_mean_sd(x, w)[0]


def _log_ret(c, w):
    lr = np.full(len(c), np.nan, dtype=np.float64)
    lr[w:] = np.log(c[w:] / c[:-w])
    return lr


class StatSignal:
    name = "stat"       # 与 BaseModel 子类接口一致(run.py 用 m.name)
    NAME = "stat"

    # 原始K线统计特征 (12) + 时间周期特征 (3)
    RAW_FEATS = [
        "lr_5", "lr_15", "lr_60", "lr_240", "lr_1440",   # 多窗口动量
        "rvol_60", "rvol_1440",                          # 波动率(每分钟收益std)
        "z_60", "z_1440",                                # 收盘价均值回归 z 分
        "tbr_60", "cvd_60", "body_60",                   # 量能微观结构
    ]
    TIME_FEATS = ["hour_sin", "hour_cos", "weekday"]

    def __init__(self, seed=None):
        self.seed = seed if seed is not None else config.SEED
        self.clf = None
        self.mu = None
        self.scale = None
        self.feat_names = self.RAW_FEATS + self.TIME_FEATS
        self._Xc = None          # 缓存整段特征矩阵

    # ---------- 特征 ----------
    def _raw_features(self, ctx):
        c = ctx.c.astype(np.float64)
        o = ctx.o.astype(np.float64)
        h = ctx.h.astype(np.float64)
        l = ctx.l.astype(np.float64)
        tb = ctx.tb.astype(np.float64)
        sv = ctx.vol.astype(np.float64)
        rng = h - l
        rng_safe = np.where(rng > 0, rng, 1.0)

        lr1 = np.zeros_like(c)
        lr1[1:] = np.log(c[1:] / c[:-1])

        cols = {f"lr_{w}": _log_ret(c, w) for w in (5, 15, 60, 240, 1440)}
        _, sd60 = _roll_mean_sd(lr1, 60)
        _, sd1440 = _roll_mean_sd(lr1, 1440)
        cols["rvol_60"] = sd60
        cols["rvol_1440"] = sd1440
        for w in (60, 1440):
            mu_w, sd_w = _roll_mean_sd(c, w)
            safe = np.where(sd_w > 0, sd_w, np.nan)
            cols[f"z_{w}"] = (c - mu_w) / safe
        tot = tb + sv
        tbr = tb / np.where(tot > 0, tot, 1.0)
        cvd = (tb - sv) / np.where(tot > 0, tot, 1.0)
        body = np.abs(c - o) / rng_safe
        cols["tbr_60"] = _roll_mean(tbr, 60)
        cols["cvd_60"] = _roll_mean(cvd, 60)
        cols["body_60"] = _roll_mean(body, 60)

        idx = ctx.ds_to_raw
        return np.column_stack([cols[f][idx] for f in self.RAW_FEATS]).astype(np.float32)

    def _time_features(self, ctx):
        ts = ctx.ds_ts
        h = (ts % 86400) / 3600.0
        wd = (ts // 86400 + 4) % 7          # 1970-01-01 为周四(=4), 0=周一
        return np.column_stack([
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
            wd.astype(np.float32),
        ]).astype(np.float32)

    def _X(self, ctx):
        if self._Xc is None:
            self._Xc = np.hstack([self._raw_features(ctx), self._time_features(ctx)])
        return self._Xc

    # ---------- 训练接口 ----------
    def fit(self, ctx):
        X = self._X(ctx)
        Xtr = X[ctx.split_rows["train"]]
        ytr = ctx.y("train")
        fin = np.isfinite(Xtr).all(axis=1)
        Xtr, ytr = Xtr[fin], ytr[fin]

        self.mu = Xtr.mean(axis=0)
        self.scale = Xtr.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        Xs = (Xtr - self.mu) / self.scale
        self.clf = LogisticRegression(C=0.5, max_iter=1000, random_state=self.seed)
        self.clf.fit(Xs, ytr)
        print(f"[stat] fit: train={len(ytr)} rows, "
              f"p_up={ytr.mean():.3f}, n_feat={len(self.feat_names)}", flush=True)
        return self

    def predict(self, ctx, name):
        X = self._X(ctx)
        Xs = (X[ctx.split_rows[name]] - self.mu) / self.scale
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return self.clf.predict_proba(Xs)[:, 1].astype(np.float32)

    # ---------- 持久化 ----------
    def save(self, path):
        np.savez(path + ".npz",
                 coef=self.clf.coef_.astype(np.float32),
                 intercept=np.float32(self.clf.intercept_[0]),
                 mu=self.mu.astype(np.float32),
                 scale=self.scale.astype(np.float32),
                 feat_names=np.array(self.feat_names, dtype=object))

    @classmethod
    def load(cls, path):
        z = np.load(path + ".npz", allow_pickle=True)
        m = cls()
        m.feat_names = list(z["feat_names"])
        m.mu = z["mu"]
        m.scale = z["scale"]
        m.clf = LogisticRegression()
        m.clf.coef_ = z["coef"].reshape(1, -1)
        m.clf.intercept_ = z["intercept"].reshape(-1)
        m.clf.classes_ = np.array([0, 1])
        return m
