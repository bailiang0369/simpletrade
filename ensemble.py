"""集成: 堆叠(meta模型) + 平均投票两种, 基于各基模型的"样本外"概率。
置信度 = max(p, 1-p), 用于1%覆盖率选择。
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from models.base import BaseModel


class Ensemble:
    def __init__(self, models):
        self.models = models          # list of fitted BaseModel
        self.meta = None
        self.fit_time = None

    def collect(self, ctx, split):
        """收集各模型在该切分上的概率 (N, M)。"""
        M = len(self.models)
        rows = ctx.split_rows[split].sum()
        P = np.zeros((rows, M), dtype=np.float32)
        for j, m in enumerate(self.models):
            P[:, j] = m.predict(ctx, split)
        return P

    def meta_features(self, P):
        edges = np.abs(2 * P - 1)
        return np.hstack([P, edges])

    def fit_meta(self, ctx):
        """在 meta_val 上训练堆叠meta(逻辑回归)。"""
        t0 = time.time()
        Pv = self.collect(ctx, "meta_val")
        yv = ctx.y("meta_val")
        Xm = self.meta_features(Pv)
        self.meta = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
        self.meta.fit(Xm, yv)
        self.fit_time = time.time() - t0
        print(f"[ensemble] meta trained on meta_val ({len(yv)} rows) in {self.fit_time:.0f}s", flush=True)
        return self

    def predict_meta(self, ctx, split):
        P = self.collect(ctx, split)
        return self.meta.predict_proba(self.meta_features(P))[:, 1], P

    def predict_vote(self, P):
        return P.mean(axis=1)

    @staticmethod
    def confidence(p):
        return np.maximum(p, 1 - p)
