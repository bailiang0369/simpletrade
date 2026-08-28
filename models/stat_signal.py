"""统计/规则信号模型: 可解释的经典信号(均值回归、趋势、突破、主动流、时段偏差)
经逻辑回归 + 保序回归校准为概率。与黑盒模型形成异构。
"""
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .base import BaseModel

FEATURES = [
    "z_30", "z_60", "ema_dev_30", "ema_dev_60", "pos_60", "pos_240",
    "dd_60", "ru_60", "lr_15", "lr_30", "mom_60", "gap",
    "cvd_30", "cvd_60", "tbr_30", "tbr_60", "buyvol_strength_30",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us", "is_eu",
    "rvol_30", "body_ratio", "ngreen_10",
]


class StatSignal(BaseModel):
    name = "stat"

    def __init__(self, seed=None, max_train=1_000_000):
        super().__init__(seed)
        self.max_train = max_train
        self.lr = None
        self.ir = None

    def _X(self, ctx, mask):
        return ctx.X_subset(FEATURES, mask)

    def fit(self, ctx):
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        Xtr = self._X(ctx, ctx.split_rows["train"])
        ytr = ctx.label[ctx.split_rows["train"]]
        if len(Xtr) > self.max_train:
            idx = rng.choice(len(Xtr), self.max_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        self.lr = LogisticRegression(C=0.1, max_iter=800, n_jobs=1)
        self.lr.fit(Xtr, ytr)
        # 保序校准(在 early_stop 上)
        Xes = self._X(ctx, ctx.split_rows["early_stop"])
        yes = ctx.label[ctx.split_rows["early_stop"]]
        p = self.lr.predict_proba(Xes)[:, 1]
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.ir.fit(p, yes)
        self.fitted = True
        print(f"[stat] {ctx.symbol} fit done {time.time()-t0:.0f}s", flush=True)
        return self

    def predict(self, ctx, split):
        X = self._X(ctx, ctx.split_rows[split])
        p = self.lr.predict_proba(X)[:, 1]
        return np.asarray(self.ir.predict(p), dtype=np.float32)

    def save(self, path):
        import joblib
        joblib.dump({"lr": self.lr, "ir": self.ir}, path)

    def load(self, ctx, path):
        import joblib
        d = joblib.load(path)
        self.lr, self.ir = d["lr"], d["ir"]
        self.fitted = True
