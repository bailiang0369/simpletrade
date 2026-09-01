"""CatBoost 表格模型: 与 LightGBM 完全不同的 Ordered Boosting 算法, 期望产生正交信号。

与 GBDTModel 共享同一精选特征子集, 接口对齐 fit/predict/save/load。
用 5 种子 bagging + 排名平均, 与 GBDT 一致, 便于公平对比。
"""
import time

import numpy as np

import config
from .base import BaseModel
from .gbdt import FEATURES                      # 复用同一特征子集


class CatBoostModel(BaseModel):
    name = "catboost"

    BAGGED_SEEDS = [42, 49, 56, 63, 70]

    def __init__(self, seed=None, max_train=None, seeds=None, **params):
        super().__init__(seed)
        self.seeds = list(seeds) if seeds is not None else list(self.BAGGED_SEEDS)
        self.max_train = max_train or 1_400_000
        self.params = dict(
            loss_function="Logloss",
            eval_metric="Logloss",
            learning_rate=0.03,
            depth=8,                            # 对应 num_leaves≈127
            l2_leaf_reg=3.0,
            random_strength=0.5,
            bagging_temperature=0.7,
            min_data_in_leaf=150,
            thread_count=config.N_JOBS,
            verbose=False,
            allow_writing_files=False,
        )
        self.params.update(params)
        self.models = []                        # list[CatBoost], 每颗种子一个

    def fit(self, ctx):
        from catboost import CatBoost, Pool
        t0 = time.time()
        feats = list(FEATURES)
        trm = ctx.split_rows["train"]
        esm = ctx.split_rows["early_stop"]
        tr_idx_all = np.where(trm)[0]

        Xes = ctx.X_subset(feats, esm)
        yes = ctx.label[esm]
        des_pool = Pool(Xes, yes)

        self.models = []
        for seed in self.seeds:
            rng = np.random.default_rng(seed)
            tr_idx = tr_idx_all
            if len(tr_idx) > self.max_train:
                keep = rng.choice(len(tr_idx), self.max_train, replace=False)
                tr_idx = tr_idx[keep]
            train_mask = np.zeros_like(trm, dtype=bool)
            train_mask[tr_idx] = True
            Xtr = ctx.X_subset(feats, train_mask)
            ytr = ctx.label[train_mask]
            tr_pool = Pool(Xtr, ytr)

            params = dict(self.params)
            params["random_seed"] = seed
            m = CatBoost(params)
            m.fit(
                tr_pool,
                eval_set=des_pool,
                early_stopping_rounds=200,
                verbose_eval=False,
                plot=False,
            )
            self.models.append(m)
            del tr_pool, Xtr
            best_iter = m.get_best_iteration() or len(m)
            print(f"[catboost] {ctx.symbol} seed{seed} best_iter={best_iter} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        self.fitted = True
        print(f"[catboost] {ctx.symbol} {len(self.models)}-seed bagging fit done in "
              f"{time.time()-t0:.0f}s", flush=True)
        return self

    def predict(self, ctx, split):
        """排名平均: 每种子概率转切分内百分位秩, 再取均值。"""
        feats = list(FEATURES)
        X = ctx.X_subset(feats, ctx.split_rows[split])
        n = len(X)
        R = np.zeros((len(self.models), n), dtype=np.float64)
        for i, m in enumerate(self.models):
            p = m.predict(X, prediction_type="Probability")[:, 1]
            R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        return np.asarray(R.mean(axis=0), dtype=np.float32)

    def save(self, path):
        for i, m in enumerate(self.models):
            m.save_model(f"{path}.s{i}")

    def load(self, path):
        from catboost import CatBoost
        self.models = []
        for i in range(len(self.seeds)):
            m = CatBoost()
            m.load_model(f"{path}.s{i}")
            self.models.append(m)
        self.fitted = True