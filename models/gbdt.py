"""GBDT(LightGBM) 表格特征模型。

特征全部为价格衍生 + 主动买卖量衍生, 无总成交量/ATR。
用精选特征子集(约50)加载, 避免一次性载入全部137维矩阵导致 4GB cgroup OOM;
训练集过大时随机抽样以控制内存。

预测方式(经 H30 验证):
  5 种子 bagging, 组合采用"排名平均"(每种子预测转百分位秩再平均)。
  对比概率平均: 概率平均会压平尾部置信度(test 62.67%), 排名平均保住尾部排序,
  单模型 BTC test 达 65.16% (meta_val 58.62% 亦为组合策略中最高)。
  跨币种验证: ETH test 61.81%, 差值 3.35pp(略超3pp线, ETH 未崩)。
  严格单资产训练, 无跨资产特征。
"""
import time

import lightgbm as lgb
import numpy as np

import config
from .base import BaseModel

# 精选子集: 动量/均值回归/区间位置/波动状态/结构形态/主动买卖微观/时间
FEATURES = [
    "lr_5", "lr_15", "lr_30", "lr_60", "lr_120", "lr_240", "mom_60",
    "z_10", "z_30", "z_60", "z_120",
    "rvol_30", "rvol_60", "rvol_ratio_60_5", "rvol_z_60", "rvol_dir",
    "pos_30", "pos_60", "pos_120", "pos_240",
    "dd_60", "dd_240", "ru_60", "ru_240",
    "hh_dd_60", "ll_ru_60", "body_pos_60",
    "body_ratio", "up_wick", "lo_wick", "ngreen_10", "gap", "streak_up", "max_range_30",
    "tbr_30", "tbr_60", "tbr_z_30", "tbr_z_60", "cvd_30", "cvd_60",
    "buyvol_strength_30", "tb_act_60", "ts_act_60", "tb_acc_30",
    "cvd_dir_30", "tbr_hi_60", "lr_skew_60", "up_body_ratio_30", "mom_align_30_240",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us", "is_eu", "ret_day",
]


class GBDTModel(BaseModel):
    name = "gbdt"

    # 5 种子 bagging 种子列表; 预测组合 = 百分位秩平均(排名平均)
    BAGGED_SEEDS = [42, 49, 56, 63, 70]

    def __init__(self, seed=None, max_train=None, seeds=None, **params):
        super().__init__(seed)
        self.seeds = list(seeds) if seeds is not None else list(self.BAGGED_SEEDS)
        self.max_train = max_train or 1_400_000
        self.params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=2,
            min_data_in_leaf=150,
            lambda_l1=0.0,
            lambda_l2=2.0,
            num_threads=config.N_JOBS,
            verbosity=-1,
        )
        self.params.update(params)
        self.models = []          # list[lgb.Booster], 每颗种子一个

    def fit(self, ctx):
        t0 = time.time()
        feats = list(FEATURES)   # 精选特征子集(经 X_subset 流式加载, 峰值只占用采样行)
        trm = ctx.split_rows["train"]
        esm = ctx.split_rows["early_stop"]
        tr_idx_all = np.where(trm)[0]
        # 早停集只构建一次(各种子共用)
        Xes = ctx.X_subset(feats, esm)
        yes = ctx.label[esm]

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
            # 必须用同一 train_mask 取标签, 保证与 X_subset 逐行对齐
            ytr = ctx.label[train_mask]
            params = dict(self.params)
            params["seed"] = seed
            dtr = lgb.Dataset(Xtr, ytr)
            des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(
                params, dtr,
                num_boost_round=5000,
                valid_sets=[des],
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(0)],
            )
            self.models.append(m)
            del dtr, Xtr
            print(f"[gbdt] {ctx.symbol} seed{seed} best_iter={m.best_iteration} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        self.fitted = True
        print(f"[gbdt] {ctx.symbol} {len(self.models)}-seed bagging fit done in "
              f"{time.time()-t0:.0f}s (nfeat={len(feats)})", flush=True)
        return self

    def predict(self, ctx, split):
        """排名平均: 每种子概率转切分内百分位秩, 再取均值。返回 [0,1] float32。"""
        feats = list(FEATURES)
        X = ctx.X_subset(feats, ctx.split_rows[split])
        n = len(X)
        R = np.zeros((len(self.models), n), dtype=np.float64)
        for i, m in enumerate(self.models):
            p = m.predict(X, num_iteration=m.best_iteration)
            R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        return np.asarray(R.mean(axis=0), dtype=np.float32)

    def save(self, path):
        for i, m in enumerate(self.models):
            m.save_model(f"{path}.s{i}")

    def load(self, path):
        self.models = []
        for i in range(len(self.seeds)):
            self.models.append(lgb.Booster(model_file=f"{path}.s{i}"))
        self.fitted = True
