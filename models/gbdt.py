"""GBDT(LightGBM) 表格特征模型。
特征全部为价格衍生 + 主动买卖量衍生, 无总成交量/ATR。训练集过大时随机抽样以控制内存。
"""
import time

import lightgbm as lgb
import numpy as np

import config
from .base import BaseModel


"""GBDT(LightGBM) 表格特征模型。
特征全部为价格衍生 + 主动买卖量衍生, 无总成交量/ATR。训练集过大时随机抽样以控制内存。
用精选特征子集(约50)加载, 避免一次性载入全部137维矩阵导致 4GB cgroup OOM。
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

    def __init__(self, seed=None, max_train=None, **params):
        super().__init__(seed)
        self.max_train = max_train or 1_400_000
        self.params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.02,
            num_leaves=63,
            max_depth=-1,
            feature_fraction=0.5,
            bagging_fraction=0.7,
            bagging_freq=2,
            min_data_in_leaf=200,
            lambda_l1=1.0,
            lambda_l2=6.0,
            num_threads=config.N_JOBS,
            seed=self.seed,
            verbosity=-1,
        )
        self.params.update(params)
        self.model = None

    def fit(self, ctx):
        t0 = time.time()
        feats = list(ctx.feat_names)   # 全特征(经 X_subset 流式加载, 峰值只占用采样行)
        rng = np.random.default_rng(self.seed)
        trm = ctx.split_rows["train"]
        esm = ctx.split_rows["early_stop"]
        tr_idx = np.where(trm)[0]
        if len(tr_idx) > self.max_train:
            keep = rng.choice(len(tr_idx), self.max_train, replace=False)
            tr_idx = tr_idx[keep]
        train_mask = np.zeros_like(trm, dtype=bool)
        train_mask[tr_idx] = True
        Xtr = ctx.X_subset(feats, train_mask)
        ytr = ctx.label[tr_idx]
        Xes = ctx.X_subset(feats, esm)
        yes = ctx.label[esm]
        dtr = lgb.Dataset(Xtr, ytr)
        des = lgb.Dataset(Xes, yes, reference=dtr)
        self.model = lgb.train(
            self.params, dtr,
            num_boost_round=5000,
            valid_sets=[des],
            callbacks=[lgb.early_stopping(200, verbose=False),
                       lgb.log_evaluation(0)],
        )
        self.fitted = True
        print(f"[gbdt] {ctx.symbol} fit done in {time.time()-t0:.0f}s "
              f"(train={len(Xtr)}, es={len(Xes)}, nfeat={len(feats)}, "
              f"best_iter={self.model.best_iteration})", flush=True)
        return self

    def predict(self, ctx, split):
        feats = list(ctx.feat_names)
        X = ctx.X_subset(feats, ctx.split_rows[split])
        p = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.asarray(p, dtype=np.float32)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
        self.fitted = True
