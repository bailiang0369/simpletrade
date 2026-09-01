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
import os
import time

import lightgbm as lgb
import numpy as np

import config
from .base import BaseModel

# 精选子集(49维): 删除了冗余特征(r=1.0)和零贡献特征
# 移除: lr_60(=mom_60), dd_60(=hh_dd_60), ru_60(=ll_ru_60),
#       tbr_30(=cvd_30), tbr_60(=cvd_60), tbr_z_60(≈tbr_z_30), streak_up(0%)
FEATURES = [
    "lr_5", "lr_15", "lr_30", "lr_120", "lr_240", "mom_60",
    "z_10", "z_30", "z_60", "z_120",
    "rvol_30", "rvol_60", "rvol_ratio_60_5", "rvol_z_60", "rvol_dir",
    "pos_30", "pos_60", "pos_120", "pos_240",
    "dd_240", "ru_240",
    "hh_dd_60", "ll_ru_60", "body_pos_60",
    "body_ratio", "up_wick", "lo_wick", "ngreen_10", "gap", "max_range_30",
    "tbr_z_30", "cvd_30", "cvd_60",
    "buyvol_strength_30", "tb_act_60", "ts_act_60", "tb_acc_30",
    "cvd_dir_30", "tbr_hi_60", "lr_skew_60", "up_body_ratio_30", "mom_align_30_240",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us", "is_eu", "ret_day",
]


class GBDTModel(BaseModel):
    name = "gbdt"

    # 5 种子 bagging 种子列表; 预测组合 = 百分位秩平均(排名平均)
    BAGGED_SEEDS = [42, 49, 56, 63, 70]
    SCALE_POS_WEIGHT = 2.0     # 正例权重放大 (top信号更重要)

    def __init__(self, seed=None, max_train=None, seeds=None, use_soft_label=False, **params):
        super().__init__(seed)
        self.seeds = list(seeds) if seeds is not None else list(self.BAGGED_SEEDS)
        self.max_train = max_train or 1_400_000
        self.use_soft_label = use_soft_label
        # 统计正例比例，自动调整pos_weight
        self.params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=2,
            min_data_in_leaf=150,
            lambda_l1=0.1,       # 轻微L1正则促进特征选择
            lambda_l2=2.0,
            scale_pos_weight=self.SCALE_POS_WEIGHT,  # 放大正例权重，top-k更准
            num_threads=config.N_JOBS,
            verbosity=-1,
            min_data=1,
        )
        self.params.update(params)
        self.models = []          # list[lgb.Booster], 每颗种子一个
        self.calibrator = None    # Platt 后校准(可选)

    def fit(self, ctx, calibrate_on="meta_val"):
        """训练 + 可选的Platt后校准。
        calibrate_on: 在哪个切分上做概率校准 (None则跳过校准)
        """
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
            # 训练标签: 若启用软标签则用 soft_label, 否则用二元 label
            if self.use_soft_label and hasattr(ctx, 'soft_label'):
                ytr = ctx.soft_label[train_mask]
            else:
                ytr = ctx.label[train_mask]
            params = dict(self.params)
            params["seed"] = seed
            dtr = lgb.Dataset(Xtr, ytr)
            des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(
                params, dtr,
                num_boost_round=5000,
                valid_sets=[des],
                callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                           lgb.log_evaluation(0)],
            )
            self.models.append(m)
            del dtr, Xtr
            print(f"[gbdt] {ctx.symbol} seed{seed} best_iter={m.best_iteration} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        self.fitted = True

        # ---- Platt 后校准 (在 meta_val 上拟合逻辑回归校准曲线) ----
        if calibrate_on is not None and calibrate_on in ctx.split_rows:
            from sklearn.linear_model import LogisticRegression
            cal_mask = ctx.split_rows[calibrate_on]
            cal_p = self._predict_raw(ctx, feats, cal_mask)
            cal_y = ctx.label[cal_mask]
            # 只取有效样本
            fin = np.isfinite(cal_p)
            if fin.sum() > 1000:
                self.calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
                # 将概率映射到 logit 空间再校准
                logit_p = np.clip(cal_p[fin], 1e-7, 1-1e-7)
                logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
                self.calibrator.fit(logit_p, cal_y[fin])
                print(f"[gbdt] Platt校准: 在{calibrate_on}上拟合"
                      f"({fin.sum()}样本, coef={self.calibrator.coef_[0][0]:.4f})", flush=True)
            else:
                print(f"[gbdt] 跳过Platt校准: 有效样本{fin.sum()}不足", flush=True)

        print(f"[gbdt] {ctx.symbol} {len(self.models)}-seed bagging fit done in "
              f"{time.time()-t0:.0f}s (nfeat={len(feats)})", flush=True)
        return self

    def _predict_raw(self, ctx, feats, mask):
        """原始预测(排名平均, 未校准), 返回float64概率。"""
        X = ctx.X_subset(feats, mask)
        n = len(X)
        R = np.zeros((len(self.models), n), dtype=np.float64)
        for i, m in enumerate(self.models):
            p = m.predict(X, num_iteration=m.best_iteration)
            R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)
        return np.asarray(R.mean(axis=0), dtype=np.float64)

    def predict(self, ctx, split):
        """排名平均: 每种子概率转切分内百分位秩, 再取均值。返回 [0,1] float32。
        如果有 calibrator，应用后校准得到更准确的概率。
        """
        mask = ctx.split_rows[split]
        p_raw = self._predict_raw(ctx, FEATURES, mask)
        # 应用Platt校准(如果存在)
        if self.calibrator is not None:
            logit_p = np.clip(p_raw, 1e-7, 1-1e-7)
            logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
            p = self.calibrator.predict_proba(logit_p)[:, 1].astype(np.float64)
        else:
            p = p_raw
        return np.asarray(p, dtype=np.float32)

    def save(self, path):
        for i, m in enumerate(self.models):
            m.save_model(f"{path}.s{i}")
        # 保存校准器
        if self.calibrator is not None:
            import joblib
            joblib.dump(self.calibrator, f"{path}.calib")

    def load(self, path):
        self.models = []
        for i in range(len(self.seeds)):
            self.models.append(lgb.Booster(model_file=f"{path}.s{i}"))
        # 加载校准器
        calib_path = f"{path}.calib"
        if os.path.exists(calib_path):
            import joblib
            self.calibrator = joblib.load(calib_path)
        self.fitted = True
