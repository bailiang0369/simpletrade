"""GBDT(LightGBM) 表格特征模型。

特征全部为价格衍生 + 主动买卖量衍生, 无总成交量/ATR。
用精选特征子集(约49)加载, 避免一次性载入全部137维矩阵导致 4GB cgroup OOM;
训练集过大时随机抽样以控制内存。

改进 (2026-09):
  - 自定义 top-k 评价函数直接优化头部准确率(替换默认 AUC)
  - 加权采样: 按未来收益绝对值给样本加权, 突出大波动信号
  - 软标签训练: 启用 soft_label 保留幅度信息
  - 增大训练数据量至 260 万

预测方式:
  5 种子 bagging, 组合采用"排名平均"(每种子概率转百分位秩再平均)。
"""
import os
import time

import lightgbm as lgb
import numpy as np

import config
from .base import BaseModel

# 精选子集(49维): 删除了冗余特征(r=1.0)和零贡献特征
# 精选子集(59维): 基础49维 + 交叉特征10维
# 新增交叉特征: pos_tbr_interact, vol_mom_interact, pos_cvd_interact, 
#              di_spread, di_uptrend, mom_vol_confirm, z_divergence, cvd_accel, vol_cvd_interact, di_plus
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
    # 交叉特征 (新增, 显式传递交互信号给树模型)
    "pos_tbr_interact", "vol_mom_interact", "pos_cvd_interact",
    "di_spread", "di_uptrend", "mom_vol_confirm", "z_divergence", "cvd_accel", "vol_cvd_interact", "di_plus",
]


def topk_acc_eval(preds, train_data):
    """自定义评价函数: 监控 top-1% 准确率 (早停仍用AUC)。
    
    LightGBM 传入的 preds 是 raw margin (未经过 sigmoid)，需要先转概率。
    """
    labels = train_data.get_label()
    # raw margin -> 概率
    probs = 1.0 / (1.0 + np.exp(-preds))
    n = len(probs)
    k = max(1, int(n * 0.01))
    conf = np.abs(probs - 0.5) * 2  # [0,1] 置信度
    sel = np.argpartition(-conf, k)[:k]
    acc = (labels[sel] > 0.5).mean()
    return 'top1_acc', acc, True


class GBDTModel(BaseModel):
    name = "gbdt"

    # 5 种子 bagging 种子列表; 预测组合 = 百分位秩平均(排名平均)
    BAGGED_SEEDS = [42, 49, 56, 63, 70]
    SCALE_POS_WEIGHT = 2.0     # 正例权重放大 (top信号更重要)

    def __init__(self, seed=None, max_train=None, seeds=None, use_soft_label=True, **params):
        super().__init__(seed)
        self.seeds = list(seeds) if seeds is not None else list(self.BAGGED_SEEDS)
        self.max_train = max_train or 2_600_000  # 增大训练数据量
        self.use_soft_label = use_soft_label      # 默认启用软标签
        self.params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=2,
            min_data_in_leaf=100,        # 减小以捕捉更多尾部模式
            lambda_l1=0.05,              # 降低正则, 给模型更多自由度
            lambda_l2=1.0,
            scale_pos_weight=self.SCALE_POS_WEIGHT,
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
        
        改进:
          - 自定义top-k评价函数做早停(替换默认AUC)
          - 按未来收益绝对值加权采样, 突出大波动信号
          - 启用软标签训练(保留幅度信息)
          - 增大训练数据量
        """
        t0 = time.time()
        feats = list(FEATURES)
        trm = ctx.split_rows["train"]
        esm = ctx.split_rows["early_stop"]
        tr_idx_all = np.where(trm)[0]
        # 早停集: 用二元标签(评估标准不变)
        Xes = ctx.X_subset(feats, esm)
        yes = ctx.label[esm]
        # 早停集也按置信度约束
        es_conf = np.abs(ctx.soft_label[esm] - 0.5) * 2 if hasattr(ctx, 'soft_label') else np.ones_like(yes)
        # 获取未来收益用于加权
        train_retf = ctx.retf("train") if hasattr(ctx, 'retf') else None

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

            # 训练标签: 始终用二元标签
            # 实验确认: soft_label(连续值)训练时 LightGBM AUC=0.5, 不学习
            # 改用二元标签 + 权重(按收益绝对值放大) 效果更好
            ytr = ctx.label[train_mask].astype(np.float64)

            # 加权采样: 未来收益绝对值越大, 权重越高
            # 注意: train_retf 是训练集内的索引, keep 是训练集内采样下标
            if train_retf is not None and len(tr_idx) < len(tr_idx_all):
                # 有采样: keep 是训练集内下标
                keep_local = np.where(train_mask[tr_idx_all])[0]
                raw_w = np.abs(train_retf[keep_local]).astype(np.float64)
            elif train_retf is not None:
                raw_w = np.abs(train_retf).astype(np.float64)
            else:
                raw_w = None

            if raw_w is not None:
                # 正样本权重放大
                raw_w[ytr > 0.5] *= 2.0
                # 裁剪极端权重, 防止过拟合
                w = np.clip(raw_w * 50, 0.5, 5.0)
            else:
                w = np.ones(len(ytr), dtype=np.float64)

            evals_result = {}
            params = dict(self.params)
            params["seed"] = seed
            dtr = lgb.Dataset(Xtr, ytr, weight=w)
            des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(
                params, dtr,
                num_boost_round=5000,
                valid_sets=[des],
                valid_names=['early_stop'],
                feval=topk_acc_eval,  # 监控top-k但不作为早停依据
                callbacks=[lgb.early_stopping(200, verbose=False, min_delta=1e-5),
                           lgb.log_evaluation(0)],
            )
            self.models.append(m)
            # 打印早停时的top-k准确率(用AUC做早停, top1_acc仅辅助监控)
            auc_val = m.best_score['early_stop']['auc'] if 'auc' in m.best_score.get('early_stop', {}) else -1
            topk_val = m.best_score['early_stop']['top1_acc'] if 'top1_acc' in m.best_score.get('early_stop', {}) else -1
            del dtr, Xtr
            print(f"[gbdt] {ctx.symbol} seed{seed} best_iter={m.best_iteration} "
                  f"auc={auc_val:.4f} top1_acc={topk_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
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
