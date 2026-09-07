#!/usr/bin/env python3
"""在线池推理 + R2 逐日滚动阈值 (Phase1-步骤4)。

池推理 (PoolPredictor):
  15 booster (5 lgb + 5 xgb + 5 cat) -> 每 booster 概率 -> 各自转百分位秩 -> 均值融合。
  秩有多种模式:
    - "batch": 对整个输入集合转秩(等价离线 load_fused_rank, 供 replay 参照)
    - "session": 对某一 symbol 当天已见样本增量秩(在线实际使用, 计划决策3的工程近似)
滚动阈值 (RollingThreshold):
  每一 UTC 日, 阈值 = 前 90 天置信度 |p-0.5|*2 历史窗口的 99 分位; 保留当日 conf>=tau。
  冷启动: 历史窗口实际天数 < COLD_MIN_DAYS 则当日不发信号(返回 None)。
"""
import bisect

import numpy as np

import config

P99 = 99.0
WIN_DAYS = 90
SAMP = 1440
COLD_MIN_DAYS = 30
MAX_HIST = WIN_DAYS * SAMP * 2          # 与 experiment_seq_model.r2_eval 上限一致


class PoolPredictor:
    """加载 pool_dir 下 15 个 booster, 对输入特征行(batch)输出融合概率。"""

    def __init__(self, pool_dir, seeds=(42, 49, 56, 63, 70), use_seed_rank="batch"):
        self.pool_dir = pool_dir
        self.seeds = list(seeds)
        self.loaded = []
        self._use_seed_rank = use_seed_rank

    # ---- 惰性加载(首次 predict 时) ----
    def _ensure(self):
        if self.loaded:
            return
        fams = ["lgb", "xgb", "cat"]
        ext = {"lgb": "txt", "xgb": "json", "cat": "cbm"}
        for fam in fams:
            for seed in self.seeds:
                path = f"{self.pool_dir}/JOINT_{fam}_seed{seed}.{ext[fam]}"
                if fam == "lgb":
                    import lightgbm as lgb
                    from pathlib import Path
                    m = lgb.Booster(model_file=str(path))
                    best = _load_best_iter(path)
                    self.loaded.append(("lgb", m, best))
                elif fam == "xgb":
                    import xgboost as xgb
                    mm = xgb.Booster(); mm.load_model(path)
                    self.loaded.append(("xgb", mm, _load_best_iter(path)))
                else:
                    from catboost import CatBoostClassifier
                    mm = CatBoostClassifier(); mm.load_model(path)
                    self.loaded.append(("cat", mm, None))

    # ---- 融合 ----
    def fused(self, X):
        """X: [n,58] -> fusion probability [n] (整段秩 batch 模式)。"""
        self._ensure()
        n = len(X)
        P = np.zeros((len(self.loaded), n), dtype=np.float64)
        for i, (fam, m, best) in enumerate(self.loaded):
            if fam == "lgb":
                p = m.predict(X, num_iteration=best)
            elif fam == "xgb":
                import xgboost as xgb
                p = m.predict(xgb.DMatrix(X), iteration_range=(0, best)) if best else m.predict(xgb.DMatrix(X))
            else:
                p = m.predict(X, prediction_type="Probability")[:, 1]
            P[i] = np.asarray(p, dtype=np.float64)
        R = np.stack([np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1) for i in range(P.shape[0])],
                     axis=0)
        return R.mean(axis=0)


class SessionRankFuser:
    """增量"当日秩"融合: 对某 symbol 当天聚合 15 个概率, 各自增量秩后平均。

    用增量有序插入(注意: 最小化重排), 仅作为"整段秩"的在线工程近似
    (计划决策3); 其与整段秩的偏差由 replay_online 实测量化, 若超容忍再回退
    为"跨样本秩 + 重校准"(见 live/predict.py 注释中的回退方案)。每日新开实例。
    """

    def __init__(self, pool_dir, seeds=(42, 49, 56, 63, 70)):
        self.pool = PoolPredictor(pool_dir, seeds, use_seed_rank="session")
        self.pool._ensure()
        self._buf = [[] for _ in self.pool.loaded]       # 每个 booster 当天有序概率缓冲

    def fused_at(self, Xrow):
        """Xrow: [1,58] -> (融合概率, 当日置信度)。新行概率插入对应缓冲取增量秩。"""
        n = 1
        rank_sum = 0.0
        for i, (fam, m, best) in enumerate(self.pool.loaded):
            x = Xrow.reshape(1, -1)
            if fam == "lgb":
                p = float(m.predict(x, num_iteration=best)[0])
            elif fam == "xgb":
                import xgboost as xgb
                p = float(m.predict(xgb.DMatrix(x))[0])
            else:
                p = float(m.predict(x, prediction_type="Probability")[0, 1])
            buf = self._buf[i]
            pos = bisect.bisect_left(buf, p)
            buf.insert(pos, p)
            seen = len(buf)
            r = pos / (seen - 1) if seen > 1 else 0.5        # 百分位秩 [0,1]
            rank_sum += r
        fused = rank_sum / self.pool.loaded.__len__()
        conf = abs(fused - 0.5) * 2
        return float(fused), float(conf)


def _load_best_iter(path):
    """从 bundle 目录读取 best_iteration 记录; 无记录则 None(使用默认)。"""
    import os
    info = os.path.join(os.path.dirname(path), "best_iteration.json")
    if os.path.exists(info):
        import json
        d = json.load(open(info))
        key = os.path.basename(path)
        if key in d:
            return int(d[key])
    return None


class RollingThreshold:
    """R2 逐日滚动阈值: 维护历史置信度滑窗, 每日更新。"""

    def __init__(self, seed_conf):
        self._hist = list(seed_conf)

    def _days_in_hist(self):
        # 近似天数: 用历史里唯一 UTC 日计数
        return 1  # 由外部按日边界调; 见 set_seed/to_days

    def threshold(self):
        """返回当日阈值 tau(99 分位); 历史不足 COLD_MIN_DAYS 返回 None(冷启动不发)。"""
        # 天数由外部维护(见 observe 的 day 计数), 这里直接用 hist 长度近似天数
        days = max(1, int(len(self._hist) / SAMP))
        if days < COLD_MIN_DAYS:
            return None
        return float(np.percentile(np.asarray(self._hist), P99))

    def observe(self, day_confs):
        """把今天全部置信度并入滑窗并裁剪到 MAX_HIST。"""
        self._hist.extend(list(day_confs))
        if len(self._hist) > MAX_HIST:
            del self._hist[:len(self._hist) - MAX_HIST]


def conf_of(p):
    return abs(float(p) - 0.5) * 2