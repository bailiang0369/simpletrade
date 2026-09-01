"""ACE (Shape Clustering) 模型: 基于FAISS的形态聚类，统计每个形态胜率。

思路来源: 参考用户提交的FAISS v30 notebook，适配当前项目管线。

核心思路:
- 对不同窗口长度的价格/指标序列做K-Means聚类，得到"形态簇"
- 在训练集统计每个簇的胜率（edge = 胜率 - 0.5）
- 多尺度聚类结果相加得到最终ACE得分
- 严格遵守因果: 聚类只在训练集拟合，测试集只做查询不更新
"""
import gc
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import faiss
import joblib
import config
from models.base import BaseModel

# 多尺度聚类配置
HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':    {'window': 120, 'n_clusters': 600, 'repeat': 15},
}

# 使用哪些特征做聚类 (严格参照用户原始FAISS notebook)
# 只有两个: close价格序列 + stoch指标序列，各自独立做5尺度聚类
CLUSTER_FEATURES = {
    'close':      True,  # 原始价格序列聚类 (ipynb核心)
    'stoch':      True,  # stoch指标序列聚类 (ipynb用stoch_k_29_3, 从OHLC实时计算)
}
STOCH_PERIOD = 50  # stoch计算周期 (参考ipynb的29和69)


def _compute_stoch(h, l, c, period=STOCH_PERIOD):
    """向量化计算stoch_K指标。
    
    返回: 与输入等长的数组，前period-1个为nan，后续为stoch_k值。
    """
    n = len(h)
    if n < period:
        return np.full(n, np.nan, dtype=np.float32)
    
    h_windows = sliding_window_view(h, period)
    l_windows = sliding_window_view(l, period)
    h_max = np.max(h_windows, axis=-1)  # (n-p+1,)
    l_min = np.min(l_windows, axis=-1)  # (n-p+1,)
    
    rng = h_max - l_min
    rng[rng == 0] = 1e-9
    stoch = ((c[period-1:] - l_min) / rng * 100).astype(np.float32)
    
    # 前period-1个位置填nan
    result = np.full(n, np.nan, dtype=np.float32)
    result[period-1:] = stoch
    return result


class FaissShapeModel(BaseModel):
    """多尺度形态聚类 (ACE) 模型。

    流程:
    1. 在训练集上对每个尺度分别做FAISS K-Means聚类
    2. 计算每个cluster的胜率偏移 edge = p(up) - 0.5
    3. 对测试集每个样本，查询所有尺度的cluster，求和edge得到score
    4. 输出score作为预测概率（其实是得分，但要兼容管线，sigmoid转[0,1]）

    内存友好: 每个尺度独立聚类，处理完一个释放一个，不占太多内存。
    """
    def __init__(self, seed=None, max_train=None, config=None):
        super().__init__(seed)
        self.config = config or HYBRID_CONFIG
        self.max_train = max_train
        self.kms = {}            # {scale_name: faiss.Kmeans}
        self.edge_map = {}       # {scale_name: np.array[edge]} 每个cluster的胜率偏移
        self.n_samples_map = {}  # {scale_name: np.array[int]} 每个cluster的样本量
        self.scale_feat_map = {} # {scale_name: feat_name} 每个尺度用的特征名
        self.scale_cfg_map = {}  # {scale_name: {'window': w, 'n_clusters': n, 'repeat': r}}
        self.fitted = False

    def _normalize_window(self, windows, is_indicator):
        """窗口内归一化，每个窗口独立。"""
        if is_indicator:
            # 指标已经在[0,100]范围，直接缩放到[0,1]
            return windows / 100.0
        else:
            # 价格归一化到[0,1]区间
            w_min = windows.min(axis=1, keepdims=True)
            w_max = windows.max(axis=1, keepdims=True)
            rng = w_max - w_min
            rng[rng == 0] = 1e-9
            return (windows - w_min) / rng

    def _build_windows(self, data_array, window, is_indicator, end_k=3, r_count=0):
        """构建滑动窗口特征，支持末端强化（repeat末端k个特征）。"""
        n = len(data_array)
        if n < window:
            return np.zeros((0, window), dtype=np.float32)

        # 滑动窗口
        shape = (n - window + 1, window)
        strides = (data_array.strides[0], data_array.strides[0])
        windows = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
        norm = self._normalize_window(windows, is_indicator)

        # 末端强化：重复最后k个特征，让模型更关注末端走势
        if r_count > 0 and end_k > 0:
            end_feats = norm[:, -end_k:]
            norm = np.hstack([norm] + [end_feats] * r_count)

        return np.ascontiguousarray(norm, dtype=np.float32)

    def _train_one_scale(self, name, cfg, tr_data, tr_labels, is_indicator, feat_name=None):
        """训练单个尺度的聚类和计算edge。
        
        改进:
          - 增大 niter 从 40 到 100 + nredo=3 确保收敛
          - 贝叶斯平滑 edge: 小 cluster 向 0 收缩, 避免噪声估计
          - 记录每个 cluster 的样本量用于测试时权重
        """
        w = cfg['window']
        n_c = cfg['n_clusters']
        r_count = cfg.get('repeat', 0)
        print(f"[faiss_shape] {name}: window={w}, n_clusters={n_c}, repeat={r_count}", flush=True)

        windows = self._build_windows(tr_data, w, is_indicator, r_count=r_count)
        if len(windows) == 0:
            print(f"[faiss_shape] {name}: 数据不足，跳过", flush=True)
            return

        # FAISS K-Means训练 (增大niter确保收敛, nredo避免局部最优)
        dim = windows.shape[1]
        km = faiss.Kmeans(dim, n_c, niter=100, nredo=3, verbose=False, seed=self.seed)
        km.train(windows)
        _, ids = km.index.search(windows, 1)
        ids = ids.flatten()  # (n_windows,)

        # 计算每个cluster的edge (胜率 - 0.5) + 贝叶斯平滑
        labels_w = tr_labels[w-1:]
        edge = np.zeros(n_c, dtype=np.float32)
        n_samples_per_cluster = np.zeros(n_c, dtype=np.int32)
        for cid in range(n_c):
            mask = (ids == cid)
            n_s = mask.sum()
            n_samples_per_cluster[cid] = n_s
            if n_s > 0:
                raw_edge = labels_w[mask].mean() - 0.5
                # 贝叶斯收缩: 样本量少的 cluster 向 0 收缩
                # 先验强度 n_prior = 100, 样本量 < 100 时强烈收缩
                n_prior = 100
                shrink = n_s / (n_s + n_prior)
                edge[cid] = raw_edge * shrink
            else:
                edge[cid] = 0.0

        self.kms[name] = km
        self.edge_map[name] = edge
        self.n_samples_map[name] = n_samples_per_cluster  # 记录样本量
        if feat_name:
            self.scale_feat_map[name] = feat_name
        self.scale_cfg_map[name] = {'window': w, 'n_clusters': n_c, 'repeat': r_count}
        del windows; gc.collect()

    def _get_feat_arr(self, ctx, pos, feat_name):
        """获取特征序列（close或stoch）。"""
        if feat_name == 'close':
            return ctx.c[pos].astype(np.float32), False
        elif feat_name == 'stoch':
            h = ctx.h[pos]; l = ctx.l[pos]; c = ctx.c[pos]
            feat = _compute_stoch(h, l, c)
            return np.nan_to_num(feat, nan=50.0).astype(np.float32), True
        return None, None

    def fit(self, ctx):
        """训练: 仅在训练集拟合聚类，计算edge。
        严格遵循用户 original FAISS notebook 思路：多尺度滑动窗口聚类。
        """
        t0 = time.time()
        tr_mask = ctx.split_rows['train']
        tr_pos = ctx.ds_to_raw[tr_mask]
        tr_y = ctx.y('train')

        for feat_name, use_it in CLUSTER_FEATURES.items():
            if not use_it:
                continue
            feat_arr, is_indicator = self._get_feat_arr(ctx, tr_pos, feat_name)
            if feat_arr is None:
                continue

            for name, cfg in self.config.items():
                scale_name = f"{name}_{feat_name}"
                self._train_one_scale(scale_name, cfg, feat_arr, tr_y, is_indicator, feat_name=feat_name)

        self.fitted = True
        print(f"[faiss_shape] fit done in {time.time()-t0:.0f}s, scales={len(self.kms)}", flush=True)
        return self

    def _predict_one_scale(self, name, cfg, data_array, is_indicator):
        """预测单个尺度: 返回每个位置的edge得分。
        
        改进: 距离加权投票 (top-3 最近簇), 距离越远权重越低。
        """
        w = cfg['window']
        r_count = cfg.get('repeat', 0)
        km = self.kms[name]
        edges = self.edge_map[name]

        windows = self._build_windows(data_array, w, is_indicator, r_count=r_count)
        if len(windows) == 0:
            return np.zeros(len(data_array), dtype=np.float32)

        # 距离加权投票: 取 top-3 最近簇
        k = min(3, len(edges))
        dists, ids = km.index.search(windows, k)
        # 距离转权重: 距离越近权重越大
        weights = 1.0 / (dists.astype(np.float32) + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        # 加权 edge
        weighted_edge = np.sum(edges[ids] * weights, axis=1)

        scores = np.zeros(len(data_array), dtype=np.float32)
        scores[w-1:] = weighted_edge
        return scores

    def predict(self, ctx, split):
        """预测: 多尺度得分相加，然后转成概率。"""
        mask = ctx.split_rows[split]
        pos = ctx.ds_to_raw[mask]

        total_score = np.zeros(len(pos), dtype=np.float32)
        n_scales = 0

        # 遍历所有训练好的尺度
        for scale_name, km in self.kms.items():
            feat_name = self.scale_feat_map.get(scale_name)
            if feat_name is None:
                continue

            feat_arr, is_indicator = self._get_feat_arr(ctx, pos, feat_name)
            if feat_arr is None:
                continue

            feat_arr = np.ascontiguousarray(feat_arr.astype(np.float32))
            cfg = self.scale_cfg_map.get(scale_name, {'window': 60, 'n_clusters': 100, 'repeat': 0})

            s = self._predict_one_scale(scale_name, cfg, feat_arr, is_indicator)
            total_score += s
            n_scales += 1

        gc.collect()

        if n_scales == 0:
            print(f"[faiss_shape] 没有可用尺度，返回0.5", flush=True)
            return np.full(len(total_score), 0.5, dtype=np.float32)

        # 将总分通过 tanh 映射到 [0, 1]
        temp = 0.2
        prob = 0.5 + 0.5 * np.tanh(total_score / (n_scales * temp))
        return prob.astype(np.float32)

    def save(self, path):
        """保存模型: kms + edge_map + n_samples_map。"""
        state = {
            'config': self.config,
            'edge_map': self.edge_map,
            'n_samples_map': self.n_samples_map,
            'centroids': {name: km.centroids for name, km in self.kms.items()},
            'n_clusters': {name: km.k for name, km in self.kms.items()},
            'scale_feat_map': self.scale_feat_map,
            'scale_cfg_map': self.scale_cfg_map,
        }
        joblib.dump(state, f"{path}.joblib")
        print(f"[faiss_shape] saved to {path}.joblib", flush=True)

    def load(self, path):
        """加载模型。"""
        state = joblib.load(f"{path}.joblib")
        self.config = state['config']
        self.edge_map = state['edge_map']
        self.n_samples_map = state.get('n_samples_map', {})
        self.scale_feat_map = state.get('scale_feat_map', {})
        self.scale_cfg_map = state.get('scale_cfg_map', {})
        self.kms = {}
        for name, centroids in state['centroids'].items():
            n_c = state['n_clusters'][name]
            dim = centroids.shape[1]
            km = faiss.Kmeans(dim, n_c, niter=0, verbose=False)
            km.k = n_c
            km.centroids = centroids
            km.index.reset()
            km.index.add(centroids)
            self.kms[name] = km
        self.fitted = True
        print(f"[faiss_shape] loaded from {path}.joblib, scales={len(self.kms)}", flush=True)
        return self