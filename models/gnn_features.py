"""精简版 10 维核心特征 (Lightweight Feature Set for GNN / DL models).
严格内存把控：维度从 38/59 降至 10，避免图神经网络时空张量内存爆炸。
10 维核心特征：
1. lr_15: 15分钟对数收益率
2. lr_120: 120分钟对数收益率
3. rvol_30: 30分钟已实现波动率
4. rvol_60: 60分钟已实现波动率
5. z_30: 30分钟价格 Z-Score
6. z_120: 120分钟价格 Z-Score
7. pos_30: 30分钟价格区间位置
8. cvd_30: 30分钟主动买卖净量比
9. hour_sin: 日内小时正弦
10. hour_cos: 日内小时余弦
"""

CORE_GNN_FEATURES = [
    "lr_15", "lr_120",
    "rvol_30", "rvol_60",
    "z_30", "z_120",
    "pos_30",
    "cvd_30",
    "hour_sin", "hour_cos"
]
