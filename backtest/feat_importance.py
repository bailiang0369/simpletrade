#!/usr/bin/env python3
"""从已训练 JOINT LGB 模型读取特征重要性(gain), 输出排名。
用于制定特征精简清单: 保留贡献度高的, 注释掉低贡献的。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
from validate_eth_quick import FEATURES, EXTRA_FEATURE_NAMES

MODEL_ROOT = "/workspace/models_saved/pool20_joint"
NAMES = FEATURES + EXTRA_FEATURE_NAMES

scores = np.zeros(len(NAMES))
for seed in (42, 49, 56, 63, 70):
    m = lgb.Booster(model_file=f"{MODEL_ROOT}/JOINT_lgb_seed{seed}.txt")
    imp = m.feature_importance(importance_type="gain")
    scores += imp / max(imp.sum(), 1e-12)
scores /= 5.0

order = np.argsort(-scores)
print(f"特征总数: {len(NAMES)}")
print(f"{'rank':>4} {'gain%':>7}  {'特征名':<28}")
for r, i in enumerate(order):
    print(f"{r+1:>4} {scores[i]*100:6.2f}%  {NAMES[i]:<28}")

# 累计贡献
cum = np.cumsum(np.sort(scores)[::-1])
for pct in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    k = int(np.searchsorted(cum, pct)) + 1
    print(f"top {k} 特征覆盖 {pct:.0%} 总贡献")
