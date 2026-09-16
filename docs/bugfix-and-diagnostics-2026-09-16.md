# Simpletrade Bug 排查与模型天花板诊断

> 日期：2026-09-16  
> 目标：ETH H=15 top1% 准确率 ≥ 65%，每天交易 ≥ 14 笔  
> 现状：全时段 top1% 准确率 60-61%，每天约 1 笔

---

## 一、项目验收约束速查（config.py）

| 参数 | 值 | 备注 |
|------|-----|------|
| `COVERAGE` | 0.01 | top1% |
| `TARGET_ACCURACY` | 0.65 | top1% 准确率 ≥ 65% |
| `MIN_TRADES_PER_DAY` | 14 | 每天 ≥ 14 笔 |
| `CROSS_ASSET_MAX_DELTA` | 0.03 | BTC/ETH 准确率差 ≤ 3pp |
| `TRAIN_END` | 2024-06-30 | 训练截止 |
| `ES_END` | 2024-09-30 | 早停验证截止 |
| `META_VAL_END` | 2025-09-30 | 元训练/阈值选择截止 |

**问题**：`MIN_TRADES_PER_DAY = 14` 与 `COVERAGE = 0.01` 对 H=15 存在硬冲突。H=15 每天有 96 bars，top1% = 0.96 tpd。要满足 ≥ 14 tpd，需要覆盖 ≥ 15%，此时 ETH 准确率约 57%。14 tpd 合理目标可能是 **多 horizon 叠加**（H=3 + H=5 + H=15 同时预测）。

---

## 二、本次修复的 Bug 清单

### Bug #1（致命）：models/gbdt.py FEATURES 列表 29/59 列不存在

**现象**：`GBDTModel.fit()` 在 `ctx.X_subset(FEATURES, mask)` 处抛 `ArrowInvalid: No match for FieldRef.Name(z_10)`。

**根因**：`models/gbdt.py` 维护了一份手写的 FEATURES 列表（59 列），与 `features.py` 实际输出（38 列）不同步。被移除/从未实现的列包括：

```
z_10, lr_5, lr_30, pos_60, is_us, is_eu, body_ratio, body_pos_60,
up_wick, lo_wick, ngreen_10, gap, tbr_z_30, buyvol_strength_30,
tb_acc_30, cvd_dir_30, tbr_hi_60, up_body_ratio_30, rvol_ratio_60_5,
pos_tbr_interact, vol_mom_interact, pos_cvd_interact, di_spread,
di_uptrend, mom_vol_confirm, z_divergence, cvd_accel, vol_cvd_interact,
di_plus
```

**修复**：删除手写 FEATURES 列表，改为引用 `validate_eth_quick.BASE_FEATURES`（38 列，与 ds parquet 实际列一致）。

### Bug #2（致命）：GBDTModel 未拼接 cross-asset + extra features

**现象**：即使 FEATURES 修好，`GBDTModel.fit()` 只用了 ds 基础 38 列，但项目最佳实践（`validate_eth_quick.get_X`）是 38 基础 + 3 runtime extra + 17 cross-asset = 58 列。

**根因**：`models/gbdt.py` 的 fit/predict 全程用 `ctx.X_subset(feats, mask)` 只读 ds 静态列，跳过了：
- extra 特征（`hour_sin_rvol_60`, `session_minutes`, `hour_sin_hour_cos`）：由 `compute_extra_raw` 从 raw 1min 数据实时计算
- cross-asset 特征（BTC_* / ETH_* 17 列）：ds 里有但需按目标币符号切换前缀

**修复**：新增 `get_X_with_cross(ctx, extra_raw, mask, feats)` 函数，语义上完全对齐 `validate_eth_quick.get_X()`。fit() 内先 `extra_raw = compute_extra_raw(ctx)`，然后所有 X 矩阵通过该函数构建。

**修复前 vs 修复后**（ETH H15, 单 seed42, BAG params, test top1%）：

| 状态 | 特征维度 | AUC | test top1% |
|------|---------|-----|-----------|
| 修复前 | 38（缺 20 列） | ~0.531 | 56.16% |
| 修复后 | 58（完整） | 0.5336 | 60.49% |
| validate_eth_quick（基准） | 58 | 0.5348 | 60.64% |

修复后 GBDTModel 与 validate_eth_quick 结果完全对齐（差异 < 0.2pp）。

### Bug #3（中等）：build_dataset.py 输出文件名忽略 horizon 参数

**现象**：`build_symbol_dataset(symbol, horizon=30)` 仍然输出 `ds_{symbol}.parquet`，horizon 被丢弃。

**根因**：文件名硬编码为 `f"ds_{symbol}.parquet"`。

**修复**：改为 `f"ds_{symbol}_h{horizon}.parquet"`。当前数据集已是按 horizon 命名的（`ds_ETH_h15.parquet`, `ds_ETH_h30.parquet`），说明实际跑数据集时用了其他脚本（`backtest_short/build_dataset_short.py`），但 `build_dataset.py` 作为主路径必须修复。

### Bug #4（轻量）：GBDTModel.fit() 内 extra_raw 变量未定义

**现象**：修复 Bug #2 后，fit() 内调用 `get_X_with_cross(ctx, extra_raw, ...)` 但 `extra_raw` 未在该作用域定义。

**修复**：在 `feats = list(BASE_FEATURES)` 后加一行 `extra_raw = compute_extra_raw(ctx)`。

---

## 三、权重缩放：被误判为 Bug 的参数

### 结论

**`raw_w * 50` 不是 bug**。

### 证据

ETH H15 train set |ret| 均值 = 0.00275，所以：

| 缩放 | clip 范围 | clip_lo | clip_hi | w mean |
|------|-----------|---------|---------|--------|
| *1 | [0.5,5] | 100% | 0% | 0.500 |
| **\*50** | [0.5,5] | **56%** | **0%** | **0.724** |
| *1000 | [0.5,5] | 10.2% | 8.9% | 3.076 |

配合 BAG params（lr=0.02, bagging_fraction=0.8, l2=1.0）时，*50 反而是最优：

| 配置 | seed42 bi | AUC | test top1% |
|------|----------|-----|-----------|
| uniform w=1 | 57 | 0.5318 | 55.93% |
| **\*50 + bad_hours\*2** | **108** | **0.5348** | **60.64%** |
| *1000 | 49 | 0.5336 | 57.66% |

**原因**：BAG params 有强正则（l2=1.0, bagging=0.8），大权重让模型过度关注少量大波动样本，反而更早过拟合（bi=49 vs 108）。小权重 + 正则的组合效果更好。

---

## 四、Seed 稳定性诊断（10 个 seed, BAG params）

### 数据

| seed | AUC_es | bi | test top1% | avg_ret |
|------|--------|-----|-----------|---------|
| **7** | **0.5358** | 86 | **61.49%** | +2.5bps |
| **49** | **0.5351** | 112 | **61.45%** | +3.6bps |
| 2024 | 0.5345 | 105 | 60.72% | +2.4bps |
| **42** | 0.5348 | 108 | **60.64%** | +2.7bps |
| 1 | 0.5344 | 107 | 59.80% | +2.6bps |
| 123 | 0.5338 | 46 | 59.72% | +2.3bps |
| 99 | 0.5349 | 157 | 59.47% | +1.8bps |
| 63 | 0.5338 | 16 | 59.33% | +0.8bps |
| 56 | 0.5359 | 178 | 58.89% | +2.9bps |
| 70 | 0.5348 | 5 | **57.35%** | -0.2bps |

```
mean  = 59.89%    std = 1.19pp
best  = seed 7  (61.49%)
worst = seed 70 (57.35%, bi=5 异常早停)
```

### 关键发现

1. **AUC 差 0.002，top1% acc 差 4pp**。AUC 和头部准确率弱正相关，小样本的排序质量差异被 top-k 筛选放大。
2. **best_iteration 跨度 5~178**。seed70/63 异常早停（bi=5/16），训练几乎没学到东西。
3. **5-seed bagging 的排名平均 = 62.78%**，但 avg_ret 只有 +0.4bps——bagging 提高了准确率但信号质量下降。概率平均 = 60.91%，接近单 seed 均值。
4. **结论**：没有 magic seed。均值 59.9% 就是当前架构的天花板。

---

## 五、模型天花板分析

### 全时段

| 覆盖 | test top1% | trades/day | 达标? |
|------|-----------|-----------|-------|
| 0.5% | 61.9% | 0.48 | ✗ |
| **1%** | **61.1%** | **0.96** | **✗** |
| 2% | 60.2% | 1.92 | ✗ |
| 5% | 57.1% | 4.80 | ✗ |

**所有覆盖级别 ≤ 62%，永远到不了 65%。**

### 时段过滤（唯一能过 65% 的方法，但有代价）

| 方案 | meta top1% | test top1% | tpd |
|------|-----------|-----------|-----|
| 全时段 | 59.6% | 61.1% | 1.0 |
| top3h (3,14,21 UTC) | 65.2% | 71.5% | 0.11 |
| top6h (0,3,5,12,14,21) | 65.2% | 68.3% | 0.12 |

代价：每天仅 0.11 笔（约 9 天 1 笔），完全不满足 `MIN_TRADES_PER_DAY=14`。

### 数学天花板

```
当前 LightGBM AUC ≈ 0.535
    ↓
要全天 top1% ≥ 65%，需要 AUC ≈ 0.55+
要全天 top1% ≥ 70%，需要 AUC ≈ 0.57+
```

所有树模型（LGB/XGB/CatBoost）的 AUC 都在 0.532-0.537 之间。不是参数问题，是架构和数据源给的。

---

## 六、数据泄露审计（已通过）

early_stop split 上：

- 全部 58 个特征与 label 的 Spearman |rho| ≤ 0.057
- 全部 58 个特征与 ret_future 的 Spearman |rho| ≤ 0.049
- preds vs retf Spearman ρ = +0.053
- 模型 AUC = 0.5346

远低于 0.1 泄露阈值，无未来函数。

---

## 七、后续突破方向

当前树模型 AUC ≈ 0.535 → top1% 61% 已是天花板。要突破 0.55 AUC：

| 方向 | 预期提升 | 难度 |
|------|---------|------|
| **多 horizon Union**（H3+H5+H15 同时预测，覆盖不同周期噪声模式） | +2-3pp | 🟡 中 |
| **Hour-specific MoE**（每个 UTC 小时独立专家模型） | +1-2pp | 🟡 中 |
| **Transformer**（对 96-bar 序列做 end-to-end 预测） | +1-3pp | 🟠 高 |
| **新增数据源**（orderbook flow, funding 动态, ETF flow） | +2-4pp | 🟠 高 |
| **重新定义 label**（软标签 + 幅度加权训练目标） | +0.5-1pp | 🟢 低 |
| **跨币种联合训练**（BTC/ETH 共享参数, horizon=30 时已证 AUC+0.028） | +1-2pp | 🟢 低 |

---

## 八、文件改动汇总

```
models/gbdt.py                         ← Bug #1+2+4: FEATURES 对齐 + cross/extra 拼接
data_processing/build_dataset.py       ← Bug #3: 输出文件名加 horizon
validate_eth_quick.py                  ← 无改动（一直正确）
live/train_pool.py                     ← 无改动（*50 非 bug）
config.py                              ← 无改动（但 MIN_TRADES_PER_DAY=14 需重新评估）
```

---

## 九、关键代码变更

### models/gbdt.py 新增 get_X_with_cross

```python
from validate_eth_quick import (
    compute_extra_raw, get_extra_for_mask,
    FEATURES as BASE_FEATURES,
    CROSS_FEATURES, EXTRA_FEATURE_NAMES,
)

def get_X_with_cross(ctx, extra_raw, mask, feats):
    """表=ds 列 + runtime extra + cross-asset"""
    X_base = ctx.X_subset(feats, mask)
    X_extra = get_extra_for_mask(extra_raw, ctx, mask)
    prefix = "BTC_" if ctx.symbol == "ETH" else "ETH_"
    X_cross = ctx.X_subset([prefix + f for f in CROSS_FEATURES], mask)
    return np.column_stack([X_base, X_extra, X_cross])
```

### build_dataset.py 修复输出路径

```python
# 修复前:
out = os.path.join(config.DS_DIR, f"ds_{symbol}.parquet")
# 修复后:
out = os.path.join(config.DS_DIR, f"ds_{symbol}_h{horizon}.parquet")
```

---

_本文档基于 2026-09-16 完整的 ETH H=15 训练/评估/诊断流程撰写，所有数据可复现（`python run.py --stage data` 重建数据集后跑 `GBDTModel` 即可对齐）。_
