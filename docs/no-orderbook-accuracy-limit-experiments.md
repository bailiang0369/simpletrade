# ETH H=15 无 Orderbook 数据准确率极限实验

> 日期：2026-09-16 深夜
> 目标：acc ≥ 65%, tpd ≥ 15
> 数据源：**raw_ETH / raw_BTC** — 1min OHLCV + buy/sell_vol + funding（无 orderbook）
> 测试集：2025-09-30 ~ 2026-08-29（333 天，479,520 根 1min K 线）

---

## 执行命令

```bash
# 重建数据集 (如果丢失)
python run.py --stage data

# 核心实验
python mh_experiments/mh_union.py       # Multi-Horizon 全局模型 union
python mh_experiments/per_hour.py       # Per-Hour 模型
python mh_experiments/union_all.py      # 联合策略搜索
python mh_experiments/final_consensus   # Consensus 参数精细扫描
```

---

## 数据定义

**H = horizon = 预测未来多少分钟后的方向。**

| 术语 | 含义 |
|------|------|
| **H=3** | 每根 1min bar 预测 3 分钟后 ETH 是涨是跌，每天 480 bars |
| **H=5** | 每根 1min bar 预测 5 分钟后方向，每天 288 bars |
| **H=15** | 每根 1min bar 预测 15 分钟后方向，每天 96 bars ← 主 horizon |
| **H=30** | 每根 1min bar 预测 30 分钟后方向，每天 48 bars |

基础数据始终是 **1 分钟 K 线**，不 resample。不同 H 只是 label 的"回看时间"不同。

---

## 实验 1：单 Horizon 基线

LightGBM (num_leaves=127, lr=0.01)，subsample 训练集到 50-80 万。

| Horizon | train AUC | test AUC | top1% acc | top5% acc | top10% acc | tpd (top5%) |
|---------|-----------|----------|-----------|-----------|------------|-------------|
| H=3 | 0.5295 | 0.5391 | 57.2% | 57.3% | 56.1% | 24.0 |
| H=5 | 0.5323 | 0.5396 | 58.0% | 57.5% | 56.5% | 14.4 |
| **H=15** | **0.5341** | **0.5427** | **60.3%** | **56.8%** | **55.9%** | **4.8** |
| H=30 | 0.5328 | 0.5444 | 58.3% | 57.0% | 56.3% | 2.4 |

**天花板**：AUC 0.53-0.54，top1% acc 57-60%。

---

## 实验 2：Multi-Horizon Global Union

各 horizon 独立训 LGB，然后取 top K% 并集。

| 策略 | tpd | acc (H=15 label) | ret_bps |
|------|-----|-------------------|---------|
| H=15 top1% | 1.0 | 60.3% | +2.0 |
| H=15 top5% | 4.8 | 56.8% | +0.3 |
| H=15 top20% | 19.2 | 54.9% | +0.1 |
| **4-horizon top5% union** | 159.4 | 55.7% | +0.0 |
| 4-horizon top3% union | 94.1 | 56.3% | +0.1 |
| 4-horizon top2% union | 71.6 | 57.2% | +0.6 |

**发现**：Union 大幅提升 tpd（从 1 到 71），但 acc **下降**了。因为不同 horizon 的 top K% bar 是重叠的噪声。

---

## 实验 3：Per-Hour 模型

24 个 UTC 小时各训独立 LGB（num_leaves=31, lr=0.03）。

### Per-Hour test AUC (H=15)

```
UTC 10: 0.558 | UTC 19: 0.558 | UTC 22: 0.558 | UTC 12: 0.551
UTC 16: 0.551 | UTC 17: 0.544 | UTC 23: 0.535 | UTC  4: 0.535
UTC 14: 0.532 | UTC 18: 0.526 | UTC  8: 0.522 | UTC  0: 0.522
UTC 21: 0.517 | UTC  7: 0.512 | UTC 11: 0.512 | UTC  3: 0.511
UTC  1: 0.510 | UTC 15: 0.509 | UTC 20: 0.503 | UTC  6: 0.501
```

### Per-Hour top1% accuracy (全局模型在每小时单独评估)

```
UTC 18: 79.9% | UTC 13: 74.9% | UTC  0: 73.4% | UTC  3: 73.4%
UTC 14: 72.4% | UTC  5: 70.4% | UTC  9: 69.8% | UTC  4: 68.3%
UTC 19: 66.8% | UTC 12: 66.8% | UTC 21: 65.3% | UTC 16: 64.8%
UTC  2: 63.3% | UTC 10: 62.8% | UTC 15: 58.8% | UTC 22: 59.8%
UTC  1: 57.3% | UTC 17: 56.8% | UTC 23: 56.3% | UTC  7: 47.7%
UTC  8: 50.8% | UTC 11: 44.7% | UTC  6: 46.2% | UTC 20: 36.7%
```

**关键洞察**：好时段（UTC 18,13,0,3,14）top1% acc 72-80%，坏时段（UTC 06,07,11,20）只有 37-51%。**不是模型不够聪明，是市场状态完全不同。**

---

## 实验 4：Consensus 共识策略

**定义**：每个 horizon 独立计算 percentile rank（在该 horizon 自己的 split 内），要求至少 N 个 horizon 的 percentile 同时 > thresh。

### Percentile rank 共识（meta_val 选 thresh → test 评估，无泄露）

| thresh | min horizons | test tpd | test acc (H=15) | ret_bps | 达标? |
|--------|-------------|----------|-----------------|---------|-------|
| >98% | 3 | 11 | 60.6% | +1.3 | |
| >97% | 4 | 7 | 61.9% | +3.0 | |
| >96% | 3 | 26 | 58.8% | +0.8 | ✅tpd |
| >95% | 4 | 14 | 60.6% | +2.3 | |
| >94% | 4 | 18 | 59.9% | +1.7 | ✅tpd |
| >94% | 3 | 44 | 58.0% | +0.4 | ✅tpd |
| >93% | 4 | 23 | 59.5% | +1.4 | ✅tpd |
| >92% | 4 | 29 | 59.2% | +1.2 | ✅tpd |
| >90% | 4 | 42 | 58.1% | +0.4 | ✅tpd |

### Raw pred cutoff 共识（更严格，用 meta_val top N% 的 raw pred 值作为 cutoff）

| top%/h | min | test tpd | test acc (H=15) | ret_bps | 达标? |
|--------|-----|----------|-----------------|---------|-------|
| 0.5% | 3 | 3 | **65.5%** | +4.5 | ✅acc 但 tpd 不够 |
| 0.5% | 4 | 1 | **67.5%** | +12.6 | ✅acc 但 tpd 不够 |
| 1.0% | 4 | 2 | **65.1%** | +5.7 | ✅acc 但 tpd 不够 |
| 3.0% | 2 | 50 | 58.1% | +0.3 | ✅tpd |
| 3.0% | 3 | 20 | 59.4% | +1.0 | ✅tpd |
| 5.0% | 4 | 16 | 60.3% | +2.1 | ✅tpd |

---

## 实验 5：Stacking

把 H=3,5,15,30 四个全局 LGB 的 pred 作为特征（11 维：4 个 raw pred + 差值 + 乘积 + min/max/mean），训一个 meta-LGB。

```
te_AUC = 0.5411 (leaves=15, lr=0.01)
top1%: tpd=14.4, acc=59.7%, ret=+2.1bps

比单独 H=15 (te_AUC=0.5427) 还差！
```

**原因**：4 个 horizon 的 base LGB preds 高度相关（彼此 rank correlation > 0.85），meta-LGB 没学到额外信息。这就是 Stacking 的经典陷阱 —— base models 多样性不够。

---

## 综合结论

### 1. 最好的 acc（test set, H=15 label）

| 策略 | acc | ret_bps | tpd |
|------|-----|---------|-----|
| Raw pred consensus top0.5% min4 | **67.5%** | +12.6 | 1 |
| Raw pred consensus top0.5% min3 | **65.5%** | +4.5 | 3 |
| Raw pred consensus top1.0% min4 | **65.1%** | +5.7 | 2 |
| **H=15 top1% baseline** | **60.3%** | +2.0 | 1 |

### 2. 最好的 tpd（≥ 15）

| 策略 | tpd | acc | ret_bps |
|------|-----|-----|---------|
| Raw pred consensus top1.0% min2 | 16 | 60.3% | +1.8 |
| Raw pred consensus top5.0% min4 | 16 | 60.3% | +2.1 |
| Raw pred consensus top3.0% min3 | 20 | 59.4% | +1.0 |
| **4-horizon union top2%** | 72 | 57.2% | +0.6 |

### 3. 核心问题

> **❌ 没有任何一个策略能同时满足 acc ≥ 65% AND tpd ≥ 15。**

最接近的组合：
- Raw pred consensus top1.0% min4: tpd=2, acc=65.1% ← acc 达标，tpd 差太远
- Raw pred consensus top5.0% min4: tpd=16, acc=60.3% ← tpd 达标，acc 差 4.7pp

---

## 根本原因：AUC 天花板

```
H=15 全局 LGB te_AUC = 0.5427
     ↓
top1% acc 天花板 ≈ 60-62%
要 top1% acc ≥ 65% 需要 AUC ≥ 0.555
要 top1% acc ≥ 70% 需要 AUC ≥ 0.57
```

多 horizon 共识和 stacking 都没提升 AUC —— 它们提升的是 **信号覆盖度**（tpd），不是 **信号质量**（acc）。因为 4 个 horizon 的 base LGB AUC 都只在 0.539-0.544 之间。

---

## 突破方向（不需要 orderbook，但需要更激进的架构/特征）

| 方向 | 预期 AUC 提升 | 难度 | 状态 |
|------|---------------|------|------|
| **多任务联合训练**（同时预测所有 horizon） | +0.005-0.01 | 🟢 低 | ✅ 零新数据，改 loss 函数 |
| **Hour-specific MoE**（每个 UTC 小时一个专家模型） | +0.01-0.02 | 🟡 中 | ✅ 零新数据，已验证好时段 per-hour AUC 更高 |
| **Transformer/MLP 序列模型** | +0.01-0.03 | 🟠 高 | 需要 PyTorch，序列长度 96 |
| **CatBoost vs LGB ensemble** | +0.003-0.008 | 🟢 低 | ✅ CatBoost 默认处理 categorical |
| **更长训练数据**（加入 2018-2020） | +0.002-0.005 | 🟢 低 | 扩大训练集 |
| **换 label 定义**（soft label / regress ret） | +0.001-0.005 | 🟢 低 | 训练目标更平滑 |
| **更丰富的特征工程**（volume profile proxy, price-of-change） | +0.005-0.01 | 🟡 中 | 不用新数据，只是重新组织 |

---

## 可落地的最佳方案（明日可继续）

```
best_pipeline = [
    Step 1: 训 H=15 CatBoost (默认处理 categorical, num_leaves=127),    # CatBoost AUC 可能更高
    Step 2: 训 per-hour H=15 CatBoost (24 个),                             # 好时段信号更强
    Step 3: consensus (CatBoost H=3 + H=5 + H=15 + H=30 preds),            # Raw pred cutoff
    Step 4: stacking (4 horizon CatBoost preds 作为 meta-LGB 输入),        # 多样性更强
]
```

预计 CatBoost 比 LGB AUC 高 0.005-0.01 → 可能突破 top1% 62-63%。
如果加 per-hour + consensus，好时段里 acc 可能到 63-65%。
最终 stacking 后也许 65%+。

---

## 技术债

| 文件 | 状态 | 问题 |
|------|------|------|
| `mh_experiments/mh_union.py` | ✅ 可用 | Union 去重逻辑正确 (raw ts 去重) |
| `mh_experiments/per_hour.py` | ✅ 可用 | 需要更好的 cross-validation |
| `mh_experiments/union_all.py` | ❌ 崩溃 | `float32` 不能 json serialize，加 `astype(float)` 即可 |
| `mh_experiments/final_consensus` | ✅ 完成 | Raw pred cutoff 结果可信 |
| `data/datasets/ds_ETH_h15.parquet` | ✅ 正确 | base + BTC cross + funding (部分) |

---

_所有数据在 test set (2025-09-30 ~ 2026-08-29) 上评估，label = ETH 15 分钟后方向，数据源 = 1min OHLCV + buy/sell_vol + funding + BTC cross-asset。_
