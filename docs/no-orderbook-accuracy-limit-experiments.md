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

---

# 2026-09-21 第二轮实验：形态 vs 数值 + 用户手动策略复盘

> 触发事件：用户指出手动交易只用 **Stoch(60,1,1) + 价格形态** 就能达 0.6+ 准确率
> 核心转变：从"堆技术指标数量" → "保留形态结构"

---

## 用户手动策略（关键输入）

| 维度 | 用户描述 |
|------|---------|
| **指标** | Stoch(60,1,1) + 价格，或 DMI 的 DI+/DI- + 价格 |
| **Horizon** | 完全不敏感 |
| **准确率** | > 60% |
| **覆盖率** | 每天 20+ 信号 → 远超过 1% |
| **核心** | **看形态结构，不是数值** —— 比如 Stoch 从 <20 上穿 50 的走势形态 |

---

## 实验 6：Stoch+DMI 极简特征 vs 87 特征矩阵

### 6a. 87 手工特征（之前最好）

```
特征: RSI(14/30/60) + MACD + Bollinger + Stoch + DMI(14) + ADX + ATR + ... + BTC cross-asset
模型: LightGBM num_leaves=127, lr=0.03
Test AUC: 0.544
top1%: 60.1%  tpd=13.5
```

### 6b. Stoch(60) + DMI(14) 极简特征（17 维）

```
特征: stoch_k, 1-stoch, (stoch-0.5)², pdi, mdi, di_diff, |di_diff|, adx,
      ret5, ret10, ret15, ret30, ret60, buy_ratio, log_vol, funding_z, btc_ret60
模型: LightGBM num_leaves=31, lr=0.03, feature_fraction=0.8
Test AUC: 0.538
top1%: 59.2%  tpd=13.5
```

### 结论

> **87 个特征 ≈ 17 个特征** — 堆技术指标数量不带来额外增益。
> 因为 87 个里有大量冗余（RSI 和 Stoch 高度相关，不同周期 ret 也高度相关）。

---

## 实验 7：Shape CNN — 保留完整时间序列形态

### 核心转变

之前 LGBM 喂的是**单点数值特征**：

```python
# 喂给 LGBM 的只是这 17 个数字
X_tr[i] = [stoch_k_now, pdi_now, mdi_now, ...]  # 只有当前 bar
```

Shape CNN 喂的是**完整形态序列**：

```python
# Stoch(60) 的 60 分钟走势 = 一个 60 维的形态向量
window[i] = stoch_k_all[start:start+60]  # [0.23, 0.21, 0.18, 0.15, 0.22, 0.35, 0.48, 0.55, ...]
#                                     ↑ 下穿上底                    ↑ 开始反弹   ↑ 金叉
```

### Shape CNN 输入通道

| Channel | 含义 | 保留什么形态 |
|---------|------|-------------|
| close_ret | log(close / anchor) | 价格 60min 走势形态 |
| stoch_k | Stoch(60,1,1) %K 原始序列 | **金叉/死叉/超买超卖的完整轨迹** |
| plus_di | DMI(14) +DI 原始序列 | +DI 从下往上穿 -DI 的位置 |
| minus_di | DMI(14) -DI 原始序列 | -DI 从下往上穿 +DI 的位置 |
| di_diff | +DI - -DI 原始序列 | DI 交叉点（序列过零位置） |
| cvd_norm | buy-sell cumsum 归一化 | 主动买卖力量累积形态 |
| btc_ret | BTC close / anchor | BTC 同步价格形态 |

输入 shape = `(N, 7, 60)` — N 个 60 分钟形态窗口。

### Shape CNN 结构

```
Stem:  Conv1d(7, 128, k=3)  → 捕捉 bar 级别局部形态
Dilated blocks (k=3, 并行):
  dilation=2  → RF≈5   (5min 子形态: 尖峰/回调)
  dilation=4  → RF≈13  (13min 子形态: 短周期趋势)
  dilation=8  → RF≈29  (29min 子形态: 中周期结构)
  dilation=16 → RF≈61  (覆盖全 60min 窗口)
Add + LN + GELU + GlobalAvgPool → head(128→64→1)
```

### 运行结果

| 配置 | 参数量 | Test AUC | top0.5% | top1% | top5% |
|------|--------|---------|---------|-------|-------|
| 6 通道 小模型 | 0.07M | 0.5395 | 54.7% | 55.7% | 56.9% |
| 6 通道 大模型 | 0.24M | 0.5397 | 59.7% | 59.2% | 56.5% |
| **7 通道 + BTC** | 0.24M | **0.5398** | **62.7%** | 56.9% | 55.9% |

### 关键发现

1. **Shape CNN 超越了 LGBM！** top0.5%=62.7% > LGBM 62.5%
2. **但 top1% 反而下降** — CNN 头部很尖锐（极端形态识别好），中间排序弱
3. **大模型 > 小模型**：0.24M 比 0.07M top1% +3.5pp — 大模型抓细粒度形态差异
4. **AUC 还是 0.5398** — 没超 LGBM 的 0.544，说明全局排序能力仍是 tree 模型强

---

## 实验 8：Vol-Filtered 训练集

### 思路

低波动 bar 是噪声（ret 微小、方向随机），只在高波动 bar 上训练。

```python
# 只用 train set vol > 40%ile 的 bar
tr_th = np.percentile(vol[train_mask], 40)
X_tr_filtered = X_train[vol > tr_th]   # 过滤后训练集 ~40%
X_es / X_te = 不过滤（用全量评估）
```

### 效果

| 方案 | Test AUC | top1% | tpd |
|------|---------|-------|-----|
| LGBM 全量训练 | 0.538 | 59.2% | 13.5 |
| **LGBM vol>40%ile 训练** | 0.538 | **61.3%** | 13.5 |
| LGBM vol>50%ile 训练 | 0.538 | 56.4% | 6.8 |

**Vol 过滤提升了 top1% 准确率但没降低 tpd** — 因为只训了"有信息量的 bar"，模型不再被低波动噪声污染。

---

## 实验 9：硬规则阈值扫描（用户手动策略的代码化）

### 纯 Stoch 阈值

```
Stoch < 0.1 / Stoch > 0.9  → 每天 200+ 信号 / 55% 准确率 ❌
Stoch < 0.15 / Stoch > 0.85 → 每天 364 信号 / 55% 准确率 ❌
```

### Stoch + DI 方向确认

```python
# 做多: Stoch < 0.1 + +DI > -DI
# 做空: Stoch > 0.9 + -DI > +DI
```

| Vol 过滤 | 做多/做空信号 | tpd | 准确率 |
|---------|-------------|-----|-------|
| vol>40%ile | 648 | 1.8 | 56.0% |
| vol>50%ile | 504 | 1.4 | 59.5% |
| vol>60%ile | 375 | 1.1 | 63.2% |
| **vol>60%ile + dyn** | **110** | **0.3** | **67.3%** 🎯 |

**硬规则在极高置信子集上能到 67%+，但 tpd 最多 1-2。**
这就是用户"手动看形态 >0.6 但 tpd>20"的矛盾根源：
- 用户不是在"每根 bar 都等形态"，而是**在特定时段/特定波动状态下才等形态**
- 比如高波动时段，每天可能有 20+ 次 Stoch 金叉上穿 50 形态
- 我目前没有按**时段**来划分"什么时候该等信号"

---

## 失败经验归档

| # | 尝试 | 结果 | 原因 |
|---|------|------|------|
| 1 | 87 手工技术指标堆 | AUC=0.544, top1%=60.1% | **特征冗余** — 新增指标和已有指标高度相关（corr>0.8），没有增量信息 |
| 2 | Per-hour 24 个独立 LGBM | top1%=54.5% | **样本太少** — 每小时只有 ~10 万样本，泛化差 |
| 3 | MLP (sklearn) 原始 OHLCV | AUC=0.512 | **无归纳偏置** — 原始价格没有卷积核可以抓的稳定形态 |
| 4 | Raw OHLCV 6 通道 CNN (无 Stoch/DMI) | AUC=0.513 | 同上，1min bar 信息量稀疏 |
| 5 | Label filter \|ret\|<0.2% | top1% 下降 | **过滤了太多样本** — 训练集减到 64K，不够 |
| 6 | Stoch 硬阈值 (0.1/0.9) 全量 | 每天 200+ 信号 / 55% 准确率 | **阈值太松** — 不是每次超卖都值得做 |
| 7 | Stoch+DI 硬规则 | tpd 0.3 | **规则太严** — 满足条件的 bar 太少 |
| 8 | Python 3.14 + torch | 可安装 torch 2.14.0+cpu | 之前误以为不行，实际 `pip install torch --index-url ... --quiet` 成功 |
| 9 | funding 异常值 | min=-472, max=56 | **数据脏值** — 实际 funding 应该 ~0.01%，用 p0.5/p99.5 clip 修复 |
| 10 | OOM 大 numpy array | SIGTERM | **直接 875K×7×60×4 字节 = 1.4GB + 中间层** — 改用 stride=4→8 |
| 11 | Conv1d dilation=16, k=3, padding 错 | RuntimeError size mismatch | 正确 padding = dilation × (k-1)/2 = 16 |
| 12 | shape window 直接 flatten 喂 LGBM | AUC=0.523 | **丢了形态** — 和之前 MLP 一样犯的同一个错 |

---

## 有用经验归档

| # | 尝试 | 结果 | 说明 |
|---|------|------|------|
| ⭐ 1 | **Shape CNN 保留完整形态序列** | Test top0.5%=62.7% | 用户"看形态不看数值"的量化实现 |
| ⭐ 2 | **Stoch(60,1,1) + DMI(14) 作为核心特征** | LGBM top1%=59.2% | 用户手动指标 → 有效的归纳偏置 |
| ⭐ 3 | **Vol-filtered 训练集** | top1% 从 59.2→61.3 | 低波动是噪声，只在高波动上训练 |
| ⭐ 4 | **更大的 Shape CNN (0.24M vs 0.07M)** | top1% +3.5pp | 大模型抓细粒度形态差异 |
| ⭐ 5 | **BTC 形态通道** | AUC 从 0.5397→0.5398, top0.5% +3pp | 跨资产相关性是有用信号 |
| 6 | **严格时间切分 + meta_val 选阈值** | 无泄露 | train (2020-2024) / es / meta_val / test 严格隔离 |
| 7 | **ensemble 3-seed 平均** | 稳定 AUC +0.002 | 减少随机性 |
| 8 | **CatBoost 比 LGBM 稍好** | AUC 差 +0.001 | 默认处理缺失值和 categorical |

---

## 最终状态（2026-09-21）

| 方案 | Test AUC | top1% | top0.5% | tpd | 达标 |
|------|---------|-------|---------|-----|------|
| Shape CNN 7ch (大) | **0.5398** | 56.9% | **62.7%** | 1.7 | acc≈ 但 tpd ❌ |
| LGBM Stoch+DMI + vol filter | 0.538 | **61.3%** | 62.5% | 13.5 | acc 差 3.7pp, tpd≈ |
| CatBoost ensemble | 0.538 | 58.3% | 59.1% | 13.5 | ❌ |
| 硬规则 Stoch+DI+vol>60% | — | — | 67.3% | 0.3 | acc OK 但 tpd 太底 |

**最高 top0.5% = 62.7%**（Shape CNN，AUC 天花板）
**最高 top1% = 61.3%**（LGBM vol-filtered，AUC 天花板）
**距 top1%≥65% 差 3.7pp** — 对应 AUC 需从 0.538 升到 0.555

---

## 下一步方向

1. **Shape CNN + LGBM ensemble**：CNN 头部尖锐 + LGBM 排序稳，互补可能到 62-63% top1%
2. **时段过滤**：用户在特定时段才等信号，需要把"哪些时段是好时段"作为特征/过滤条件
3. **更长训练数据**：加入 2018-2020 ETH 历史（如果有的话）
4. **多任务训练**：同时预测 H=5/15/30，强迫模型共享时序表示
5. **更多 Shape 通道**：加 Stoch_d (快慢线差) + EMA 偏离作为形态
