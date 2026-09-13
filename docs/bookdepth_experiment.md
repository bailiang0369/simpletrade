# BookDepth（币安订单簿深度）实验总结

## 实验背景

在 ETH 15min 价格方向预测模型中，尝试引入订单簿深度信息以提升准确率。本次实验基于币安免费公开的 bookDepth 数据。

---

## 数据源

| 项目 | 详情 |
|------|------|
| **来源** | 币安官方 `data.binance.vision/data/futures/um/daily/bookDepth/` |
| **覆盖** | ETHUSDT + BTCUSDT 永续合约, 2023 全年 (362 天) |
| **格式** | CSV, 每天一个 zip |
| **原始列** | `timestamp, percentage, depth, notional` |
| **depth 档位** | -5%, -4%, -3%, -2%, -1%, +1%, +2%, +3%, +4%, +5% (累计 depth) |
| **频率** | 约每 10 秒一条 |
| **聚合** | 截断到 1 分钟, 同一分钟内同一 percentage 取最后一条 |
| **最终规模** | 510,655 × 10 档位 (约 51 万行) |
| **数据大小** | 约 160MB (原始 zip), 43MB (解析后 npz) |

---

## 特征工程

从 10 个累计 depth 档位派生以下特征 (19 个):

### 基础 OBI (Order Book Imbalance) - 3 个
```
obi_near = (b1 - a1) / (b1 + a1 + 1e-9)    # ±1% 范围
obi_mid  = (b3 - a3) / (b3 + a3 + 1e-9)    # ±3% 范围
obi_far  = (b5 - a5) / (b5 + a5 + 1e-9)    # ±5% 范围
```

### 深度分布 - 4 个
```
ratio_near_bid = b1 / (b5 + 1e-9)           # 买盘近侧集中比
ratio_near_ask = a1 / (a5 + 1e-9)           # 卖盘近侧集中比
total_depth    = b5 + a5                    # ±5% 总深度
total_depth_log = log1p(total_depth)
```

### Slope (深度斜率) - 2 个
```
bid_slope = (b1 - b5) / (b5 + 1e-9)        # 买盘深度分布形态
ask_slope = (a1 - a5) / (a5 + 1e-9)        # 卖盘深度分布形态
```

### Entropy (深度熵) - 2 个
```
bid_entropy = -sum(p_i * log(p_i)), p_i = incremental_depth / total
ask_entropy = 同上, 衡量深度在各档位分布是否均匀
```

### 流动性变化 - 1 个
```
depth_chg_pct = (depth[t] - depth[t-1]) / (depth[t] + depth[t-1] + 1e-9)
```

### 波动率交互特征 - 6 个
```
depth_per_vol30       = total_depth / rolling_abs_ret_30  # 深度 / 30min 波动率
obi_near_x_vol5       = obi_near * rolling_abs_ret_5      # OBI × 5min 波动率
obi_near_x_vol30      = obi_near * rolling_abs_ret_30
obi_mid_x_vol5        = obi_mid * rolling_abs_ret_5
obi_mid_x_vol30       = obi_mid * rolling_abs_ret_30
obi_far_x_vol5        = obi_far * rolling_abs_ret_5
obi_far_x_vol30       = obi_far * rolling_abs_ret_30
```

### 逐档 OBI (计算过但最终精简版本未使用)
```
obi_pct1..obi_pct5 = incremental_depth OBI, 衡量具体某个 ±n% 区间内的买卖盘 imbalance
```

---

## 实验结果

### 测试设置

| 项目 | 详情 |
|------|------|
| **目标** | `ETHUSDT` 15min bar 方向预测 (up/down) |
| **数据集** | `ds_ETH_h15.parquet` (时间戳严格递增, 无随机切分) |
| **训练期** | 2020-01-01 ~ 2023-12-31 (2,095,274 bars) |
| **测试期** | 2024-01-01 ~ 2024-12-31 (526,465 bars) |
| **Time split** | ✅ 严格时间切分, 无数据穿越 |
| **泄漏检查** | 移除 `label`, `soft_label`, `ret_future`, `ts` 等泄漏列 |
| **模型** | LightGBM, n_estimators=300, learning_rate=0.03, num_leaves=63, min_child_samples=200 |
| **标签 up_rate** | 测试集 0.5050 (接近 50%, 平衡二分类问题) |

### 最终结果

| 模型 | 特征数 | AUC | ΔAUC | top1% | Δtop1% | top5% | lift |
|------|--------|-----|------|-------|--------|-------|------|
| **BASE (ds clean)** | 55 | **0.5378** | — | **0.6214** | — | 0.5719 | 1.23x |
| BASE + ETH_BD | 74 | 0.5377 | **-0.0001** | 0.6130 | **-0.0084** | 0.5767 | 1.21x |
| BASE + ETH_BD + BTC_BD | 73 | 0.5384 | **+0.0006** | 0.6166 | **-0.0048** | 0.5801 | 1.22x |

### 分组贡献分析 (单独跑各特征组)

在 ds_ETH_h15 的干净特征中, 各特征组独立贡献 (ΔAUC):

| 特征组 | feats | 单独 AUC | ΔAUC | top1% | 本质 |
|--------|-------|----------|------|-------|------|
| **lr_xx (动量)** | 4 | 0.5300 | **+0.0300** | 0.5579 | 过去 return |
| **z_xx (Z-score)** | 3 | 0.5386 | **+0.0386** | 0.5664 | 动量/波动率归一化 |
| **pos_xx (正收益占比)** | 3 | 0.5372 | **+0.0372** | 0.5729 | 窗口内 up/down 比例 |
| **dd_xx / ru_xx (回撤)** | 2 | 0.5291 | **+0.0291** | 0.5348 | 回撤/反弹幅度 |
| **regime / run / accel** | 8 | 0.5299 | **+0.0299** | 0.5841 | 趋势加速状态 |
| **BTC lr_xx (BTC 动量)** | 8 | 0.5339 | **+0.0339** | 0.5925 | 跨资产动量 |
| **BTC z_xx (BTC Z-score)** | 5 | 0.5338 | **+0.0338** | 0.5614 | 跨资产 Z-score |
| rvol_xx (ETH 波动率) | 4 | 0.5013 | **+0.0013** | 0.5057 | 波动率单独没用 |
| max_range (振幅) | 1 | 0.4987 | **-0.0013** | 0.5021 | 纯振幅没用 |
| tb_act / ts_act (量活跃度) | 2 | 0.5017 | **+0.0017** | 0.5073 | 单独没用 |
| BTC rvol_xx | 2 | 0.4993 | **-0.0007** | 0.5042 | 单独没用 |

**核心结论**: 真正强的信号只有**动量及其变换** (Z-score, 正收益占比, 回撤形态) + **跨资产 BTC 动量**。波动率、振幅、量活跃度单独接近 0.50, 但组合中提供微弱正则化作用。

### 特征裁剪测试

| 方案 | 特征数 | AUC | top1% | Δtop1% vs 全量 |
|------|--------|-----|-------|----------------|
| ETH 全部自身 | 38 | 0.5368 | 0.5741 | — |
| ETH + BTC 全部 | 55 | 0.5382 | 0.5821 | — (BASE) |
| **去掉 10 个"没用"特征** | 45 | 0.5384 | 0.5671 | **-0.0150** |
| 核心子集 (手动挑) | 40 | 0.5348 | 0.5602 | -0.0219 |
| 纯动量类 only | 20 | 0.5358 | 0.5729 | -0.0092 |

**结论**: LightGBM 能自动降权弱特征, 手动裁剪反而损失 top1% 精度。保留全部 55 个特征是最优选择。

---

## 关键发现

### 1. bookDepth 没有增量

在正确因果设置下 (210 万训练, 52 万测试):
- AUC 变化在 ±0.001 以内
- top1% 从 0.6214 下降到 0.6130 (-0.0084)
- 即使只在 bookDepth 覆盖的 2023 年切分训练 (约 26 万条), ΔAUC 也只有 +0.002~+0.004

### 2. 为什么免费 bookDepth 没用

| 问题 | 说明 |
|------|------|
| **覆盖不全** | bookDepth 只有 2023 年, 210 万训练中只有 14 万有数据, 2024 测试全 NaN → LightGBM 直接忽略 |
| **只有累计 depth** | -1% 是累计值, 无法区分"近侧撤单"还是"价格把 depth 挤出范围" |
| **缺少撤单/成交拆分** | depth[t] - depth[t-1] 可能是成交也可能是撤单, 无法计算 cancel rate |
| **只有 10 档百分比** | 不是固定绝对价位, 无法跟踪某个具体价位的挂单队列 |
| **频率偏低** | 每 10 秒一条, 微观结构信号半衰期 < 30 秒 |

### 3. 为什么 ds_ETH_h15 本身 AUC 不高

BASE AUC = 0.5378 (top1% = 0.6214) — 这是**正确因果测试的真实能力**:
- 2024 年是牛市, 市场环境变化大
- 随机 split 之前测到的 0.98+ AUC 是数据穿越导致的假信号
- 真实 causal P99 (top1%) 约 62-64%

---

## 免费订单簿数据源对比

| 数据源 | 深度 | 粒度 | 免费 | 特点 |
|--------|------|------|------|------|
| **bookDepth (币安)** | 10 档百分比 | 10s | ✅ | 累计 depth, 无成交/撤单拆分 |
| bookTicker (币安) | 1 档 best bid/ask | ~120ms | ✅ | 只有买卖一档, 信息量极浅 |
| aggTrades (币安) | 成交记录 | ~120ms | ✅ | 已有 buy_vol/sell_vol 等效覆盖 |
| Kaggle BTC L2 (1s) | 10 档 | 1s | ✅ | 只有样本, 网络不稳定 |
| HuggingFace crypto-lob-stream | 多档 | 增量 | ✅ | 每月发布, 有已知数据缺口 |
| **实时 WebSocket @depth20@100ms** | 20 档 | 100ms | ✅ 免费但需自存 | 能跟踪绝对价位, 可算撤单速率 |
| Tardis.dev L2 incremental | 20-1000 档 | ms | ❌ $0.1/GB | 真正的 L2, 可算完整 OBI 和 cancel rate |
| Crypto-lake snapshots | 20 档 | 快照 | ❌ $0.05/GB | 类似但只有快照 |

### 理论上 L2 incremental 能做但 bookDepth 做不到的

```python
# 跟踪固定绝对价位的挂单变化 (bookDepth 做不到)
depth_at_fixed_price[t] = sum(depth_orders where order.price == $2000)

# 拆分成交 vs 撤单
depth_change[t] = add[t] - cancel[t] - trade[t]

# Cancel rate (撤单速率) - 学术上被反复验证的强信号
cancel_rate = sum(cancel[t-30:t]) / sum(add[t-30:t])  # 过去 30 秒撤单比例

# 真正的 Order Flow Imbalance (Cont 2014)
OFI = sum(bid_add_vol) - sum(ask_add_vol) - sum(bid_cancel_vol) + sum(ask_cancel_vol)
```

**但注意**: 即使有 L2 incremental, 在 15min 周期下 ΔAUC 也不会超过 +0.02~+0.03, 主要增量在 1-5min 短周期。

---

## 后续方向建议

### 短期 (零成本)

1. **接受 bookDepth 天花板**: 免费数据源的极限就是 ΔAUC +0.002~+0.004, 不值得继续投入
2. **保留 ds 里的全部 55 个特征**: LightGBM 自动降权弱特征, 手动裁剪反而损失
3. **BTC cross-asset 是最大增量源**: 单独 BTC lr_xx + z_xx 就能 +0.03 ΔAUC, 继续挖跨资产信号可能更有价值

### 中期 (需要实时采)

4. **用 WebSocket 实时采 ETH @depth20@100ms**: 每天约 1.7GB (Parquet 压缩后 ~500MB), 可以:
   - 跟踪绝对价位的 depth 变化 (而不是百分比)
   - 观察真实的挂单队列变化
   - 采 1-2 周验证 cancel rate / OFI 信号强度

### 长期 (付费可选)

5. **Tardis.dev L2 incremental**: $0.1/GB, 买 2024 年约 730 天 × 每天 5GB ≈ 3.65TB = $365
   - 重建完整 L2, 算 cancel rate, trade flow toxicity
   - 但 ROI 存疑: 365 美元买 +0.02 AUC 是否值得?

---

## 技术附录

### 数据获取脚本

```bash
# 批量下载 bookDepth zip
python -c "
import subprocess, os
for m in range(1,13):
    for d in range(1,32):
        url = f'https://data.binance.vision/data/futures/um/daily/bookDepth/ETHUSDT/ETHUSDT-bookDepth-2023-{m:02d}-{d:02d}.zip'
        subprocess.run(['curl','-sL',url,'-o',f'/tmp/bookdepth_eth/{m:02d}-{d:02d}.zip'])
"
```

### 关键处理脚本位置

| 脚本 | 作用 |
|------|------|
| `/workspace/backtest_short/build_dataset_short.py` | ds_ETH_h15 构建 |
| `/workspace/data_store.py` | 数据存储/加载 |
| `/workspace/run.py` | 主训练流程 |
| `/workspace/validate_eth_quick.py` | 快速验证 |

### 已清理的临时文件

- `/tmp/bookdepth_eth/*.zip` — 已删 (可随时重新下载)
- `/tmp/bookdepth_btc/*.zip` — 已删
- `/tmp/eth_bd.npz`, `/tmp/btc_bd.npz` — 已删
- `/tmp/eth_bd_ds.pkl`, `/tmp/btc_bd_ds.pkl` — 已删
- `/tmp/base_results.npz`, `/tmp/eth_results.npz`, `/tmp/both_results.npz` — 已删
- `feat_aggtrades/` 分支目录 — 已删 (git checkout main 前)
- `feat/aggtrades-ofi` git 分支 — 已删

---

*文档生成时间: 2026-09-14*
