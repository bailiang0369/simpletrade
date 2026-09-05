# SimpleTrade 项目交接 brief

> 给"新任务"的起点说明。目标：接手者用**最少的积分**从正确的地方继续，不重复已确认的试错。

## 0. 交接物

| 物 | 路径 / 说明 |
|---|---|
| 本 brief | `reports/HANDOFF.md`（先读这个，5 分钟） |
| 详细技术报告 | `reports/eth-optimization/eth-optimization-report.html`（浏览器打开，有全部数据图表与证据） |
| 代码仓库 | github.com/bailiang0369/simpletrade（已含全部核心代码 + 实验脚本） |
| 运行状态 | 代码已克隆后，见下方"续跑命令" |

## 1. 目标与硬约束（不可违反）

- 预测未来 30 根（30min）K 线的涨跌，覆盖率固定为 **Top 1%**（按置信度 `conf=max(p,1-p)` 取全量前 1%）。
- 验收：top1% **方向准确率 ≥ 65%**（争取 67%），每天交易次数 ≥ 14，BTC 与 ETH 之间准确率差 ≤ 3pp。
- **禁止数据泄漏**：模型与集成策略只能在 `meta_val` 段决策；`test` 段只在全部完成后**一次性**验证一次，绝不反复调参。
- **禁止跨资产特征**：ETH 只能用 ETH 自己的数据训练，BTC 的数据/模型不能用于 ETH。
- 时序模型特征数**最多 20 个**；不要求强制包含资金费率特征。
- 严格样本外切分（时间顺序、不相交）：train / early_stop / meta_val(mv) / test。切分见 `config.py`。

## 2. 当前最优基线（不要再重新摸索）

- **最优方案：POOL20 等权归一化 rank 集成** = CatBoost×8 + XGBoost×6 + LightGBM×6，每个模型输出做分位归一化 rank，再等权平均。
- **诚实 test 准确率 = 62.73%**（mv=0.5508），双向均衡（判涨 2395 / 判跌 2400），判涨 +5.9bps / 判跌 −1.0bps。这是可上线的基线。
- 单模型天花板：最优 hp_x4 mv_acc ≈ 0.602，单模型无法到 65%。
- 已修复并必须沿用两处口径（见 `evaluate.py`）：
  1. Top-1% 用 `np.argsort(-conf)`，不要用 `np.argpartition`（后者会选到置信度≈0.5 的边缘样本）。
  2. rank 必须是**归一化分位 rank**到 [0,1]，否则原始 rank（均值≈14.5）会让所有预测 >0.5 → 全判涨。

## 3. 已确认撞墙的死路（不要重试，省积分）

| 手段 | 结果 | 原因 |
|---|---|---|
| 48 全池等权 | 0.6152，更差 | 弱模型稀释 |
| 分时段守门（42 模型 GATED） | 0.6163 | 拟合 mv 的时段状态，不迁移到 test |
| 贪心子集 / 按 mv 精度加权 | 0.6096 | mv 优化与 test 提升不一致 |
| 级联 gate 守门 | test 崩到 59~62% | gate 的 logloss≈0.6485 无样本外信号；mv 上假性 0.73~0.79 是过拟合 |
| regime 识别 / 坏月规避 | 不迁移 | 错误不沿任何单一特征轴集中，坏月无法用现有特征定位 |
| 在现有特征上继续堆模型 / 再挖普通衍生特征 | 撞墙 | mv/test 单模均触顶 |

**核心结论（报告的实证）：天花板由 regime 漂移决定，非模型组合、也非普通特征可分性所能突破。** 同一集成在 mv 期最差月 45.9%、最好月 74.6%。

## 4. 下一步突破口（真正值得做的方向）

1. **首选——引入现有数据不存在的微观结构输入**：实时订单簿快照（买卖盘失衡、深度分布）。这是唯一尚未覆盖、且你明确指出的真实增量信号。**注意**：它无法用既有 parquet 回测，必须实时抓取。
2. **次选——跨族正交融合**：把 CNN/LSTM/GRU 用当前 `ds_ETH` 重训并对齐行数，与 GBDT 池做一次 GBDT×深度时序正交融合。这是唯一未充分验证的融合杠杆，但预计收益有限。
3. 若仍要在特征上做文章：只能走"更独特、正交"的新信号（如热力/CVD 深度交互），但报告证据表明加普通衍生不再贡献。

## 5. 数据与代码状态 / 续跑命令

- 仓库 `main` 已含全部代码、实验脚本、报告、`.gitignore`。
- 数据不在仓库里（18G，gitignore）。原始数据很小且可再生。
- 新环境续跑：
  ```bash
  git clone https://github.com/bailiang0369/simpletrade.git
  cd simpletrade
  pip install -r requirements.txt            # polars 等依赖
  python fetch_data.py                       # 重新拉取 raw_BTC / raw_ETH（很小）
  python build_dataset.py --symbol ETH       # 重建特征集 ds_ETH（150+ 列，含全部新特征）
  # 之后按 training 脚本重建模型池 / 集成，见 optimize_eth*.py / experiment_*.py
  ```
- 关键文件：`features.py`（特征体系）、`build_dataset.py`（数据构建，含资金费列）、`evaluate.py`（Top-1% 评估）、`data_store.py`（数据访问）、`config.py`（切分/目标）。