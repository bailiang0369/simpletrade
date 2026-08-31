"""全局配置：数据路径、任务参数、严格样本外切分、验收阈值。"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据缓存目录（取回的 raw parquet + 特征数据集；bn 数据缓存目录见 fetch_data.BN_CACHE）
# 默认放项目内 data/，可用环境变量 BN_DATA_DIR 覆盖（沙箱开发时指向大容量临时盘）
DATA_DIR = os.environ.get("BN_DATA_DIR", os.path.join(PROJECT_DIR, "data"))
DS_DIR = os.path.join(DATA_DIR, "datasets")
MODEL_DIR = os.path.join(PROJECT_DIR, "models_saved")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
for _d in (DATA_DIR, DS_DIR, MODEL_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- 交易对（短名, 与 bn_data 发布数据命名一致, 无 USDT 后缀）----
# fetch_data 会从 GitHub Release 拉取这些币种的合并数据并落地 raw_{SYM}.parquet。
SYMBOLS = ["BTC", "ETH"]

# ---- 预测任务 ----
HORIZON_MIN = 30        # 预测未来多少分钟之后的涨跌 (H=30 经稳定性验证, test top1% 62.5%)
LOOKBACK_MIN = 60       # 图形/序列模型使用的回看窗口长度

# ---- 验收目标（不可私自放宽）----
COVERAGE = 0.01                     # 覆盖率 1%
TARGET_ACCURACY = 0.65              # top1% 准确率 >= 65%
MIN_TRADES_PER_DAY = 14             # 每天交易次数 >= 14
CROSS_ASSET_MAX_DELTA = 0.03        # BTC/ETH 之间 top1% 准确率相差 <= 3pp

# ---- 严格样本外切分 ----
TRAIN_END = "2024-06-30"        # 训练集截止
ES_END = "2024-09-30"           # GBDT/DL 早停验证集截止
META_VAL_END = "2025-09-30"     # 堆叠meta训练 + 置信度阈值选择（模型完全未见）
# TEST: META_VAL_END 之后 至 最新数据（严格样本外，任何环节都不允许触碰）

# 时间切分常量（字符串便于日志）
SPLITS = {
    "train": ("2020-01-01", TRAIN_END),
    "early_stop": (TRAIN_END, ES_END),
    "meta_val": (ES_END, META_VAL_END),
    "test": (META_VAL_END, "2099-12-31"),
}

# ---- 计算资源 ----
N_JOBS = 3
N_DL_SAMPLES = 400_000          # 深度模型训练抽样规模（过快、尽量覆盖多种状态）
CNN_SAMPLES = 12_000            # CNN 图像内存占用大(HxWx3x4B/样本), 单独限制(4GB cgroup 下需克制)
CNN_ES_SAMPLES = 4_000
DL_EPOCHS = 5
BATCH_SIZE = 1024
GRU_HIDDEN = 128
PATTERN_TEMPLATES = 4000        # 模式识别模板数量
DTW_TEMPLATES = 3000            # DTW 模板数量
DTW_CAND = 8                    # FAISS 预筛后做 DTW 的近邻数
DTW_WINDOW = 20                 # DTW 使用的降采样窗口长度

# ---- 变长图形匹配(时间归一化) ----
# 多尺度窗口: 同一形态在30根/40根等不同长度上完成时, 先重采样到固定N点再匹配
TN_WINDOWS = [16, 32, 48, 64, 96, 128]   # 候选窗口尺度(分钟)
TN_N = 24                                 # 重采样后的固定点数(时间归一化)
TN_TEMPLATES = 90_000                     # 每个尺度的模板数
TN_K = 64                                 # 每尺度近邻数
TN_NLIST = 256
TN_NPROBE = 24
# DTW 变长模板尺度
DTW_WINDOWS = [48, 96]

# ---- 随机种子 ----
SEED = 42
