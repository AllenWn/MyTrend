# config.py v2.0
# 核心更新：支持动态特征权重学习，明确标签与特征边界
import os

# 数据配置
FEATURE_COLS = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "ma5",
    "ma10",
    "vol5",
    "vol10",
    "rsi",
    "bb_upper",
    "bb_lower",
]  # 仅含历史特征
INPUT_WINDOW = 20  # 输入窗口（前20天历史数据）
OUTPUT_WINDOW = 5  # 预测窗口（未来5天涨幅）
FUTURE_RETURN_THRESHOLD = 1.15  # 未来5天涨幅≥15%则标签为1（训练目标）
DERIVED_FEATURE_DAYS = 20  # 技术指标计算所需最大天数
INITIAL_START_DATE = "2023-01-01"
INITIAL_END_DATE = "2024-01-01"  # 或其他日期格式

# 特征筛选配置
MIN_SUPPORT = 0.3  # 正样本中最小支持度
MAX_COMBINATION_LEN = 3  # 最大组合长度
NEG_THRESHOLD = 0.3  # 负样本中最大允许出现频率

# 模型配置
INPUT_SIZE = len(FEATURE_COLS)
HIDDEN_SIZE = 64
NUM_CLASSES = 2  # 0: 未达15%，1: 达15%
KEY_FEATURE_IDX = 1  # 初始关注特征（如close，仅静态模式生效）
FEATURE_WEIGHT_MODE = "dynamic"  # "dynamic"（自动学习权重）或 "static"（固定权重）
INITIAL_KEY_WEIGHT = 1.15  # 初始权重（仅静态模式生效）

# 训练配置
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
PATIENCE = 3  # 早停耐心值
MIN_IMPROVEMENT = 0.001  # 最小提升阈值
MAX_GRAD_NORM = 1.0  # 梯度裁剪阈值

# 路径配置
CACHE_DIR = "stock_data/cache"
SCALER_DIR = os.path.join(CACHE_DIR, "scalers")
MODEL_DIR = os.path.join(CACHE_DIR, "models")
METRICS_DIR = "metrics"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"best_model_input{INPUT_SIZE}.pth")
METRICS_PATH = os.path.join(METRICS_DIR, f"metrics_input{INPUT_SIZE}.json")
FEATURE_COMBINATION_PATH = os.path.join(METRICS_DIR, "feature_patterns.json")
