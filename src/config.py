"""
config.py
=========
集中管理项目路径、默认配置和常量。
"""

import os

# ─────────────────────────────────────────────────────────────────
# 项目根目录
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────
# 数据路径
# ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
CSV_PATH   = os.path.join(DATA_DIR, "creditcard.csv")
KAGGLE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# ─────────────────────────────────────────────────────────────────
# 输出路径
# ─────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 数据列定义
# ─────────────────────────────────────────────────────────────────
TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]

# ─────────────────────────────────────────────────────────────────
# 实验默认配置
# ─────────────────────────────────────────────────────────────────
DEFAULT_STRATEGIES = ["baseline", "oversample", "undersample", "hybrid"]
DEFAULT_MODELS     = ["logistic_regression", "random_forest", "xgboost"]
RANDOM_SEED        = 42

# ─────────────────────────────────────────────────────────────────
# 阈值优化配置
# ─────────────────────────────────────────────────────────────────
OPTIMIZE_THRESHOLD    = True
THRESHOLD_METRIC      = "f1"
VAL_SPLIT_RATIO       = 0.2
THRESHOLD_CACHE_FILE  = os.path.join(RESULTS_DIR, "optimal_thresholds.json")
RECALCULATE_THRESHOLD = False

# ─────────────────────────────────────────────────────────────────
# 可视化配置
# ─────────────────────────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
