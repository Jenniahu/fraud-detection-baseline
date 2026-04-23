"""
run_experiment.py
=================
主实验入口脚本

用法:
  # 运行全部实验（4 策略 × 3 模型 = 12 组）
  python run_experiment.py

  # 只运行特定策略
  python run_experiment.py --strategies baseline hybrid

  # 只运行特定模型
  python run_experiment.py --models logistic_regression xgboost

  # 自定义数据路径
  python run_experiment.py --data_path data/creditcard.csv

  # 静默模式（不打印逐条详情）
  python run_experiment.py --quiet
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import (
    RESULTS_DIR, FIGURES_DIR, DEFAULT_STRATEGIES, DEFAULT_MODELS,
    RANDOM_SEED, OPTIMIZE_THRESHOLD, THRESHOLD_METRIC, VAL_SPLIT_RATIO,
    THRESHOLD_CACHE_FILE, RECALCULATE_THRESHOLD,
)
from src.threshold_cache import (
    _load_threshold_cache, _get_cached_threshold, _cache_threshold,
)
from src.results_io import _save_results, _print_summary
from src.experiment import run_all_experiments
from src.data_loader  import load_data
from src.preprocessing import preprocess
from src.resampling   import get_resampled, list_strategies
from src.models       import build_model, list_models
from src.evaluation   import evaluate_model, find_optimal_threshold, get_model_proba
from src.visualization import (
    plot_pr_curves,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_class_distribution,
    plot_resampling_comparison,
)

# ─────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="欺诈检测 Baseline 实验框架"
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=list_strategies(),
        help="要运行的重采样策略（默认全部）",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS,
        choices=list_models(),
        help="要运行的分类器（默认全部）",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="CSV 数据文件路径（默认 data/creditcard.csv）",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式，减少输出",
    )
    parser.add_argument(
        "--no_threshold_opt", action="store_true",
        help="禁用阈值优化，使用固定阈值 0.5（用于对比实验）",
    )
    parser.add_argument(
        "--threshold_metric", type=str, default="f1",
        choices=["f1", "g_mean", "precision", "recall"],
        help="阈值优化目标指标（默认 f1）",
    )
    parser.add_argument(
        "--recalculate_threshold", action="store_true",
        help="强制重新计算阈值（忽略缓存，用于更新阈值）",
    )
    parser.add_argument(
        "--clear_threshold_cache", action="store_true",
        help="清空阈值缓存文件",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 处理清空缓存请求
    if args.clear_threshold_cache:
        if os.path.exists(THRESHOLD_CACHE_FILE):
            os.remove(THRESHOLD_CACHE_FILE)
            print(f"[INFO] 已清空阈值缓存: {THRESHOLD_CACHE_FILE}")
        else:
            print(f"[INFO] 阈值缓存文件不存在: {THRESHOLD_CACHE_FILE}")

    # 根据命令行参数更新阈值优化配置
    optimize_threshold = not args.no_threshold_opt
    threshold_metric = args.threshold_metric
    recalculate = args.recalculate_threshold

    # 临时修改全局配置（用于传递给 run_all_experiments）
    import run_experiment as re_module
    re_module.OPTIMIZE_THRESHOLD = optimize_threshold
    re_module.THRESHOLD_METRIC = threshold_metric
    re_module.RECALCULATE_THRESHOLD = recalculate

    results = run_all_experiments(
        strategies=args.strategies,
        models=args.models,
        data_path=args.data_path,
        verbose=not args.quiet,
    )
