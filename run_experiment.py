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
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader  import load_data, preprocess
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
# 实验配置
# ─────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEFAULT_STRATEGIES = ["baseline", "oversample", "undersample", "hybrid"]
# 策略说明:
#   baseline    : 原始不平衡数据
#   oversample  : SMOTE 过采样
#   undersample : SMOTE-ENN（先 SMOTE 平衡，再 ENN 清理）
#   hybrid      : SMOTE-Tomek（Proposal 核心研究对象）
DEFAULT_MODELS     = ["logistic_regression", "random_forest", "xgboost"]
RANDOM_SEED        = 42

# 阈值优化配置
OPTIMIZE_THRESHOLD = True          # 是否启用阈值优化
THRESHOLD_METRIC   = "f1"          # 优化目标: f1 | g_mean | precision | recall
VAL_SPLIT_RATIO    = 0.2           # 从训练集划分验证集的比例

# 阈值缓存配置
THRESHOLD_CACHE_FILE = os.path.join(RESULTS_DIR, "optimal_thresholds.json")
RECALCULATE_THRESHOLD = False      # 是否强制重新计算阈值（忽略缓存）

# ─────────────────────────────────────────────────────────────────
# 阈值缓存管理
# ─────────────────────────────────────────────────────────────────
def _load_threshold_cache(cache_file: str = THRESHOLD_CACHE_FILE) -> dict:
    """
    加载阈值缓存文件

    Returns
    -------
    dict: { (strategy, model, metric): threshold, ... }
    """
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        # 转换 key 从字符串元组格式
        return {tuple(k.split('|')): v for k, v in cache.items()}
    except Exception:
        return {}


def _save_threshold_cache(cache: dict, cache_file: str = THRESHOLD_CACHE_FILE):
    """
    保存阈值缓存到文件

    Parameters
    ----------
    cache : dict
        { (strategy, model, metric): threshold, ... }
    """
    # 转换 key 为可序列化的字符串格式
    serializable_cache = {'|'.join(k): v for k, v in cache.items()}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_cache, f, indent=2, ensure_ascii=False)


def _get_cached_threshold(strategy: str, model: str, metric: str,
                          cache: dict = None) -> float:
    """
    从缓存中获取阈值

    Returns
    -------
    float or None: 如果缓存存在返回阈值，否则返回 None
    """
    if cache is None:
        cache = _load_threshold_cache()
    key = (strategy, model, metric)
    return cache.get(key)


def _cache_threshold(strategy: str, model: str, metric: str,
                     threshold: float, cache: dict = None):
    """
    将阈值存入缓存并保存到文件
    """
    if cache is None:
        cache = _load_threshold_cache()
    key = (strategy, model, metric)
    cache[key] = round(threshold, 6)
    _save_threshold_cache(cache)


# ─────────────────────────────────────────────────────────────────
# 核心实验流程
# ─────────────────────────────────────────────────────────────────
def run_single_experiment(strategy_name: str,
                           model_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           verbose: bool = True,
                           optimize_threshold: bool = OPTIMIZE_THRESHOLD,
                           threshold_metric: str = THRESHOLD_METRIC) -> dict:
    """
    运行单组实验: 一种重采样策略 + 一种分类器

    Parameters
    ----------
    optimize_threshold : bool
        是否启用阈值优化（从训练集划分验证集搜索最优阈值）
    threshold_metric : str
        阈值优化目标指标: f1 | g_mean | precision | recall

    Returns
    -------
    dict 包含 strategy, model, 各项指标, 训练时间, y_proba
    """
    from sklearn.model_selection import train_test_split

    label = f"[{strategy_name.upper()} × {model_name}]"
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {label}")
        print(f"{'─'*55}")

    # 1. 重采样
    t0 = time.time()
    X_res, y_res = get_resampled(
        strategy_name, X_train, y_train,
        random_state=RANDOM_SEED
    )
    resample_time = time.time() - t0

    # 2. 划分训练/验证集（用于阈值优化）
    if optimize_threshold:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_res, y_res,
            test_size=VAL_SPLIT_RATIO,
            stratify=y_res,
            random_state=RANDOM_SEED,
        )
    else:
        X_tr, y_tr = X_res, y_res
        X_val, y_val = None, None

    # 3. 构建 & 训练模型
    model = build_model(model_name)
    t1 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t1

    # 4. 阈值优化（使用缓存或搜索）
    best_threshold = 0.5
    cache = _load_threshold_cache()
    cached_thresh = _get_cached_threshold(strategy_name, model_name,
                                          threshold_metric, cache)

    if optimize_threshold and X_val is not None:
        if cached_thresh is not None and not RECALCULATE_THRESHOLD:
            # 使用缓存的阈值
            best_threshold = cached_thresh
            if verbose:
                print(f"  💾 阈值缓存: 使用已保存的阈值={best_threshold:.3f} "
                      f"({strategy_name}|{model_name}|{threshold_metric})")
        else:
            # 搜索最优阈值
            y_val_proba = get_model_proba(model, X_val)
            best_threshold, best_val_score, _ = find_optimal_threshold(
                y_val, y_val_proba, metric=threshold_metric
            )
            # 保存到缓存
            _cache_threshold(strategy_name, model_name, threshold_metric,
                            best_threshold, cache)
            if verbose:
                source = "重新计算" if RECALCULATE_THRESHOLD else "首次计算"
                print(f"  🔧 阈值优化: 最优阈值={best_threshold:.3f} "
                      f"(验证集 {threshold_metric.upper()}={best_val_score:.4f}) [{source}]")

    # 5. 在测试集上评估（使用最优阈值）
    metrics = evaluate_model(model, X_test, y_test,
                             threshold=best_threshold, verbose=verbose)

    # 6. 保存预测概率（用于后续绘图）
    y_proba = get_model_proba(model, X_test)

    result = {
        "strategy"       : strategy_name,
        "model"          : model_name,
        "resample_time"  : round(resample_time, 3),
        "train_time"     : round(train_time,    3),
        "threshold"      : round(best_threshold, 3),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "_y_proba"       : y_proba,
        "_cm"            : metrics["confusion_matrix"],
    }

    if verbose:
        opt_info = "(优化后)" if optimize_threshold else "(固定 0.5)"
        print(f"\n  ⏱  重采样: {resample_time:.2f}s | 训练: {train_time:.2f}s "
              f"| 阈值{opt_info}: {best_threshold:.3f}")

    return result


def run_all_experiments(strategies: list,
                         models: list,
                         data_path: str = None,
                         verbose: bool = True) -> pd.DataFrame:
    """
    运行全矩阵实验并返回汇总 DataFrame

    Parameters
    ----------
    strategies : list  重采样策略名列表
    models     : list  模型名列表
    data_path  : str   可选，自定义 CSV 路径
    verbose    : bool

    Returns
    -------
    pd.DataFrame  每行为一组实验的所有指标
    """
    # ── 加载与预处理数据 ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  🚀 欺诈检测 Baseline 实验框架")
    print("  Hybrid Resampling for Imbalanced Financial Fraud Detection")
    print("═" * 60)

    df = load_data(csv_path=data_path or _default_data_path(),
                   verbose=verbose)
    X_train, X_test, y_train, y_test = preprocess(df, verbose=verbose)

    # ── EDA 可视化 ────────────────────────────────────────────────
    print("\n[Step 1] 绘制原始类别分布图 …")
    plot_class_distribution(df["Class"].values, save_name="01_class_distribution.png")

    # ── 重采样分布统计 ────────────────────────────────────────────
    print("[Step 2] 统计重采样分布 …")
    strategy_counts = {}
    from collections import Counter
    for strat in strategies:
        _, y_r = get_resampled(strat, X_train, y_train,
                               random_state=RANDOM_SEED)
        c = Counter(y_r)
        strategy_counts[strat] = {"normal": c[0], "fraud": c[1]}
    plot_resampling_comparison(strategy_counts,
                               save_name="02_resampling_comparison.png")

    # ── 运行全矩阵实验 ────────────────────────────────────────────
    print(f"\n[Step 3] 开始全矩阵实验 "
          f"({len(strategies)} 策略 × {len(models)} 模型) …")

    all_results = []
    probas_by_strategy = {s: {} for s in strategies}

    for strategy in strategies:
        for model_name in models:
            result = run_single_experiment(
                strategy, model_name,
                X_train, y_train, X_test, y_test,
                verbose=verbose,
            )
            probas_by_strategy[strategy][model_name] = result.pop("_y_proba")
            _ = result.pop("_cm")
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # ── 可视化：PR / ROC 曲线 ─────────────────────────────────────
    print("\n[Step 4] 绘制 PR / ROC 曲线 …")
    for strategy in strategies:
        plot_pr_curves(
            probas_by_strategy[strategy], y_test,
            title=f"PR Curves — {strategy}",
            save_name=f"03_pr_{strategy}.png",
        )
        plot_roc_curves(
            probas_by_strategy[strategy], y_test,
            title=f"ROC Curves — {strategy}",
            save_name=f"04_roc_{strategy}.png",
        )

    # ── 可视化：全局指标对比 ──────────────────────────────────────
    print("[Step 5] 绘制指标对比图 …")
    plot_metrics_comparison(
        results_df,
        metrics=["recall", "precision", "f1", "g_mean", "auprc"],
        title="All Strategies × Models — Performance Comparison",
        save_name="05_metrics_comparison.png",
    )

    # ── 保存结果 ──────────────────────────────────────────────────
    print("\n[Step 6] 保存实验结果 …")
    _save_results(results_df)

    # ── 打印汇总表 ────────────────────────────────────────────────
    _print_summary(results_df)

    return results_df


# ─────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────
def _default_data_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "creditcard.csv")


def _save_results(df: pd.DataFrame):
    """将结果保存为 CSV 和 JSON"""
    csv_path  = os.path.join(RESULTS_DIR, "experiment_results.csv")
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")

    # CSV（不含内部列）
    save_cols = [c for c in df.columns if not c.startswith("_")]
    df[save_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  ✅ CSV  → {csv_path}")

    # JSON（便于程序读取）
    records = df[save_cols].to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON → {json_path}")


def _print_summary(df: pd.DataFrame):
    """打印实验汇总表"""
    print("\n" + "═" * 85)
    print("  实验汇总（按 AUPRC 降序排列）")
    print("═" * 85)
    display_cols = ["strategy", "model", "threshold", "recall", "precision",
                    "f1", "g_mean", "auprc", "auroc"]
    summary = df[display_cols].sort_values("auprc", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("═" * 85)

    # 最优组合
    best = summary.iloc[0]
    print(f"\n  🏆 最优组合 (AUPRC): "
          f"{best['strategy']} × {best['model']}  "
          f"阈值={best['threshold']:.3f}  "
          f"AUPRC={best['auprc']:.4f}  Recall={best['recall']:.4f}")

    # 阈值优化效果对比（如果有对比数据）
    if OPTIMIZE_THRESHOLD:
        print(f"\n  📊 阈值优化说明:")
        print(f"     每个 (策略, 模型) 组合在验证集上独立搜索最优阈值")
        print(f"     优化目标: {THRESHOLD_METRIC.upper()}")
        print(f"     确保各策略都处于最优决策点后再进行公平比较")


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
