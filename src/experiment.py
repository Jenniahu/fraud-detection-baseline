"""
experiment.py
=============
核心实验逻辑模块。

提供单组实验与全矩阵实验的执行流程。
"""

import os
import time
from collections import Counter
import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED, OPTIMIZE_THRESHOLD, THRESHOLD_METRIC, VAL_SPLIT_RATIO,
    RECALCULATE_THRESHOLD,
)
from src.threshold_cache import _load_threshold_cache, _get_cached_threshold, _cache_threshold
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.resampling import get_resampled
from src.models import build_model
from src.evaluation import evaluate_model, find_optimal_threshold, get_model_proba
from src.visualization import (
    plot_pr_curves,
    plot_roc_curves,
    plot_class_distribution,
    plot_resampling_comparison,
    plot_metrics_comparison,
)
from src.results_io import _save_results, _print_summary
from src.experiment_tracker import _generate_basename, save_metadata


# ─────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────
def _default_data_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "creditcard.csv")


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
    # XGBoost: baseline 使用原始数据计算 scale_pos_weight，重采样后禁用权重
    if model_name == "xgboost":
        if strategy_name == "baseline":
            model = build_model(model_name, y_train=y_tr)  # 使用原始比例
        else:
            model = build_model(model_name, scale_pos_weight=1)  # 重采样后禁用权重
    else:
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
    basename = _generate_basename(strategies, models)
    save_metadata(basename, strategies, models, random_seed=RANDOM_SEED)
    _save_results(results_df, basename=basename)

    # ── 打印汇总表 ────────────────────────────────────────────────
    _print_summary(results_df)

    return results_df
