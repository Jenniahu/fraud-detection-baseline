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
from src.evaluation   import evaluate_model
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
DEFAULT_MODELS     = ["logistic_regression", "random_forest", "xgboost"]
RANDOM_SEED        = 42

# ─────────────────────────────────────────────────────────────────
# 核心实验流程
# ─────────────────────────────────────────────────────────────────
def run_single_experiment(strategy_name: str,
                           model_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           verbose: bool = True) -> dict:
    """
    运行单组实验: 一种重采样策略 + 一种分类器

    Returns
    -------
    dict 包含 strategy, model, 各项指标, 训练时间, y_proba
    """
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

    # 2. 构建 & 训练模型
    model = build_model(model_name)
    t1    = time.time()
    model.fit(X_res, y_res)
    train_time = time.time() - t1

    # 3. 评估
    metrics = evaluate_model(model, X_test, y_test, verbose=verbose)

    # 4. 保存预测概率（用于后续绘图）
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        df = model.decision_function(X_test)
        y_proba = (df - df.min()) / (df.max() - df.min() + 1e-9)

    result = {
        "strategy"     : strategy_name,
        "model"        : model_name,
        "resample_time": round(resample_time, 3),
        "train_time"   : round(train_time,    3),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "_y_proba"     : y_proba,           # 不保存到 CSV
        "_cm"          : metrics["confusion_matrix"],
    }

    if verbose:
        print(f"\n  ⏱  重采样: {resample_time:.2f}s | 训练: {train_time:.2f}s")

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
    print("\n" + "═" * 75)
    print("  实验汇总（按 AUPRC 降序排列）")
    print("═" * 75)
    display_cols = ["strategy", "model", "recall", "precision",
                    "f1", "g_mean", "auprc", "auroc"]
    summary = df[display_cols].sort_values("auprc", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("═" * 75)

    # 最优组合
    best = summary.iloc[0]
    print(f"\n  🏆 最优组合 (AUPRC): "
          f"{best['strategy']} × {best['model']}  "
          f"AUPRC={best['auprc']:.4f}  Recall={best['recall']:.4f}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_all_experiments(
        strategies=args.strategies,
        models=args.models,
        data_path=args.data_path,
        verbose=not args.quiet,
    )
