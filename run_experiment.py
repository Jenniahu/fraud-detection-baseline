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
import joblib

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
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_STRATEGIES = ["baseline", "oversample", "undersample", "hybrid", "easy_ensemble"]
# 策略说明:
#   baseline    : 原始不平衡数据
#   oversample  : SMOTE 过采样
#   undersample : SMOTE-ENN（先 SMOTE 平衡，再 ENN 清理）
#   hybrid      : SMOTE-Tomek（Proposal 核心研究对象）
DEFAULT_MODELS     = ["logistic_regression", "random_forest", "xgboost"]  # resample_boost 作为独立算法，需手动指定 --models
RANDOM_SEED        = 42

# 阈值优化配置
OPTIMIZE_THRESHOLD = True          # 是否启用阈值优化
THRESHOLD_METRIC   = "f1"          # 优化目标: f1 | g_mean | precision | recall
VAL_SPLIT_RATIO    = 0.2           # 从训练集划分验证集的比例

# 阈值缓存配置
THRESHOLD_CACHE_FILE = os.path.join(RESULTS_DIR, "optimal_thresholds.json")
RECALCULATE_THRESHOLD = False      # 是否强制重新计算阈值（忽略缓存）

# 当前 Amount 变换方式（用于区分不同实验配置的阈值缓存）
_CURRENT_AMOUNT_TRANSFORM = "robust"

# ─────────────────────────────────────────────────────────────────
# 阈值缓存管理
# ─────────────────────────────────────────────────────────────────
def _load_threshold_cache(cache_file: str = None) -> dict:
    """
    加载阈值缓存文件

    Returns
    -------
    dict: { (amount_transform, strategy, model, metric): threshold, ... }
    """
    if cache_file is None:
        cache_file = THRESHOLD_CACHE_FILE
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        # 转换 key 从字符串元组格式
        return {tuple(k.split('|')): v for k, v in cache.items()}
    except Exception:
        return {}


def _save_threshold_cache(cache: dict, cache_file: str = None):
    """
    保存阈值缓存到文件

    Parameters
    ----------
    cache : dict
        { (amount_transform, strategy, model, metric): threshold, ... }
    """
    if cache_file is None:
        cache_file = THRESHOLD_CACHE_FILE
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
    key = (_CURRENT_AMOUNT_TRANSFORM, strategy, model, metric)
    return cache.get(key)


def _cache_threshold(strategy: str, model: str, metric: str,
                     threshold: float, cache: dict = None):
    """
    将阈值存入缓存并保存到文件
    """
    if cache is None:
        cache = _load_threshold_cache()
    key = (_CURRENT_AMOUNT_TRANSFORM, strategy, model, metric)
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
                           threshold_metric: str = THRESHOLD_METRIC,
                           sample_weights: np.ndarray = None,
                           amount_test: np.ndarray = None,
                           save_model_path: str = None,
                           load_model_path: str = None,
                           force_retrain: bool = False) -> dict:
    """
    运行单组实验: 一种重采样策略 + 一种分类器

    Parameters
    ----------
    optimize_threshold : bool
        是否启用阈值优化（从训练集划分验证集搜索最优阈值）
    threshold_metric : str
        阈值优化目标指标: f1 | g_mean | precision | recall
    sample_weights : np.ndarray
        原始训练集的样本权重（如金额权重），与重采样后样本对齐
    amount_test : np.ndarray
        测试集原始交易金额（用于计算金额敏感指标）

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

    # 1. 重采样（集成模型内部处理采样，外部跳过）
    t0 = time.time()
    if model_name == "resample_boost" or strategy_name == "easy_ensemble":
        # resample_boost / easy_ensemble 策略：采样在模型训练阶段内部完成
        X_res, y_res = X_train, y_train
        weights_res = sample_weights.copy() if sample_weights is not None else None
        reason = "resample_boost 内部迭代重采样" if model_name == "resample_boost" else "EasyEnsemble 内部多次RUS"
        print(f"  [{strategy_name}] 外部不重采样，{reason}")
    else:
        X_res, y_res = get_resampled(
            strategy_name, X_train, y_train,
            random_state=RANDOM_SEED
        )
        # 为重采样后的样本匹配原始金额权重（合成样本默认权重=1）
        if sample_weights is not None:
            weights_res = _match_sample_weights(X_res, X_train, sample_weights)
        else:
            weights_res = None
    resample_time = time.time() - t0

    # 2. 划分训练/验证集（用于阈值优化）
    if optimize_threshold:
        if weights_res is not None:
            X_tr, X_val, y_tr, y_val, weights_tr, weights_val = train_test_split(
                X_res, y_res, weights_res,
                test_size=VAL_SPLIT_RATIO,
                stratify=y_res,
                random_state=RANDOM_SEED,
            )
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_res, y_res,
                test_size=VAL_SPLIT_RATIO,
                stratify=y_res,
                random_state=RANDOM_SEED,
            )
            weights_tr = None
    else:
        X_tr, y_tr = X_res, y_res
        X_val, y_val = None, None
        weights_tr = weights_res

    # 3. 构建 & 训练模型（支持缓存加载）
    model = None
    model_loaded = False
    if not force_retrain and load_model_path and os.path.exists(load_model_path):
        try:
            model = joblib.load(load_model_path)
            model_loaded = True
            if verbose:
                print(f"  💾 加载缓存模型: {os.path.basename(load_model_path)}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  缓存加载失败，重新训练: {e}")

    if not model_loaded:
        # XGBoost: baseline 使用原始数据计算 scale_pos_weight，重采样后禁用权重
        # resample_boost: 内部迭代重采样
        # easy_ensemble 策略: 用 EasyEnsembleWrapper 包装基础模型，内部多次RUS+Bagging
        if model_name == "resample_boost":
            model = build_model(model_name, strategy=strategy_name)
        elif strategy_name == "easy_ensemble":
            # 构建基础模型（EasyEnsemble内部已平衡，XGBoost无需scale_pos_weight）
            if model_name == "xgboost":
                base_model = build_model(model_name, scale_pos_weight=1)
            else:
                base_model = build_model(model_name)
            # 用 EasyEnsembleWrapper 包装，实现多次RUS + 多模型集成
            from src.ensemble_sampler import EasyEnsembleWrapper
            model = EasyEnsembleWrapper(
                base_estimator=base_model,
                n_subsets=10,
                random_state=RANDOM_SEED,
            )
        elif model_name == "xgboost":
            if strategy_name == "baseline":
                model = build_model(model_name, y_train=y_tr)  # 使用原始比例
            else:
                model = build_model(model_name, scale_pos_weight=1)  # 重采样后禁用权重
        else:
            model = build_model(model_name)
        t1 = time.time()
        fit_kwargs = {}
        if weights_tr is not None:
            fit_kwargs["sample_weight"] = weights_tr
        # XGBoost: 每 10 轮打印一次训练进度
        if model_name == "xgboost" and X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = 10
        model.fit(X_tr, y_tr, **fit_kwargs)
        train_time = time.time() - t1

        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            joblib.dump(model, save_model_path)
            if verbose:
                print(f"  💾 保存模型: {os.path.basename(save_model_path)}")
    else:
        train_time = 0.0

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
                             threshold=best_threshold, verbose=verbose,
                             amount_test=amount_test)

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
                         verbose: bool = True,
                         amount_transform: str = "robust",
                         use_amount_weight: bool = False,
                         force_retrain: bool = False) -> pd.DataFrame:
    """
    运行全矩阵实验并返回汇总 DataFrame

    Parameters
    ----------
    strategies : list  重采样策略名列表
    models     : list  模型名列表
    data_path  : str   可选，自定义 CSV 路径
    verbose    : bool
    amount_transform : str
        Amount 字段预处理方式: "robust" | "log1p"
    use_amount_weight : bool
        是否启用金额加权损失（大额欺诈漏报惩罚更高）

    Returns
    -------
    pd.DataFrame  每行为一组实验的所有指标
    """
    # 设置全局 Amount 变换标记（用于阈值缓存区分）
    global _CURRENT_AMOUNT_TRANSFORM
    _CURRENT_AMOUNT_TRANSFORM = amount_transform

    # ── 加载与预处理数据 ──────────────────────────────────────────
    transform_label = "RobustScaler" if amount_transform == "robust" else "log1p+RobustScaler"
    weight_label = " + AmountWeighted" if use_amount_weight else ""
    print("\n" + "═" * 60)
    print("  🚀 欺诈检测 Baseline 实验框架")
    print(f"  Amount 处理: {transform_label}{weight_label}")
    print("  Hybrid Resampling for Imbalanced Financial Fraud Detection")
    print("═" * 60)

    df = load_data(csv_path=data_path or _default_data_path(),
                   verbose=verbose)
    X_train, X_test, y_train, y_test, amount_train, amount_test = preprocess(
        df, verbose=verbose, amount_transform=amount_transform, return_amount=True
    )

    # 计算金额加权样本权重（基于原始交易金额）
    if use_amount_weight:
        from src.data_loader import compute_amount_weight
        sample_weights = compute_amount_weight(amount_train, method="log1p")
        print(f"\n[Amount Weight] 训练集金额权重范围: {sample_weights.min():.3f} ~ {sample_weights.max():.3f} (均值=1.0)")
    else:
        sample_weights = None

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
            cache_path = _get_model_cache_path(
                strategy, model_name, amount_transform, use_amount_weight
            )
            result = run_single_experiment(
                strategy, model_name,
                X_train, y_train, X_test, y_test,
                verbose=verbose,
                sample_weights=sample_weights,
                amount_test=amount_test if use_amount_weight else None,
                save_model_path=cache_path,
                load_model_path=cache_path,
                force_retrain=force_retrain,
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
    # 文件名自动包含策略、模型、amount_transform、是否权重
    if len(strategies) == 1:
        strategy_part = strategies[0]
    else:
        strategy_part = f"{len(strategies)}strats"
    if len(models) == 1:
        model_part = models[0]
    else:
        model_part = f"{len(models)}models"
    suffix = f"{strategy_part}_{model_part}_{amount_transform}"
    if use_amount_weight:
        suffix += "_weighted"
    _save_results(results_df, suffix=suffix)

    # ── 打印汇总表 ────────────────────────────────────────────────
    _print_summary(results_df)

    return results_df


# ─────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────
def _default_data_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "creditcard.csv")


def _get_model_cache_path(strategy_name: str, model_name: str,
                          amount_transform: str, use_amount_weight: bool) -> str:
    """生成模型缓存文件路径"""
    weight_suffix = "_weighted" if use_amount_weight else ""
    filename = f"{strategy_name}_{model_name}_{amount_transform}{weight_suffix}.joblib"
    return os.path.join(MODELS_DIR, filename)


def _match_sample_weights(X_res: np.ndarray,
                          X_train: np.ndarray,
                          weights_train: np.ndarray) -> np.ndarray:
    """
    为重采样后的样本匹配原始金额权重。
    对于 SMOTE/ADASYN 生成的合成样本（无法在原始集中找到精确匹配），
    赋予默认权重=1，避免对模型产生不当影响。
    """
    weights_res = np.ones(len(X_res), dtype=float)
    for i, x in enumerate(X_res):
        match = np.all(np.isclose(X_train, x), axis=1)
        if np.any(match):
            weights_res[i] = weights_train[match][0]
    return weights_res


def _save_results(df: pd.DataFrame, suffix: str = ""):
    """将结果保存为 CSV 和 JSON"""
    suffix_str = f"_{suffix}" if suffix else ""
    csv_path  = os.path.join(RESULTS_DIR, f"experiment_results{suffix_str}.csv")
    json_path = os.path.join(RESULTS_DIR, f"experiment_results{suffix_str}.json")

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
    # 如果有金额敏感指标，加入显示列
    if "fn_amount" in df.columns:
        display_cols += ["fn_amount", "amount_recall", "amount_precision"]
    summary = df[display_cols].sort_values("auprc", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("═" * 85)

    # 最优组合
    best = summary.iloc[0]
    print(f"\n  🏆 最优组合 (AUPRC): "
          f"{best['strategy']} × {best['model']}  "
          f"阈值={best['threshold']:.3f}  "
          f"AUPRC={best['auprc']:.4f}  Recall={best['recall']:.4f}")

    # 金额敏感指标最优（如果有）
    if "fn_amount" in df.columns:
        best_fn = df.loc[df["fn_amount"].idxmin()]
        best_ar = df.loc[df["amount_recall"].idxmax()]
        print(f"\n  💰 最低漏报金额 (FN_Amount): "
              f"{best_fn['strategy']} × {best_fn['model']}  "
              f"FN_Amount=€{best_fn['fn_amount']:,.2f}")
        print(f"  💰 最高金额召回 (Amount_Recall): "
              f"{best_ar['strategy']} × {best_ar['model']}  "
              f"Amount_Recall={best_ar['amount_recall']:.4f}")

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
    parser.add_argument(
        "--amount_transform", type=str, default="robust",
        choices=["robust", "log1p"],
        help="Amount 字段预处理方式: robust(默认, 直接RobustScaler) | log1p(先对数变换再RobustScaler)",
    )
    parser.add_argument(
        "--use_amount_weight", action="store_true",
        help="启用金额加权损失：大额欺诈漏报的惩罚更高（基于log1p(amount)计算权重）",
    )
    parser.add_argument(
        "--force_retrain", action="store_true",
        help="强制重新训练模型，忽略已保存的模型缓存",
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
        amount_transform=args.amount_transform,
        use_amount_weight=args.use_amount_weight,
        force_retrain=args.force_retrain,
    )
