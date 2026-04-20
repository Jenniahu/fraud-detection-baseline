"""
visualization.py
================
结果可视化模块

功能:
  - 绘制 Precision-Recall Curve (PR Curve)
  - 绘制 ROC Curve
  - 绘制混淆矩阵热力图
  - 绘制各策略 × 模型的指标对比条形图
  - 绘制类别分布饼图 / 条形图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无头环境（无 GUI）使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
)

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 全局绘图风格
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ─────────────────────────────────────────────────────────────────
# 1. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────
def plot_pr_curves(models_probas: dict,
                   y_test: np.ndarray,
                   title: str = "Precision-Recall Curves",
                   save_name: str = "pr_curves.png"):
    """
    绘制多条 PR 曲线（用于策略/模型对比）

    Parameters
    ----------
    models_probas : dict
        { 'label': y_proba_array, ... }
    y_test : np.ndarray
        真实标签
    title : str
    save_name : str
        保存文件名（存于 figures/）
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, proba) in enumerate(models_probas.items()):
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = auc(rec, prec)
        ax.plot(rec, prec,
                label=f"{label} (AUPRC={ap:.3f})",
                color=COLORS[i % len(COLORS)],
                linewidth=2)

    # 随机猜测基线
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--",
               linewidth=1, label=f"Random (AUPRC={baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 2. ROC Curve
# ─────────────────────────────────────────────────────────────────
def plot_roc_curves(models_probas: dict,
                    y_test: np.ndarray,
                    title: str = "ROC Curves",
                    save_name: str = "roc_curves.png"):
    """绘制多条 ROC 曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, proba) in enumerate(models_probas.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label=f"{label} (AUC={roc_auc:.3f})",
                color=COLORS[i % len(COLORS)],
                linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 3. 混淆矩阵热力图
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray,
                           title: str = "Confusion Matrix",
                           save_name: str = "confusion_matrix.png"):
    """绘制单个混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 4. 指标对比条形图
# ─────────────────────────────────────────────────────────────────
def plot_metrics_comparison(results_df,
                             metrics: list = None,
                             title: str = "Model Performance Comparison",
                             save_name: str = "metrics_comparison.png"):
    """
    绘制 strategy × model 的指标对比条形图

    Parameters
    ----------
    results_df : pd.DataFrame
        列应包含: strategy, model, precision, recall, f1, g_mean, auprc, auroc
    metrics : list
        要展示的指标列表，默认 ['recall','precision','f1','g_mean','auprc']
    """
    import pandas as pd

    if metrics is None:
        metrics = ["recall", "precision", "f1", "g_mean", "auprc"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics,
                              figsize=(4 * n_metrics, 5),
                              sharey=False)
    if n_metrics == 1:
        axes = [axes]

    strategies = results_df["strategy"].unique()
    models     = results_df["model"].unique()
    x          = np.arange(len(strategies))
    bar_width  = 0.8 / len(models)

    for ax, metric in zip(axes, metrics):
        for i, mdl in enumerate(models):
            sub = results_df[results_df["model"] == mdl]
            sub = sub.set_index("strategy").reindex(strategies)
            vals = sub[metric].values.astype(float)

            offset = (i - len(models) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals,
                          width=bar_width,
                          label=mdl,
                          color=COLORS[i % len(COLORS)],
                          alpha=0.85,
                          edgecolor="white")

            # 数值标注
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}",
                            ha="center", va="bottom",
                            fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if metric == metrics[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 5. 类别分布图
# ─────────────────────────────────────────────────────────────────
def plot_class_distribution(y: np.ndarray,
                             title: str = "Class Distribution",
                             save_name: str = "class_dist.png"):
    """绘制类别分布条形图"""
    from collections import Counter
    counter = Counter(y)
    labels  = ["Normal (0)", "Fraud (1)"]
    values  = [counter[0], counter[1]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 条形图
    colors = ["#4C72B0", "#C44E52"]
    axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for i, v in enumerate(values):
        axes[0].text(i, v + max(values) * 0.01,
                     f"{v:,}\n({v/sum(values)*100:.3f}%)",
                     ha="center", fontsize=10)
    axes[0].set_title("Count")
    axes[0].set_ylabel("Number of Samples")

    # 饼图（对数视角）
    axes[1].pie(values, labels=labels, colors=colors,
                autopct="%1.3f%%", startangle=140,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Proportion")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 6. 重采样后分布对比
# ─────────────────────────────────────────────────────────────────
def plot_resampling_comparison(strategy_counts: dict,
                                save_name: str = "resampling_comparison.png"):
    """
    对比各重采样策略后的样本量分布

    Parameters
    ----------
    strategy_counts : dict
        { 'strategy_name': {'normal': int, 'fraud': int}, ... }
    """
    strategies = list(strategy_counts.keys())
    normals    = [v["normal"] for v in strategy_counts.values()]
    frauds     = [v["fraud"]  for v in strategy_counts.values()]
    x          = np.arange(len(strategies))
    w          = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, normals, w, label="Normal",
                color="#4C72B0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, frauds,  w, label="Fraud",
                color="#C44E52", alpha=0.85, edgecolor="white")

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 500, f"{h:,.0f}",
                    ha="center", va="bottom", fontsize=9)

    _label_bars(b1); _label_bars(b2)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Distribution After Resampling")
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, save_name)


# ─────────────────────────────────────────────────────────────────
# 7. Tomek Links 可视化
# ─────────────────────────────────────────────────────────────────
def plot_tomek_links(X: np.ndarray,
                     y: np.ndarray,
                     title: str = "Tomek Links Visualization",
                     save_name: str = "tomek_links.png"):
    """
    可视化 Tomek Links 的位置和删除效果

    Parameters
    ----------
    X : np.ndarray
        特征数据（会被 PCA 降维到 2D）
    y : np.ndarray
        标签（0=多数类, 1=少数类）
    title : str
    save_name : str
        保存文件名

    Returns
    -------
    dict: 统计信息 {'n_tomek_links': int, 'n_removed': int, 'removal_ratio': float}
    """
    from imblearn.under_sampling import TomekLinks
    from sklearn.decomposition import PCA

    # 执行 Tomek Links
    tomek = TomekLinks(n_jobs=-1)
    X_res, y_res = tomek.fit_resample(X, y)

    # 统计删除情况
    n_original = len(y)
    n_resampled = len(y_res)
    n_removed = n_original - n_resampled
    removal_ratio = n_removed / n_original * 100

    # 找到被删除的样本索引
    removed_mask = np.ones(n_original, dtype=bool)
    if hasattr(tomek, 'sample_indices_'):
        removed_mask[tomek.sample_indices_] = False
    else:
        # 通过对比找到被删除的样本
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(X, X_res)
        min_dists = dist_matrix.min(axis=1)
        removed_mask = min_dists > 1e-10

    n_tomek_links = removed_mask.sum()

    # PCA 降维到 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：原始数据
    ax1 = axes[0]
    ax1.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                c='#4C72B0', label=f'Normal (n={sum(y==0)})',
                alpha=0.5, s=20, edgecolors='none')
    ax1.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                c='#C44E52', label=f'Fraud (n={sum(y==1)})',
                alpha=0.7, s=30, edgecolors='none')
    ax1.set_title(f'Original Data (n={n_original})', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 右图：标记 Tomek Links
    ax2 = axes[1]

    # 保留的样本
    kept_mask = ~removed_mask
    ax2.scatter(X_2d[kept_mask & (y == 0), 0], X_2d[kept_mask & (y == 0), 1],
                c='#4C72B0', label=f'Normal (kept, n={sum(kept_mask & (y==0))})',
                alpha=0.5, s=20, edgecolors='none')
    ax2.scatter(X_2d[kept_mask & (y == 1), 0], X_2d[kept_mask & (y == 1), 1],
                c='#C44E52', label=f'Fraud (kept, n={sum(kept_mask & (y==1))})',
                alpha=0.7, s=30, edgecolors='none')

    # 被删除的 Tomek Links（用黑色叉标记）
    if n_tomek_links > 0:
        ax2.scatter(X_2d[removed_mask, 0], X_2d[removed_mask, 1],
                    c='black', marker='x', s=100, linewidths=2,
                    label=f'Removed (Tomek, n={n_tomek_links})')

    ax2.set_title(f'After Tomek Links (removed {n_removed}, {removal_ratio:.3f}%)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, save_name)

    stats = {
        'n_tomek_links': int(n_tomek_links),
        'n_removed': int(n_removed),
        'removal_ratio': round(removal_ratio, 4),
        'n_original': int(n_original),
        'n_resampled': int(n_resampled),
    }

    print(f"[Tomek Links 统计]")
    print(f"  原始样本: {n_original:,}")
    print(f"  删除样本: {n_removed:,} ({removal_ratio:.4f}%)")
    print(f"  剩余样本: {n_resampled:,}")

    return stats


# ─────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────
def _save_figure(fig, filename: str):
    """保存图片到 figures/ 目录"""
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] 图片已保存: {path}")
