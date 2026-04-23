"""
viz_curves.py
=============
PR / ROC 曲线绘制模块。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from src.viz_utils import COLORS, _save_figure


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
