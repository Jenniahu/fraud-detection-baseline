"""
visualization.py
================
结果可视化兼容层。

所有绘图函数已从本模块迁移至独立的子模块，
本文件通过 re-export 保持原有导入路径的向后兼容。
"""

# 混淆矩阵热力图（较小，保留在本文件）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.viz_utils import FIGURE_DIR, _save_figure


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


# 从子模块 re-export，保持原有导入路径可用
from src.viz_curves import plot_pr_curves, plot_roc_curves
from src.viz_comparison import plot_metrics_comparison
from src.viz_distribution import plot_class_distribution, plot_resampling_comparison
from src.viz_analysis import plot_tomek_links

