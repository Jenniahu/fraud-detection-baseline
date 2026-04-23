"""
viz_distribution.py
===================
类别分布与重采样分布绘制模块。
"""

import numpy as np
import matplotlib.pyplot as plt
from src.viz_utils import _save_figure


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
