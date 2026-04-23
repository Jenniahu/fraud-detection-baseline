"""
viz_comparison.py
=================
指标对比条形图绘制模块。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src.viz_utils import COLORS, _save_figure


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
