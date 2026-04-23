"""
viz_utils.py
============
可视化工具函数与公共配置。
"""

import os
import matplotlib
matplotlib.use("Agg")          # 无头环境（无 GUI）使用非交互后端
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR as FIGURE_DIR, COLORS


def _save_figure(fig, filename: str):
    """保存图片到 figures/ 目录"""
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] 图片已保存: {path}")
