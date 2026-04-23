"""
viz_analysis.py
===============
分析类可视化模块（Tomek Links 等）。
"""

import numpy as np
import matplotlib.pyplot as plt
from src.viz_utils import _save_figure


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
