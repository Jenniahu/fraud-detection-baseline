"""
analyze_tomek.py
================
分析 Tomek Links 的效果

运行:
    python analyze_tomek.py

输出:
    - figures/06_tomek_links_train.png: 训练集 Tomek Links 可视化
    - 控制台统计信息
"""

import os
import sys
import numpy as np

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.visualization import plot_tomek_links


def main():
    print("=" * 60)
    print("Tomek Links 效果分析")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    df = load_data(verbose=True)
    X_train, X_test, y_train, y_test = preprocess(df, verbose=True)

    # 分析训练集上的 Tomek Links
    print("\n[2] 分析训练集的 Tomek Links...")
    print("-" * 60)

    stats = plot_tomek_links(
        X_train, y_train,
        title="Tomek Links Analysis (Training Set)",
        save_name="06_tomek_links_train.png"
    )

    # 打印详细统计
    print("\n" + "=" * 60)
    print("详细统计")
    print("=" * 60)
    print(f"原始训练集样本数: {stats['n_original']:,}")
    print(f"Tomek Links 删除数: {stats['n_removed']:,}")
    print(f"删除比例: {stats['removal_ratio']:.4f}%")
    print(f"剩余样本数: {stats['n_resampled']:,}")

    # 类别分布变化
    n_fraud_original = np.sum(y_train)
    n_normal_original = len(y_train) - n_fraud_original

    print(f"\n类别分布变化:")
    print(f"  原始: Normal={n_normal_original:,}, Fraud={n_fraud_original:,}")
    print(f"  不平衡比: 1:{n_normal_original // n_fraud_original}")

    print("\n" + "=" * 60)
    print("分析完成！图片保存至: figures/06_tomek_links_train.png")
    print("=" * 60)

    # 结论
    print("\n[结论]")
    if stats['removal_ratio'] < 0.1:
        print("⚠️  Tomek Links 删除样本极少 (<0.1%)，对极度不平衡数据效果有限")
        print("    建议考虑: 1) 结合 SMOTE 使用 (SMOTE-Tomek)")
        print("              2) 使用更强的欠采样策略 (如 ENN)")
        print("              3) 调整 Tomek Links 参数或自定义实现")
    elif stats['removal_ratio'] < 1:
        print("ℹ️  Tomek Links 删除样本较少 (<1%)，有一定清理效果")
        print("    建议结合 SMOTE 使用以获得更好效果")
    else:
        print("✅ Tomek Links 删除了较多样本，边界清理效果显著")


if __name__ == "__main__":
    main()
