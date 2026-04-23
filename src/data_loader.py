"""
data_loader.py
==============
数据加载与预处理模块

功能:
  - 自动下载 / 加载 Kaggle 信用卡欺诈检测数据集
  - 特征标准化 (RobustScaler)
  - 分层 80/20 训练测试分割
  - 数据集统计信息展示
"""

import os
import numpy as np
import pandas as pd
from src.config import DATA_DIR, CSV_PATH, KAGGLE_URL, TARGET_COL, RANDOM_SEED


# ─────────────────────────────────────────────────────────────────
# 公共接口
# ─────────────────────────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    加载原始 CSV 数据集。
    若本地不存在则尝试从网络下载。

    Parameters
    ----------
    csv_path : str
        本地 CSV 文件路径（默认 data/creditcard.csv）
    verbose : bool
        是否打印统计信息

    Returns
    -------
    pd.DataFrame
        原始数据框
    """
    if not os.path.exists(csv_path):
        print(f"[data_loader] 未找到本地数据: {csv_path}")
        print(f"[data_loader] 尝试从网络下载: {KAGGLE_URL}")
        _download_data(csv_path)

    df = pd.read_csv(csv_path)

    if verbose:
        _print_stats(df)

    return df


def get_feature_names(df: pd.DataFrame) -> list:
    """返回去掉目标列后的特征名列表"""
    return [c for c in df.columns if c != TARGET_COL]


# ─────────────────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────────────────
def _download_data(save_path: str):
    """从 TensorFlow 镜像下载数据集"""
    import urllib.request
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("[data_loader] 下载中，请稍候（约 66 MB）…")
    urllib.request.urlretrieve(KAGGLE_URL, save_path)
    print(f"[data_loader] 数据已保存至: {save_path}")


def _print_stats(df: pd.DataFrame):
    """打印数据集基本统计"""
    n_total  = len(df)
    n_fraud  = df[TARGET_COL].sum()
    n_normal = n_total - n_fraud
    ratio    = n_fraud / n_total * 100

    print("=" * 55)
    print("  数据集统计信息")
    print("=" * 55)
    print(f"  总样本数   : {n_total:>10,}")
    print(f"  正常交易   : {n_normal:>10,}  ({100 - ratio:.4f}%)")
    print(f"  欺诈交易   : {n_fraud:>10,}  ({ratio:.4f}%)")
    print(f"  类别不平衡比: 1 : {n_normal // n_fraud:.0f}")
    print(f"  特征维度   : {df.shape[1] - 1}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    from src.preprocessing import preprocess
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
