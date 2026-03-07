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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH    = os.path.join(DATA_DIR, "creditcard.csv")
KAGGLE_URL  = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

TARGET_COL  = "Class"
SCALE_COLS  = ["Time", "Amount"]   # 需要 RobustScaler 的列
RANDOM_SEED = 42


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


def preprocess(df: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = RANDOM_SEED,
               verbose: bool = True):
    """
    完整预处理流程:
      1. 用 RobustScaler 标准化 Time / Amount
      2. 分层 80/20 训练测试分割

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    test_size : float
        测试集比例，默认 0.2
    random_state : int
        随机种子
    verbose : bool
        是否打印分割结果信息

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    df = df.copy()

    # 1. RobustScaler 标准化（对异常值鲁棒）
    scaler = RobustScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    # 2. 特征 / 标签分离
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    # 3. 分层分割（保持正负样本比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if verbose:
        _print_split_stats(y_train, y_test)

    return X_train, X_test, y_train, y_test


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


def _print_split_stats(y_train: np.ndarray, y_test: np.ndarray):
    """打印训练/测试集分割统计"""
    def _stats(y, name):
        n = len(y); f = y.sum()
        print(f"  {name:<6}: 总计 {n:>7,}  | 欺诈 {f:>4,} ({f/n*100:.4f}%)")

    print("\n─── 数据分割结果 (80/20 stratified) ───")
    _stats(y_train, "Train")
    _stats(y_test,  "Test ")
    print("─" * 45)


# ─────────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
