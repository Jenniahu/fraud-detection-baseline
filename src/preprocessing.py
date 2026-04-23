"""
preprocessing.py
================
数据预处理模块。

提供特征标准化、训练测试分割等预处理流程。
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from src.config import SCALE_COLS, TARGET_COL, RANDOM_SEED


def preprocess(df,
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


def _print_split_stats(y_train: np.ndarray, y_test: np.ndarray):
    """打印训练/测试集分割统计"""
    def _stats(y, name):
        n = len(y); f = y.sum()
        print(f"  {name:<6}: 总计 {n:>7,}  | 欺诈 {f:>4,} ({f/n*100:.4f}%)")

    print("\n─── 数据分割结果 (80/20 stratified) ───")
    _stats(y_train, "Train")
    _stats(y_test,  "Test ")
    print("─" * 45)
