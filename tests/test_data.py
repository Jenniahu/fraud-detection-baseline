"""
test_data.py
============
数据加载与预处理模块的冒烟测试。
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.config import TARGET_COL


def test_load_data_shape():
    df = load_data(verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert TARGET_COL in df.columns


def test_preprocess_split_ratio():
    df = load_data(verbose=False)
    X_train, X_test, y_train, y_test = preprocess(df, verbose=False)
    assert len(X_train) > len(X_test)
    # 默认 test_size=0.2，允许一定浮动
    assert 0.15 <= len(X_test) / len(df) <= 0.25


def test_preprocess_returns_numpy():
    df = load_data(verbose=False)
    X_train, X_test, y_train, y_test = preprocess(df, verbose=False)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
