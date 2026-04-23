"""
test_sampling.py
================
重采样策略模块的冒烟测试。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.resampling import get_resampled, list_strategies


def test_all_strategies_return_arrays():
    df = load_data(verbose=False)
    X_train, _, y_train, _ = preprocess(df, verbose=False)
    strategies = list_strategies()
    assert len(strategies) > 0
    for strat in strategies:
        X_res, y_res = get_resampled(strat, X_train, y_train, random_state=42)
        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)
        assert len(np.unique(y_res)) <= 2  # 最多两类


def test_baseline_no_change():
    df = load_data(verbose=False)
    X_train, _, y_train, _ = preprocess(df, verbose=False)
    X_res, y_res = get_resampled("baseline", X_train, y_train, random_state=42)
    assert len(X_res) == len(X_train)
    assert len(y_res) == len(y_train)
