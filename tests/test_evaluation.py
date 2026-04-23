"""
test_evaluation.py
==================
模型评估模块的冒烟测试。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_model, find_optimal_threshold


class _DummyModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        n = len(X) if X is not None else len(self._proba)
        proba = self._proba[:n]
        return np.column_stack([1 - proba, proba])


def test_evaluate_model_keys():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    proba = rng.random(100)
    model = _DummyModel(proba)
    metrics = evaluate_model(model, None, y_true, verbose=False)
    expected_keys = {"precision", "recall", "f1", "g_mean", "auprc", "auroc",
                     "tp", "fp", "tn", "fn", "confusion_matrix", "threshold"}
    assert expected_keys.issubset(metrics.keys())


def test_find_optimal_threshold_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    proba = rng.random(100)
    best_thresh, best_score, all_scores = find_optimal_threshold(y_true, proba, metric="f1")
    assert 0.01 <= best_thresh <= 0.99
    assert best_score >= 0
    assert len(all_scores) > 0
