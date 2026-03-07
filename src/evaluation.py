"""
evaluation.py
=============
模型评估模块

实现 Proposal 要求的全套评估指标:
  - Precision / Recall / F1
  - G-Mean  = sqrt(TPR × TNR)
  - AUPRC   (Area Under Precision-Recall Curve) —— 主要评估指标
  - AUROC   (Area Under ROC Curve)              —— 辅助参考
  - Confusion Matrix

提供 evaluate_model() 统一接口，返回指标字典。
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,   # AUPRC
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────────
# 核心评估函数
# ─────────────────────────────────────────────────────────────────
def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   threshold: float = 0.5,
                   verbose: bool = True) -> dict:
    """
    在测试集上评估模型，返回全套指标字典。

    Parameters
    ----------
    model :
        已训练的 sklearn 兼容分类器
    X_test : np.ndarray
    y_test : np.ndarray
    threshold : float
        分类阈值（默认 0.5，可调以优化 Recall/Precision 平衡）
    verbose : bool
        是否打印详细报告

    Returns
    -------
    dict 包含以下 key:
        precision, recall, f1, g_mean, auprc, auroc,
        confusion_matrix, threshold
    """
    # ── 获取预测概率 ──────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        y_proba = (df - df.min()) / (df.max() - df.min() + 1e-9)
    else:
        raise AttributeError("模型需要支持 predict_proba 或 decision_function")

    # ── 根据阈值得到二值预测 ──────────────────────────────────────
    y_pred = (y_proba >= threshold).astype(int)

    # ── 混淆矩阵 ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── 基础指标 ─────────────────────────────────────────────────
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    # ── G-Mean = sqrt(TPR × TNR) ─────────────────────────────────
    tpr    = tp / (tp + fn + 1e-9)   # Sensitivity / Recall
    tnr    = tn / (tn + fp + 1e-9)   # Specificity
    g_mean = np.sqrt(tpr * tnr)

    # ── AUPRC（主要指标）& AUROC ──────────────────────────────────
    auprc = average_precision_score(y_test, y_proba)
    auroc = roc_auc_score(y_test, y_proba)

    results = {
        "precision"       : round(precision, 6),
        "recall"          : round(recall,    6),
        "f1"              : round(f1,         6),
        "g_mean"          : round(g_mean,     6),
        "auprc"           : round(auprc,      6),
        "auroc"           : round(auroc,      6),
        "tp"              : int(tp),
        "fp"              : int(fp),
        "tn"              : int(tn),
        "fn"              : int(fn),
        "confusion_matrix": cm,
        "threshold"       : threshold,
    }

    if verbose:
        _print_results(results)

    return results


def compute_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """单独计算 G-Mean（TPR * TNR 的几何平均）"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    return float(np.sqrt(tpr * tnr))


# ─────────────────────────────────────────────────────────────────
# 打印格式化报告
# ─────────────────────────────────────────────────────────────────
def _print_results(res: dict):
    """美化打印评估指标"""
    print("\n" + "═" * 45)
    print("  评估结果摘要")
    print("═" * 45)
    print(f"  Precision : {res['precision']:.4f}")
    print(f"  Recall    : {res['recall']:.4f}")
    print(f"  F1-Score  : {res['f1']:.4f}")
    print(f"  G-Mean    : {res['g_mean']:.4f}   ← sqrt(TPR×TNR)")
    print(f"  AUPRC     : {res['auprc']:.4f}   ← 主要评估指标")
    print(f"  AUROC     : {res['auroc']:.4f}")
    print("─" * 45)
    cm = res['confusion_matrix']
    print(f"  混淆矩阵  :  TN={res['tn']:>5}  FP={res['fp']:>5}")
    print(f"              FN={res['fn']:>5}  TP={res['tp']:>5}")
    print("═" * 45)


def print_classification_report(y_test: np.ndarray, y_pred: np.ndarray):
    """打印 sklearn 原始分类报告"""
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Fraud"],
                                 digits=4))


# ─────────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 用随机数据做简单验证
    rng   = np.random.default_rng(42)
    y_t   = rng.integers(0, 2, 1000)
    proba = rng.random(1000)

    class _DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - proba, proba])

    dummy = _DummyModel()
    evaluate_model(dummy, None, y_t)
