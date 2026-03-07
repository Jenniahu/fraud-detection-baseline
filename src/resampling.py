"""
resampling.py
=============
重采样策略模块

支持的策略:
  - baseline      : 不做任何重采样（原始不平衡数据）
  - oversample    : SMOTE 过采样
  - undersample   : Tomek Links 欠采样
  - hybrid        : SMOTE-Tomek 混合采样（先 SMOTE 后 Tomek）
  - smoteenn      : SMOTE-ENN（额外对比基准，可选）

所有策略返回 (X_resampled, y_resampled) numpy 数组。
"""

import numpy as np
from collections import Counter
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import TomekLinks
from imblearn.combine         import SMOTETomek, SMOTEENN


# ─────────────────────────────────────────────────────────────────
# 策略注册表 (name -> callable)
# ─────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict = {}


def register(name: str):
    """装饰器：将函数注册为重采样策略"""
    def decorator(fn):
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────
# 具体策略实现
# ─────────────────────────────────────────────────────────────────
@register("baseline")
def strategy_baseline(X_train: np.ndarray,
                       y_train: np.ndarray,
                       random_state: int = 42,
                       **kwargs):
    """
    基线：不做任何重采样，直接返回原始训练集。
    用于对比，体现类别不平衡对模型的影响。
    """
    _print_distribution("Baseline (no resampling)", y_train)
    return X_train, y_train


@register("oversample")
def strategy_smote(X_train: np.ndarray,
                   y_train: np.ndarray,
                   random_state: int = 42,
                   k_neighbors: int = 5,
                   **kwargs):
    """
    SMOTE 过采样（Synthetic Minority Over-sampling Technique）
    - 对少数类生成合成样本，使正负类比例达到 1:1
    - k_neighbors: KNN 邻居数，默认 5
    """
    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=random_state,
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    _print_distribution("SMOTE (Oversample)", y_res)
    return X_res, y_res


@register("undersample")
def strategy_tomek(X_train: np.ndarray,
                   y_train: np.ndarray,
                   random_state: int = 42,
                   **kwargs):
    """
    Tomek Links 欠采样
    - 移除多数类中与少数类最近邻的样本，清晰化决策边界
    - 不会大幅减少样本量，仅做边界清理
    """
    tomek = TomekLinks(n_jobs=-1)
    X_res, y_res = tomek.fit_resample(X_train, y_train)
    _print_distribution("Tomek Links (Undersample)", y_res)
    return X_res, y_res


@register("hybrid")
def strategy_smotetomek(X_train: np.ndarray,
                         y_train: np.ndarray,
                         random_state: int = 42,
                         k_neighbors: int = 5,
                         **kwargs):
    """
    SMOTE-Tomek 混合策略（Proposal 核心研究对象）
    - Step 1: SMOTE 生成合成少数类样本
    - Step 2: Tomek Links 清理合成样本附近的噪声/边界模糊样本
    - 兼顾过采样和欠采样优势，旨在得到更清晰的决策边界
    """
    smote_tomek = SMOTETomek(
        smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
    )
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    _print_distribution("SMOTE-Tomek (Hybrid)", y_res)
    return X_res, y_res


@register("smoteenn")
def strategy_smoteenn(X_train: np.ndarray,
                       y_train: np.ndarray,
                       random_state: int = 42,
                       k_neighbors: int = 5,
                       **kwargs):
    """
    SMOTE-ENN（额外对比基准，可选）
    - 用 Edited Nearest Neighbours 替代 Tomek Links 做边界清理
    - ENN 清理力度更强，移除误分类样本
    """
    smote_enn = SMOTEENN(
        smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
    )
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    _print_distribution("SMOTE-ENN (Extra Baseline)", y_res)
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────
# 公共接口
# ─────────────────────────────────────────────────────────────────
def get_resampled(strategy_name: str,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  random_state: int = 42,
                  **kwargs):
    """
    统一调用接口

    Parameters
    ----------
    strategy_name : str
        策略名，必须是 STRATEGY_REGISTRY 中的 key:
        'baseline' | 'oversample' | 'undersample' | 'hybrid' | 'smoteenn'
    X_train, y_train : np.ndarray
        训练集特征与标签
    random_state : int
        随机种子（默认 42）
    **kwargs :
        传给各策略的额外参数（如 k_neighbors）

    Returns
    -------
    X_res, y_res : np.ndarray
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"未知策略: '{strategy_name}'。可用策略: {available}"
        )
    fn = STRATEGY_REGISTRY[strategy_name]
    return fn(X_train, y_train, random_state=random_state, **kwargs)


def list_strategies() -> list:
    """返回所有已注册的策略名列表"""
    return list(STRATEGY_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────
def _print_distribution(name: str, y: np.ndarray):
    """打印重采样后的类别分布"""
    counter = Counter(y)
    total   = len(y)
    print(f"\n[{name}]")
    print(f"  正常 (0): {counter[0]:>8,}  ({counter[0]/total*100:.2f}%)")
    print(f"  欺诈 (1): {counter[1]:>8,}  ({counter[1]/total*100:.2f}%)")
    print(f"  总计    : {total:>8,}")


# ─────────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_data, preprocess

    df = load_data(verbose=False)
    X_train, X_test, y_train, y_test = preprocess(df, verbose=False)

    print("可用策略:", list_strategies())
    for strat in ["baseline", "oversample", "undersample", "hybrid"]:
        get_resampled(strat, X_train, y_train)
