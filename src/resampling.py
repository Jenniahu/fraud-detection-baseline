"""
resampling.py
=============
重采样策略模块

支持的策略:
  - baseline      : 不做任何重采样（原始不平衡数据）
  - oversample    : SMOTE 过采样
  - undersample   : SMOTE-ENN 混合采样（先 SMOTE 后 ENN 清理）
  - hybrid        : SMOTE-Tomek 混合采样（先 SMOTE 后 Tomek）
  - smoteenn      : SMOTE-ENN（与 undersample 相同，保留别名）

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
                   sampling_strategy: float = 0.05,  # 更轻度：5% 而非 10%
                   **kwargs):
    """
    Borderline-SMOTE 轻度过采样（优化版）
    - 仅对边界样本生成合成数据，避免噪声扩散
    - sampling_strategy=0.05，更保守的采样比例
    - k_neighbors=3，减少近邻数量避免引入噪声
    """
    from imblearn.over_sampling import BorderlineSMOTE

    smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=3,  # 减少近邻数，更严格
        random_state=random_state,
        kind='borderline-1',  # 只针对边界样本
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    _print_distribution(f"Borderline-SMOTE (strategy={sampling_strategy})", y_res)
    return X_res, y_res


@register("undersample")
def strategy_adasyn_clean(X_train: np.ndarray,
                          y_train: np.ndarray,
                          random_state: int = 42,
                          k_neighbors: int = 5,
                          **kwargs):
    """
    ADASYN + ENN 混合采样（优化版）
    - Step 1: ADASYN 自适应生成，对难分类样本生成更多合成数据
    - Step 2: ENN 清理噪声，保留干净样本
    - 比 SMOTE-ENN 更智能，避免均匀生成导致的噪声
    """
    from imblearn.over_sampling import ADASYN

    # 第一步：ADASYN 自适应过采样
    adasyn = ADASYN(
        sampling_strategy=0.05,  # 轻度采样
        n_neighbors=3,
        random_state=random_state,
    )
    X_temp, y_temp = adasyn.fit_resample(X_train, y_train)

    # 第二步：ENN 清理
    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours(
        n_neighbors=3,
        kind_sel='mode',
    )
    X_res, y_res = enn.fit_resample(X_temp, y_temp)

    _print_distribution("ADASYN+ENN (Undersample)", y_res)
    return X_res, y_res


@register("hybrid")
def strategy_smotetomek(X_train: np.ndarray,
                         y_train: np.ndarray,
                         random_state: int = 42,
                         k_neighbors: int = 5,
                         **kwargs):
    """
    SMOTE-Tomek 混合策略（保守参数优化版）
    - Step 1: 轻度 SMOTE (sampling_strategy=0.05)
    - Step 2: Tomek Links 清理边界噪声
    - 更保守的采样比例，避免过度合成
    """
    from imblearn.over_sampling import SMOTE

    smote_tomek = SMOTETomek(
        smote=SMOTE(
            sampling_strategy=0.05,  # 更保守
            k_neighbors=3,
            random_state=random_state
        ),
        random_state=random_state,
    )
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    _print_distribution("SMOTE-Tomek Conservative (Hybrid)", y_res)
    return X_res, y_res


@register("smoteenn")
def strategy_adasyn_alias(X_train: np.ndarray,
                          y_train: np.ndarray,
                          random_state: int = 42,
                          k_neighbors: int = 5,
                          **kwargs):
    """
    ADASYN+ENN（undersample 的别名，用于兼容）
    """
    return strategy_adasyn_clean(X_train, y_train, random_state, k_neighbors, **kwargs)


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
