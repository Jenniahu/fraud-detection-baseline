"""
models.py
=========
分类器定义模块

包含三种 Proposal 规定的分类器:
  - LogisticRegression  (线性可解释基线)
  - RandomForestClassifier (集成 Bagging)
  - XGBClassifier (梯度提升)

提供统一的构建接口 build_model(name, **kwargs) 和超参数默认配置。
"""

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier


# ─────────────────────────────────────────────────────────────────
# 默认超参数配置（可通过 run_experiment.py 的 model_params 覆盖）
# ─────────────────────────────────────────────────────────────────
DEFAULT_PARAMS: dict = {

    "logistic_regression": {
        "max_iter"    : 1000,       # 足够的迭代轮次保证收敛
        "solver"      : "lbfgs",    # 适合小/中型数据集
        "C"           : 1.0,        # 正则化强度（1/lambda）
        "class_weight": None,       # 不使用内置权重（由重采样处理不平衡）
        "random_state": 42,
        "n_jobs"      : -1,
    },

    "random_forest": {
        "n_estimators"    : 100,    # 树的数量
        "max_depth"       : None,   # 不限深度（可改为 10/20 防过拟合）
        "min_samples_leaf": 2,      # 叶节点最小样本数
        "class_weight"    : None,
        "random_state"    : 42,
        "n_jobs"          : -1,
    },

    "xgboost": {
        "n_estimators"     : 200,
        "max_depth"        : 6,
        "learning_rate"    : 0.1,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "scale_pos_weight" : 577,   # 处理不平衡: 多数类/少数类 ≈ 284315/492
        "use_label_encoder": False,
        "eval_metric"      : "logloss",
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    },
}

# 模型名 -> sklearn 类 的映射
MODEL_REGISTRY: dict = {
    "logistic_regression": LogisticRegression,
    "random_forest"      : RandomForestClassifier,
    "xgboost"            : XGBClassifier,
}


# ─────────────────────────────────────────────────────────────────
# 公共接口
# ─────────────────────────────────────────────────────────────────
def build_model(model_name: str, **override_params):
    """
    构建分类器实例。

    Parameters
    ----------
    model_name : str
        模型名，必须是 MODEL_REGISTRY 中的 key:
        'logistic_regression' | 'random_forest' | 'xgboost'
    **override_params :
        覆盖默认超参数（例如 n_estimators=200）
        特殊参数:
        - y_train: 传入训练集标签，自动计算 xgboost 的 scale_pos_weight

    Returns
    -------
    sklearn-compatible classifier instance
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知模型: '{model_name}'。可用模型: {available}"
        )

    cls    = MODEL_REGISTRY[model_name]
    params = {**DEFAULT_PARAMS[model_name], **override_params}

    # 如果传入 y_train，自动计算 scale_pos_weight
    if model_name == "xgboost" and "y_train" in params:
        y_train = params.pop("y_train")
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        params["scale_pos_weight"] = n_neg / n_pos
        print(f"[XGBoost] 自动计算 scale_pos_weight: {n_neg}/{n_pos} = {n_neg/n_pos:.2f}")

    return cls(**params)


def list_models() -> list:
    """返回所有已注册的模型名列表"""
    return list(MODEL_REGISTRY.keys())


def get_default_params(model_name: str) -> dict:
    """返回指定模型的默认超参数字典（副本）"""
    if model_name not in DEFAULT_PARAMS:
        raise ValueError(f"未知模型: '{model_name}'")
    return dict(DEFAULT_PARAMS[model_name])


# ─────────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in list_models():
        m = build_model(name)
        print(f"[{name}]  {m}")
