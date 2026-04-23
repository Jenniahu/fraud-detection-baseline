"""
ensemble_sampler.py
===================
集成增强训练器：将 resampling.py 中的策略嵌入每轮 AdaBoost 迭代。

核心思想：
  每轮迭代前，对训练集重新执行指定的重采样策略，训练一个弱学习器，
  然后根据预测误差更新样本权重，进入下一轮。

这解决了"先采样后训练"的缺陷：
  - 噪声不会被固定放大（每轮采样不同）
  - 模型更稳健（多轮不同的采样子集平均化偏差）

新增: EasyEnsembleClassifier
  针对单一欠采样（RUS）导致信息丢失的问题，采用 Bagging 思想，
  通过多次随机欠采样生成多个平衡子集并行训练，最后集成预测。
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from src.resampling import get_resampled


class ResampleBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    自定义 Boosting：每轮迭代前调用 resampling.py 的策略重采样。

    Parameters
    ----------
    strategy : str
        resampling.py 中注册的策略名: 'baseline'|'oversample'|'undersample'|'hybrid'
    n_estimators : int
        Boosting 迭代轮数（弱学习器数量）
    learning_rate : float
        每轮弱学习器的权重收缩系数
    max_depth : int
        弱学习器（决策树）的最大深度
    random_state : int
        随机种子
    """

    def __init__(self,
                 strategy: str = "oversample",
                 n_estimators: int = 50,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 random_state: int = 42):
        self.strategy = strategy
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """
        训练：每轮迭代前重采样，训练决策树，更新样本权重。

        Parameters
        ----------
        sample_weight : np.ndarray, optional
            外部传入的样本权重（如金额权重），与 AdaBoost 内部权重结合。
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # 初始化样本权重：外部权重（如金额权重）与均匀权重结合
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            sample_weight = sample_weight / sample_weight.sum()
        else:
            sample_weight = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.errors_ = []

        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0 or i == self.n_estimators - 1:
                print(f"  ResampleBoost 第 {i+1}/{self.n_estimators} 轮 …")
            # ── 每轮迭代的核心：按策略重采样 ──
            # 根据当前样本权重进行加权随机采样
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weight
            )
            X_weighted = X[indices]
            y_weighted = y[indices]

            # 调用 resampling.py 中的策略进行重采样
            X_res, y_res = get_resampled(
                self.strategy,
                X_weighted,
                y_weighted,
                random_state=self.random_state + i  # 每轮不同的种子
            )

            # 训练弱学习器
            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + i
            )
            estimator.fit(X_res, y_res)

            # 在原始数据上评估误差（用于更新权重）
            y_pred = estimator.predict(X)
            incorrect = (y_pred != y)

            # 计算加权错误率
            error = np.dot(sample_weight, incorrect)

            # 防止除零和极端情况
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # 计算弱学习器权重
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / error)

            # 更新样本权重（提升错分样本）
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= sample_weight.sum()  # 归一化

            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.errors_.append(error)

        return self

    def predict_proba(self, X):
        """
        预测概率：加权平均所有弱学习器的预测。
        """
        proba_sum = np.zeros((X.shape[0], 2))

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            proba = estimator.predict_proba(X)
            proba_sum += weight * proba

        # 归一化
        proba_sum /= np.sum(proba_sum, axis=1, keepdims=True)
        return proba_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ─────────────────────────────────────────────────────────────────
# EasyEnsemble: 多次随机欠采样 + Bagging 集成
# ─────────────────────────────────────────────────────────────────
class EasyEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    EasyEnsemble 分类器

    针对单一欠采样（RUS）导致信息丢失的问题，采用 Bagging 思想：
      1. 对多数类进行多次随机欠采样，每次生成一个平衡子集
      2. 每个平衡子集独立训练一个 AdaBoost 分类器
      3. 预测时对所有基学习器做平均（Bagging 集成）

    Parameters
    ----------
    n_subsets : int
        生成的平衡子集数量（默认 10）
    n_estimators : int
        每个子集上 AdaBoost 的弱学习器数量（默认 50）
    learning_rate : float
        AdaBoost 学习率（默认 0.1）
    max_depth : int
        弱学习器（决策树）的最大深度（默认 3）
    random_state : int
        随机种子
    n_jobs : int
        并行作业数（当前通过子集间并行实现）
    """

    def __init__(self,
                 n_subsets: int = 10,
                 n_estimators: int = 50,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 random_state: int = 42,
                 n_jobs: int = -1):
        self.n_subsets = n_subsets
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        训练：生成多个平衡子集，每个子集训练一个 AdaBoost。
        """
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)

        # 分离多数类与少数类
        minority_class = np.bincount(y).argmin()
        majority_class = np.bincount(y).argmax()

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        n_minority = len(minority_indices)
        n_majority = len(majority_indices)

        self.estimators_ = []
        self.classes_ = np.unique(y)

        print(f"[EasyEnsemble] 多数类({majority_class}): {n_majority:,} | "
              f"少数类({minority_class}): {n_minority:,} | "
              f"子集数: {self.n_subsets}")

        for i in range(self.n_subsets):
            # 随机欠采样：从多数类中随机选取与少数类等量的样本
            sampled_majority = np.random.choice(
                majority_indices,
                size=n_minority,
                replace=False,
            )

            # 组合平衡子集
            subset_indices = np.concatenate([minority_indices, sampled_majority])
            np.random.shuffle(subset_indices)  # 打乱顺序

            X_subset = X[subset_indices]
            y_subset = y[subset_indices]

            # 训练 AdaBoost（作为基学习器）
            estimator = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                ),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state + i,
            )
            estimator.fit(X_subset, y_subset)
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        """
        预测概率：所有基学习器预测概率的平均（Bagging）。
        """
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))

        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            proba_sum += proba

        proba_sum /= len(self.estimators_)
        return proba_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ─────────────────────────────────────────────────────────────────
# EasyEnsembleWrapper: 将任意分类器包装成 EasyEnsemble 集成
# ─────────────────────────────────────────────────────────────────
class EasyEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    EasyEnsemble 包装器

    将任意 sklearn-compatible 分类器包装成
    "多次随机欠采样 + 多模型 Bagging 集成" 的形式。

    这样 EasyEnsemble 可以作为"采样策略"配合不同基学习器使用，
    例如: easy_ensemble + random_forest, easy_ensemble + xgboost 等。

    Parameters
    ----------
    base_estimator : sklearn estimator
        基学习器实例（如 RandomForestClassifier、XGBClassifier）
    n_subsets : int
        生成的平衡子集数量（默认 10）
    random_state : int
        随机种子
    """

    def __init__(self,
                 base_estimator,
                 n_subsets: int = 10,
                 random_state: int = 42):
        self.base_estimator = base_estimator
        self.n_subsets = n_subsets
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """
        训练：生成多个平衡子集，每个子集克隆并训练 base_estimator。

        Parameters
        ----------
        sample_weight : np.ndarray, optional
            外部传入的样本权重（如金额权重）。
        """
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)

        # 分离多数类与少数类
        minority_class = np.bincount(y).argmin()
        majority_class = np.bincount(y).argmax()

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        n_minority = len(minority_indices)
        n_majority = len(majority_indices)

        self.estimators_ = []
        self.classes_ = np.unique(y)

        print(f"[EasyEnsembleWrapper] 多数类({majority_class}): {n_majority:,} | "
              f"少数类({minority_class}): {n_minority:,} | "
              f"子集数: {self.n_subsets} | "
              f"基学习器: {type(self.base_estimator).__name__}")

        for i in range(self.n_subsets):
            # 随机欠采样：从多数类中随机选取与少数类等量的样本
            sampled_majority = np.random.choice(
                majority_indices,
                size=n_minority,
                replace=False,
            )

            # 组合平衡子集
            subset_indices = np.concatenate([minority_indices, sampled_majority])
            np.random.shuffle(subset_indices)

            X_subset = X[subset_indices]
            y_subset = y[subset_indices]

            # 克隆基学习器并训练（避免污染原始实例）
            estimator = clone(self.base_estimator)
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[subset_indices]
            estimator.fit(X_subset, y_subset, **fit_kwargs)
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        """
        预测概率：所有基学习器预测概率的平均（Bagging）。
        """
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))

        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            proba_sum += proba

        proba_sum /= len(self.estimators_)
        return proba_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
