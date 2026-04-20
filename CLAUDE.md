# CLAUDE.md - 欺诈检测实验开发指南

> 本项目：An Empirical Research on the Efficacy of Hybrid Resampling Strategies for Extremely Imbalanced Financial Fraud Detection

---

## 1. 项目概述

### 1.1 研究目标

验证 SMOTE-Tomek 混合重采样策略在极度不平衡金融欺诈检测中的有效性，对比单一重采样方法（SMOTE、Tomek Links）和原始数据基线。

### 1.2 核心研究问题 (RQ)

| RQ | 研究问题 | 验证方法 |
|----|---------|---------|
| **RQ1** | SMOTE-Tomek 混合策略对决策边界的清晰度有何协同效果？ | 对比 baseline/SMOTE/Tomek/SMOTE-Tomek 的 AUPRC 和 G-Mean |
| **RQ2** | 不同分类器（LR/RF/XGBoost）在重采样后的表现差异如何？ | 分析 4 种策略 × 3 种模型的 12 组实验结果 |
| **RQ3** | AUPRC 和 G-Mean 相比传统 Accuracy 是否更科学有效？ | 对比各指标在极度不平衡场景下的区分能力 |

### 1.3 数据集

- **来源**: Kaggle Credit Card Fraud Detection
- **规模**: 284,807 条交易记录，492 条欺诈（0.172%）
- **特征**: 30 维（Time, Amount, V1-V28 PCA 特征）
- **不平衡比**: 1:577

---

## 2. 实验流程

### 2.1 当前 Baseline 已完成

✅ **数据预处理**
- RobustScaler 标准化 Time/Amount
- 80/20 分层训练测试分割

✅ **重采样策略**
- `baseline`: 无重采样
- `oversample`: SMOTE 过采样
- `undersample`: Tomek Links 欠采样
- `hybrid`: SMOTE-Tomek 混合策略

✅ **分类器**
- Logistic Regression（线性基线）
- Random Forest（Bagging 集成）
- XGBoost（梯度提升）

✅ **评估指标**
- Precision / Recall / F1
- G-Mean（几何平均）
- AUPRC（主要指标）
- AUROC（辅助参考）

### 2.2 下一步实验计划

#### Phase 1: 阈值优化（当前 → Week 7-8）

当前默认使用 0.5 阈值，但极度不平衡场景下需要优化分类阈值以平衡 Recall 和 Precision。

**待实现功能**:
1. **阈值搜索**: 在 [0.1, 0.9] 范围内搜索最优阈值
2. **优化目标**: 最大化 F1-Score 或 G-Mean
3. **可视化**: 绘制阈值-指标曲线

**代码位置**: `src/evaluation.py` 新增 `find_optimal_threshold()`

#### Phase 2: 超参数调优（Week 7-8）

当前使用默认超参数，需进行系统调优。

**调优范围**:

| 模型 | 关键参数 | 搜索空间 |
|------|---------|---------|
| Logistic Regression | C | [0.01, 0.1, 1, 10, 100] |
| Random Forest | n_estimators, max_depth | [50, 100, 200], [5, 10, 20, None] |
| XGBoost | n_estimators, max_depth, learning_rate | [100, 200, 300], [3, 6, 9], [0.05, 0.1, 0.2] |

**实现方式**: GridSearchCV 或 Optuna

#### Phase 3: 交叉验证（Week 7-8）

当前单次分割可能引入随机性，需增加 Stratified K-Fold 交叉验证。

**配置**:
- K = 5
- 每层保持类别比例一致
- 报告均值 ± 标准差

#### Phase 4: 扩展对比实验（Week 9-10）

**新增对比基准**:
1. **SMOTE-ENN**: 已预留接口，需运行实验
2. **不同 SMOTE 比例**: 不仅 1:1，尝试 1:0.5, 1:0.3 等
3. **不同 k_neighbors**: SMOTE 的 KNN 邻居数影响（3, 5, 7）

**新增分类器**（可选）:
- LightGBM
- CatBoost

#### Phase 5: 结果分析与可视化（Week 9-10）

**待生成图表**:
1. 决策边界可视化（PCA 降维后）
2. 特征重要性分析（SHAP 值）
3. 不同 imbalance ratio 下的鲁棒性分析

---

## 3. 代码结构

```
fraud_detection/
├── src/
│   ├── data_loader.py      # 数据加载 & 预处理
│   ├── resampling.py       # 重采样策略（核心模块）
│   ├── models.py           # 分类器定义
│   ├── evaluation.py       # 评估指标（需扩展阈值优化）
│   └── visualization.py    # 可视化
├── notebooks/
│   └── baseline_experiment.ipynb  # 交互式实验
├── run_experiment.py       # 主实验入口
└── CLAUDE.md              # 本文件
```

### 3.1 关键接口

**运行实验**:
```bash
# 全部实验
python run_experiment.py

# 特定策略
python run_experiment.py --strategies baseline hybrid

# 特定模型
python run_experiment.py --models xgboost

# 静默模式
python run_experiment.py --quiet
```

**重采样调用**:
```python
from src.resampling import get_resampled

X_res, y_res = get_resampled("hybrid", X_train, y_train, random_state=42)
```

**模型构建**:
```python
from src.models import build_model

model = build_model("xgboost", n_estimators=300, max_depth=6)
```

**模型评估**:
```python
from src.evaluation import evaluate_model

metrics = evaluate_model(model, X_test, y_test, threshold=0.5)
# 返回: precision, recall, f1, g_mean, auprc, auroc
```

---

## 4. 开发规范

### 4.1 代码风格

- 使用 Python 3.10+
- 遵循 PEP 8
- 函数添加 docstring（Google 风格）
- 类型注解（`def func(x: np.ndarray) -> dict:`）

### 4.2 实验记录

每次实验后更新:
1. `results/experiment_results.csv` - 自动保存
2. `figures/` - 自动保存可视化
3. 手动记录关键发现到实验笔记

### 4.3 Git 提交规范

```
feat: 添加阈值优化功能
fix: 修复 SMOTE k_neighbors 参数传递问题  
docs: 更新 CLAUDE.md 实验计划
refactor: 重构 evaluation 模块
experiment: 完成 Phase 1 阈值优化实验
```

---

## 5. 关键指标解读

### 5.1 AUPRC（主要指标）

- **适用场景**: 极度不平衡数据（正样本 < 1%）
- **优点**: 对负样本数量不敏感，关注正样本识别能力
- **解读**: 0.5 表示随机水平，>0.7 表示较好，>0.9 表示优秀

### 5.2 G-Mean

- **公式**: √(TPR × TNR)
- **含义**: 正负类识别能力的几何平均
- **优点**: 平衡考虑两类性能，避免偏向多数类

### 5.3 Recall vs Precision 权衡

金融欺诈场景通常更关注 **Recall**（不漏掉欺诈），但需控制 **Precision**（减少误报成本）。

---

## 6. 常见问题

### Q1: 运行实验时内存不足？

SMOTE 会生成大量合成样本，若内存不足:
1. 减少 `k_neighbors`（默认 5 → 3）
2. 使用 `sampling_strategy` 参数限制过采样比例

### Q2: XGBoost 安装失败？

```bash
# macOS
brew install libomp
pip install xgboost

# 或使用 conda
conda install -c conda-forge xgboost
```

### Q3: 如何添加新的重采样策略？

在 `src/resampling.py` 中:
```python
@register("my_strategy")
def strategy_my(X_train, y_train, random_state=42, **kwargs):
    # 实现重采样逻辑
    return X_res, y_res
```

### Q4: 如何添加新的评估指标？

在 `src/evaluation.py` 的 `evaluate_model()` 中添加计算逻辑，并更新返回字典。

---

## 7. 实验检查清单

### 运行前
- [ ] 数据文件 `data/creditcard.csv` 已放置
- [ ] 依赖已安装 `pip install -r requirements.txt`
- [ ] 随机种子固定（默认 42）

### 运行后
- [ ] 检查 `results/experiment_results.csv` 是否生成
- [ ] 检查 `figures/` 图表是否正常
- [ ] 验证 AUPRC 指标是否符合预期（SMOTE-Tomek 应优于 baseline）

### 报告前
- [ ] 所有图表已整理
- [ ] 关键发现已记录
- [ ] 实验可复现（固定随机种子）

---

## 8. 参考资源

- **Proposal**: `proposal(1).pdf`
- **Baseline Notebook**: `notebooks/baseline_experiment.ipynb`
- **imbalanced-learn 文档**: https://imbalanced-learn.org/
- **XGBoost 文档**: https://xgboost.readthedocs.io/

---

*最后更新: 2026-04-20*
