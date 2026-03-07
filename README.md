# 欺诈检测 Baseline 实验框架

**An Empirical Research on the Efficacy of Hybrid Resampling Strategies for Extremely Imbalanced Financial Fraud Detection**

Group 3 | HU Chenjing · HE Minghao · SUN Fanhong · LI Mingzheng · LING Yupeng

---

## 项目结构

```
fraud_detection/
├── data/                        # 数据目录（放 creditcard.csv）
├── src/
│   ├── data_loader.py           # 数据加载 & 预处理（RobustScaler + stratified split）
│   ├── resampling.py            # 重采样策略模块（baseline/SMOTE/Tomek/SMOTE-Tomek）
│   ├── models.py                # 分类器定义（LR/RF/XGBoost）
│   ├── evaluation.py            # 评估指标（Precision/Recall/F1/G-Mean/AUPRC/AUROC）
│   └── visualization.py        # 结果可视化（PR曲线/ROC曲线/混淆矩阵/条形图）
├── notebooks/
│   └── baseline_experiment.ipynb  # Jupyter 交互实验版本（推荐使用）
├── results/                     # 实验结果输出（CSV/JSON）
├── figures/                     # 生成的图片
├── run_experiment.py            # 主实验入口脚本
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

从 Kaggle 下载 [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 数据集，
将 `creditcard.csv` 放入 `data/` 目录。

> 若本地无数据，框架会自动尝试从 TensorFlow 镜像下载（约 66MB）

### 3. 运行实验

**方式一：命令行（全矩阵实验）**

```bash
# 运行全部 4×3=12 组实验
python run_experiment.py

# 只测试特定策略
python run_experiment.py --strategies baseline hybrid

# 只测试特定模型
python run_experiment.py --models xgboost random_forest

# 静默模式
python run_experiment.py --quiet
```

**方式二：Jupyter Notebook（推荐，支持交互可视化）**

```bash
jupyter notebook notebooks/baseline_experiment.ipynb
```

---

## 实验设计

### 研究问题

| RQ | 问题 |
|----|------|
| RQ1 | SMOTE-Tomek 混合策略对决策边界的清晰度有何协同效果？ |
| RQ2 | 不同分类器（LR/RF/XGBoost）在重采样后的表现差异如何？ |
| RQ3 | AUPRC 和 G-Mean 相比传统 Accuracy 是否更科学有效？ |

### 重采样策略

| 策略名 | 方法 | 说明 |
|--------|------|------|
| `baseline` | 无重采样 | 原始不平衡数据，作为对照组 |
| `oversample` | SMOTE | KNN 合成少数类样本 |
| `undersample` | Tomek Links | 移除决策边界的多数类噪声样本 |
| `hybrid` | SMOTE-Tomek | 先 SMOTE 后 Tomek，**Proposal 核心研究对象** |

### 分类器

| 模型 | 类型 | 说明 |
|------|------|------|
| `logistic_regression` | 线性 | 可解释基线 |
| `random_forest` | Bagging 集成 | 对噪声鲁棒 |
| `xgboost` | Boosting 集成 | 梯度提升框架 |

### 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision | TP / (TP+FP) | 预测欺诈中真正欺诈的比例 |
| Recall | TP / (TP+FN) | 欺诈被成功识别的比例 |
| F1 | 2×P×R / (P+R) | 精确率与召回率的调和平均 |
| **G-Mean** | √(TPR×TNR) | 正负类平衡性能 |
| **AUPRC** | PR 曲线下面积 | **主要评估指标**，适合极度不平衡场景 |
| AUROC | ROC 曲线下面积 | 辅助参考 |

---

## 输出文件

### results/

- `experiment_results.csv` — 全部 12 组实验结果表
- `experiment_results.json` — 同上（JSON 格式，便于程序读取）

### figures/

| 文件名 | 内容 |
|--------|------|
| `01_class_distribution.png` | 原始数据类别分布（饼图+柱状图） |
| `02_resampling_comparison.png` | 各重采样策略后的样本量对比 |
| `03_pr_*.png` | 各策略的 PR 曲线（三个模型叠加） |
| `04_roc_*.png` | 各策略的 ROC 曲线 |
| `05_metrics_comparison.png` | 全量指标对比条形图 |

---

## 后续扩展方向

- [ ] 超参数调优（GridSearchCV / Optuna）
- [ ] 分类阈值优化（最大化 F1 或 G-Mean）
- [ ] SHAP 特征重要性解释
- [ ] SMOTE-ENN 额外对比基准
- [ ] 不同 imbalance ratio 下的鲁棒性分析
- [ ] 交叉验证（Stratified K-Fold）
- [ ] LightGBM / CatBoost 额外分类器对比

---

## 技术栈

- Python 3.10+
- scikit-learn · imbalanced-learn · XGBoost
- Pandas · NumPy · Matplotlib · Seaborn
- Jupyter Notebook
