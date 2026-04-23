"""
experiment_tracker.py
=====================
实验结果追踪与自动文件命名模块。

根据实验配置自动生成语义化的结果文件名，
并记录实验元数据以便后续追溯。
"""

import os
import json
from datetime import datetime
from src.config import RESULTS_DIR, DEFAULT_STRATEGIES, DEFAULT_MODELS


def _generate_basename(strategies, models, amount_transform=None, use_amount_weight=False):
    """
    根据实验配置生成结果文件的基础名。

    命名规则（与已有历史约定保持一致）:
      - 默认配置（DEFAULT_STRATEGIES × DEFAULT_MODELS，无特殊变换）:
          experiment_results
      - 非默认配置: experiment_results_{描述后缀}
    """
    is_default_strats = set(strategies) == set(DEFAULT_STRATEGIES)
    is_default_models = set(models) == set(DEFAULT_MODELS)
    is_default = (
        is_default_strats and is_default_models
        and amount_transform is None and not use_amount_weight
    )

    if is_default:
        return "experiment_results"

    parts = ["experiment_results"]

    # 策略描述
    if len(strategies) == 1:
        parts.append(strategies[0])
    elif not is_default_strats:
        parts.append(f"{len(strategies)}strats")

    # 模型描述
    if len(models) == 1:
        parts.append(models[0])
    elif not is_default_models:
        parts.append(f"{len(models)}models")

    # 数据变换
    if amount_transform:
        parts.append(amount_transform)

    # 金额权重
    if use_amount_weight:
        parts.append("weighted")

    return "_".join(parts)


def save_metadata(basename, strategies, models, **kwargs):
    """
    保存实验元数据到 {basename}_metadata.json

    Parameters
    ----------
    basename : str
        结果文件基础名
    strategies : list
        使用的策略列表
    models : list
        使用的模型列表
    **kwargs :
        额外要记录的实验参数（如 random_seed, amount_transform 等）
    """
    metadata = {
        "basename": basename,
        "timestamp": datetime.now().isoformat(),
        "strategies": list(strategies),
        "models": list(models),
        **kwargs,
    }

    meta_path = os.path.join(RESULTS_DIR, f"{basename}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metadata → {meta_path}")


def list_experiments():
    """
    列出 results/ 目录下所有带 metadata 的实验。

    Returns
    -------
    list[dict]
        按时间戳降序排列的实验元数据列表
    """
    experiments = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_metadata.json"):
            path = os.path.join(RESULTS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    experiments.append(json.load(f))
            except Exception:
                continue
    return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)
