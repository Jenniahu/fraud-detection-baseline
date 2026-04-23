"""
threshold_cache.py
==================
阈值缓存管理模块。

提供阈值缓存的加载、保存、读取与写入功能，
支持以 (strategy, model, metric) 为键的持久化存储。
"""

import os
import json
from src.config import THRESHOLD_CACHE_FILE


# ─────────────────────────────────────────────────────────────────
# 阈值缓存管理
# ─────────────────────────────────────────────────────────────────
def _load_threshold_cache(cache_file: str = THRESHOLD_CACHE_FILE) -> dict:
    """
    加载阈值缓存文件

    Returns
    -------
    dict: { (strategy, model, metric): threshold, ... }
    """
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        # 转换 key 从字符串元组格式
        return {tuple(k.split('|')): v for k, v in cache.items()}
    except Exception:
        return {}


def _save_threshold_cache(cache: dict, cache_file: str = THRESHOLD_CACHE_FILE):
    """
    保存阈值缓存到文件

    Parameters
    ----------
    cache : dict
        { (strategy, model, metric): threshold, ... }
    """
    # 转换 key 为可序列化的字符串格式
    serializable_cache = {'|'.join(k): v for k, v in cache.items()}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_cache, f, indent=2, ensure_ascii=False)


def _get_cached_threshold(strategy: str, model: str, metric: str,
                          cache: dict = None) -> float:
    """
    从缓存中获取阈值

    Returns
    -------
    float or None: 如果缓存存在返回阈值，否则返回 None
    """
    if cache is None:
        cache = _load_threshold_cache()
    key = (strategy, model, metric)
    return cache.get(key)


def _cache_threshold(strategy: str, model: str, metric: str,
                     threshold: float, cache: dict = None):
    """
    将阈值存入缓存并保存到文件
    """
    if cache is None:
        cache = _load_threshold_cache()
    key = (strategy, model, metric)
    cache[key] = round(threshold, 6)
    _save_threshold_cache(cache)
