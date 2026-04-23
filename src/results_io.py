"""
results_io.py
=============
实验结果保存与汇总打印模块。
"""

import os
import json
import pandas as pd
from src.config import RESULTS_DIR, OPTIMIZE_THRESHOLD, THRESHOLD_METRIC


# ─────────────────────────────────────────────────────────────────
# 结果保存
# ─────────────────────────────────────────────────────────────────
def _save_results(df: pd.DataFrame, basename: str = "experiment_results"):
    """将结果保存为 CSV 和 JSON"""
    csv_path  = os.path.join(RESULTS_DIR, f"{basename}.csv")
    json_path = os.path.join(RESULTS_DIR, f"{basename}.json")

    # CSV（不含内部列）
    save_cols = [c for c in df.columns if not c.startswith("_")]
    df[save_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  ✅ CSV  → {csv_path}")

    # JSON（便于程序读取）
    records = df[save_cols].to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON → {json_path}")


# ─────────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame):
    """打印实验汇总表"""
    print("\n" + "═" * 85)
    print("  实验汇总（按 AUPRC 降序排列）")
    print("═" * 85)
    display_cols = ["strategy", "model", "threshold", "recall", "precision",
                    "f1", "g_mean", "auprc", "auroc"]
    summary = df[display_cols].sort_values("auprc", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("═" * 85)

    # 最优组合
    best = summary.iloc[0]
    print(f"\n  🏆 最优组合 (AUPRC): "
          f"{best['strategy']} × {best['model']}  "
          f"阈值={best['threshold']:.3f}  "
          f"AUPRC={best['auprc']:.4f}  Recall={best['recall']:.4f}")

    # 阈值优化效果对比（如果有对比数据）
    if OPTIMIZE_THRESHOLD:
        print(f"\n  📊 阈值优化说明:")
        print(f"     每个 (策略, 模型) 组合在验证集上独立搜索最优阈值")
        print(f"     优化目标: {THRESHOLD_METRIC.upper()}")
        print(f"     确保各策略都处于最优决策点后再进行公平比较")
