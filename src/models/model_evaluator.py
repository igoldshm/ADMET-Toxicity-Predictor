"""
Model Evaluation Utilities
===========================
Centralises metric computation, ROC/PR curve plotting, and per-task
result tables so pipeline.py and notebooks share identical evaluation logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def per_task_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_weight: np.ndarray,
    task_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-task AUC-ROC, AUC-PR, F1, and balanced accuracy.

    Handles missing labels (w=0) by masking per-task.

    Parameters
    ----------
    y_true   : (n_samples, n_tasks) ground truth labels
    y_prob   : (n_samples, n_tasks) predicted probabilities
    y_weight : (n_samples, n_tasks) sample weights (0 = missing)
    task_names : list of task name strings
    threshold  : probability threshold for binary classification

    Returns
    -------
    pd.DataFrame with one row per task.
    """
    tasks = task_names or [f"task_{i}" for i in range(y_true.shape[1])]
    rows  = []

    for i, task in enumerate(tasks):
        mask    = y_weight[:, i] != 0
        n_valid = mask.sum()

        if n_valid < 10:
            logger.warning("Task %s: only %d labelled samples — skipping", task, n_valid)
            rows.append({"task": task, "n": n_valid, "AUC-ROC": np.nan,
                         "AUC-PR": np.nan, "F1": np.nan, "BalAcc": np.nan})
            continue

        yt  = y_true[mask, i].astype(int)
        yp  = y_prob[mask, i]
        yh  = (yp >= threshold).astype(int)

        if len(np.unique(yt)) < 2:
            logger.warning("Task %s: only one class in labelled subset", task)
            rows.append({"task": task, "n": n_valid, "AUC-ROC": np.nan,
                         "AUC-PR": np.nan, "F1": np.nan, "BalAcc": np.nan})
            continue

        rows.append({
            "task":    task,
            "n":       n_valid,
            "pos_rate": round(yt.mean(), 4),
            "AUC-ROC": round(roc_auc_score(yt, yp), 4),
            "AUC-PR":  round(average_precision_score(yt, yp), 4),
            "F1":      round(f1_score(yt, yh, zero_division=0), 4),
            "BalAcc":  round(balanced_accuracy_score(yt, yh), 4),
        })

    df = pd.DataFrame(rows)
    summary_row = pd.DataFrame([{
        "task":    "MEAN",
        "n":       int(df["n"].sum()),
        "pos_rate": round(df["pos_rate"].mean(), 4) if "pos_rate" in df else np.nan,
        "AUC-ROC": round(df["AUC-ROC"].mean(), 4),
        "AUC-PR":  round(df["AUC-PR"].mean(), 4),
        "F1":      round(df["F1"].mean(), 4),
        "BalAcc":  round(df["BalAcc"].mean(), 4),
    }])
    df = pd.concat([df, summary_row], ignore_index=True)
    return df


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_weight: np.ndarray,
    task_names: Optional[List[str]] = None,
    title: str = "ROC Curves — All Tasks",
    save_path: Optional[Path] = None,
) -> None:
    """Plot per-task ROC curves on a single figure."""
    tasks  = task_names or [f"task_{i}" for i in range(y_true.shape[1])]
    n_cols = 4
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flat

    cmap   = plt.cm.get_cmap("tab20", len(tasks))

    for i, (task, ax) in enumerate(zip(tasks, axes)):
        mask = y_weight[:, i] != 0
        yt   = y_true[mask, i].astype(int)
        yp   = y_prob[mask, i]

        if len(np.unique(yt)) < 2 or mask.sum() < 10:
            ax.text(0.5, 0.5, "Insufficient\nlabelled data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(task, fontsize=9)
            continue

        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=cmap(i), lw=1.8, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.08, color=cmap(i))
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.set_title(task, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    # Hide unused axes
    for ax in list(axes)[len(tasks):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)
    plt.show()


def plot_model_comparison(
    results: Dict[str, pd.DataFrame],
    metric: str = "AUC-ROC",
    save_path: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing multiple models across all tasks.

    Parameters
    ----------
    results : dict of {model_name: per_task_metrics DataFrame}
    """
    task_filter = [t for t in TOX21_TASKS]
    models = list(results.keys())
    x = np.arange(len(task_filter))
    width = 0.8 / len(models)
    cmap = plt.cm.get_cmap("Set2", len(models))

    fig, ax = plt.subplots(figsize=(16, 5))
    for j, (model_name, df) in enumerate(results.items()):
        vals = [
            df.loc[df["task"] == t, metric].values[0] if t in df["task"].values else 0.0
            for t in task_filter
        ]
        offset = (j - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model_name, color=cmap(j), alpha=0.9, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(task_filter, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim([0.5, 1.0])
    ax.set_title(f"Model Comparison — {metric} per Tox21 Task")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="Random")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
