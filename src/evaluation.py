from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .baselines import CLASS_ORDER


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    for idx, label in enumerate(CLASS_ORDER):
        metrics[f"precision_{label.lower()}"] = float(precision[idx])
        metrics[f"recall_{label.lower()}"] = float(recall[idx])
        metrics[f"f1_{label.lower()}"] = float(f1[idx])
        metrics[f"support_{label.lower()}"] = int(support[idx])
    return metrics


def predicted_distribution(y_pred: pd.Series) -> dict[str, float]:
    counts = y_pred.value_counts(normalize=True)
    return {f"pred_pct_{label.lower()}": float(counts.get(label, 0.0)) for label in CLASS_ORDER}


def confidence_summary(probabilities: pd.DataFrame) -> dict[str, float]:
    max_prob = probabilities.max(axis=1)
    return {
        "confidence_mean": float(max_prob.mean()),
        "confidence_std": float(max_prob.std(ddof=0)),
        "confidence_p10": float(max_prob.quantile(0.10)),
        "confidence_p50": float(max_prob.quantile(0.50)),
        "confidence_p90": float(max_prob.quantile(0.90)),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
