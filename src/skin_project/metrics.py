"""Classification metrics shared by training and evaluation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _as_arrays(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float] | None
):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    prob = None if y_prob is None else np.asarray(y_prob, dtype=float)
    return y_true, y_pred, prob


def sensitivity_specificity(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[float, float]:
    """Sensitivity (positive recall) and specificity (negative recall).

    Robust to a partition that happens to contain only one class.
    """
    y_true, y_pred, _ = _as_arrays(y_true, y_pred, None)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sensitivity), float(specificity)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return the full metric dictionary used across the project.

    Keys: macro_f1, balanced_accuracy, sensitivity, specificity, auroc, accuracy.
    `auroc` is NaN if probabilities are missing or only one class is present.
    """
    y_true, y_pred, prob = _as_arrays(y_true, y_pred, y_prob)

    observed = np.concatenate([y_true, y_pred]) if y_true.size else np.asarray([], dtype=int)
    max_label = int(observed.max()) if observed.size else 1
    metric_labels = list(range(max(2, max_label + 1)))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=metric_labels,
            average="macro",
            zero_division=0,
        )
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    is_binary = max_label <= 1
    sens, spec = (
        sensitivity_specificity(y_true, y_pred)
        if is_binary
        else (float("nan"), float("nan"))
    )
    accuracy = float((y_true == y_pred).mean()) if y_true.size else float("nan")

    auroc = float("nan")
    if prob is not None and is_binary and len(np.unique(y_true)) == 2:
        try:
            binary_prob = prob[:, 1] if prob.ndim == 2 else prob.ravel()
            auroc = float(roc_auc_score(y_true, binary_prob))
        except (ValueError, IndexError):
            auroc = float("nan")
    elif prob is not None and not is_binary and prob.ndim == 2:
        try:
            auroc = float(
                roc_auc_score(
                    y_true,
                    prob,
                    labels=metric_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            auroc = float("nan")

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "auroc": auroc,
        "accuracy": accuracy,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    order = ["macro_f1", "balanced_accuracy", "sensitivity", "specificity", "auroc", "accuracy"]
    parts = []
    for key in order:
        if key in metrics:
            val = metrics[key]
            parts.append(f"{key}={val:.4f}" if val == val else f"{key}=nan")
    return "  ".join(parts)
