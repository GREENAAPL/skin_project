"""Training-label utilities for the seven-class image model."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def class_weights_from_labels(labels: Iterable[int], num_classes: int = 7) -> list[float]:
    """Return mean-normalised inverse-frequency weights from training labels."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    values = np.asarray(list(labels), dtype=int)
    if values.size == 0:
        raise ValueError("cannot compute class weights from empty labels")
    if values.min() < 0 or values.max() >= num_classes:
        raise ValueError(
            f"labels must be in [0, {num_classes - 1}], got "
            f"min={values.min()} max={values.max()}"
        )
    counts = np.bincount(values, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights /= weights.mean()
    return weights.tolist()
