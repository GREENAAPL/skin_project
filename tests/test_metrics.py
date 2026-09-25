"""Metric tests (README §22)."""
import math

from skin_project.metrics import compute_metrics, sensitivity_specificity


def test_perfect_predictions():
    y = [0, 1, 0, 1]
    m = compute_metrics(y, y, [0.1, 0.9, 0.2, 0.8])
    assert math.isclose(m["macro_f1"], 1.0)
    assert math.isclose(m["balanced_accuracy"], 1.0)
    assert math.isclose(m["auroc"], 1.0)


def test_sensitivity_specificity_directions():
    # all-positive prediction: sensitivity 1, specificity 0
    sens, spec = sensitivity_specificity([0, 0, 1, 1], [1, 1, 1, 1])
    assert sens == 1.0 and spec == 0.0


def test_auroc_nan_when_single_class():
    m = compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])
    assert m["auroc"] != m["auroc"]  # NaN


def test_macro_f1_penalizes_majority_only():
    # dataset 75% negative; always predict negative
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["macro_f1"] < 0.5  # macro-F1 punishes ignoring the positive class


def test_multiclass_macro_f1_includes_all_classes():
    y_true = [0, 1, 2, 3, 4, 5, 6]
    y_pred = [0, 1, 2, 3, 4, 5, 5]
    m = compute_metrics(y_true, y_pred)
    assert 0.0 < m["macro_f1"] < 1.0
    assert m["sensitivity"] != m["sensitivity"]  # binary-only metric -> NaN
    assert m["specificity"] != m["specificity"]


def test_multiclass_auroc_accepts_probability_matrix():
    y_true = [0, 1, 2, 0, 1, 2]
    probabilities = [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]
    m = compute_metrics(y_true, y_true, probabilities)
    assert math.isclose(m["auroc"], 1.0)
