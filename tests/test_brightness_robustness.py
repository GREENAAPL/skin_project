import math

import numpy as np
from PIL import Image

from skin_project.robustness import add_drop_columns, build_summary, parse_brightness_factors
from skin_project.transforms import build_eval_transforms


def test_eval_brightness_is_deterministic_and_ordered():
    image = Image.fromarray(np.full((12, 12, 3), 100, dtype=np.uint8), mode="RGB")
    dark_transform = build_eval_transforms(12, brightness_factor=0.5)
    clean_transform = build_eval_transforms(12, brightness_factor=1.0)
    bright_transform = build_eval_transforms(12, brightness_factor=1.5)

    dark = dark_transform(image)
    clean = clean_transform(image)
    bright = bright_transform(image)

    assert dark.mean() < clean.mean() < bright.mean()
    assert np.array_equal(clean.numpy(), clean_transform(image).numpy())


def test_brightness_factors_require_clean_reference():
    assert parse_brightness_factors("1.5, 1.0, 0.5, 1.0") == [0.5, 1.0, 1.5]

    try:
        parse_brightness_factors("0.5,1.5")
    except ValueError as exc:
        assert "1.0" in str(exc)
    else:
        raise AssertionError("a missing clean reference should fail")


def test_drop_columns_and_summary_use_factor_one_as_reference():
    rows = [
        {
            "brightness_factor": 0.5,
            "macro_f1": 0.60,
            "balanced_accuracy": 0.62,
            "sensitivity": 0.50,
            "specificity": 0.74,
            "auroc": 0.70,
            "accuracy": 0.62,
        },
        {
            "brightness_factor": 1.0,
            "macro_f1": 0.80,
            "balanced_accuracy": 0.81,
            "sensitivity": 0.82,
            "specificity": 0.80,
            "auroc": 0.90,
            "accuracy": 0.81,
        },
        {
            "brightness_factor": 1.5,
            "macro_f1": 0.70,
            "balanced_accuracy": 0.71,
            "sensitivity": 0.69,
            "specificity": 0.73,
            "auroc": 0.79,
            "accuracy": 0.71,
        },
    ]

    frame = add_drop_columns(rows)
    dark = frame.loc[frame["brightness_factor"] == 0.5].iloc[0]
    assert math.isclose(dark["macro_f1_drop"], 0.20)
    assert math.isclose(dark["macro_f1_relative_drop_pct"], 25.0)

    summary = build_summary(frame)
    assert summary["clean_macro_f1"] == 0.80
    assert summary["worst_brightness_factor"] == 0.5
    assert math.isclose(summary["maximum_macro_f1_drop"], 0.20)
