import pandas as pd

from skin_project.isic2018 import (
    ISIC2018_CLASS_CODES,
    make_lesion_disjoint_split,
)
from skin_project.preprocessing import class_weights_from_labels


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for label, code in enumerate(ISIC2018_CLASS_CODES):
        for lesion_number in range(10):
            lesion_id = f"{code}_{lesion_number}"
            for image_number in range(1 + (lesion_number % 2)):
                rows.append(
                    {
                        "sample_id": f"{lesion_id}_{image_number}",
                        "lesion_id": lesion_id,
                        "label": label,
                        "class_code": code,
                    }
                )
    return pd.DataFrame(rows)


def test_lesion_split_has_no_leakage_and_covers_every_image():
    frame = _synthetic_frame()
    train, dev = make_lesion_disjoint_split(frame, n_splits=5, dev_fold=0, seed=42)
    assert len(train) + len(dev) == len(frame)
    assert set(train["sample_id"]).isdisjoint(dev["sample_id"])
    assert set(train["lesion_id"]).isdisjoint(dev["lesion_id"])
    assert set(train["label"]) == set(range(7))
    assert set(dev["label"]) == set(range(7))


def test_seven_class_weights_are_finite_and_rare_class_is_larger():
    labels = [0] * 50 + [1] * 30 + [2] * 20 + [3] * 10 + [4] * 8 + [5] * 3 + [6]
    weights = class_weights_from_labels(labels, num_classes=7)
    assert len(weights) == 7
    assert weights[6] > weights[5] > weights[0]
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-7
