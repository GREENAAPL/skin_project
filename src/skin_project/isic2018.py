"""Prepare the ISIC 2018 Task 3 data for image-only seven-class training.

The official files use one-hot class columns and may contain multiple photographs
of the same lesion. This module converts them to the project's standard metadata
schema and creates a stratified, lesion-disjoint train/development split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .preprocessing import class_weights_from_labels
from .utils import LABEL_COL, LESION_ID_COL, SAMPLE_ID_COL


ISIC2018_CLASS_CODES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
ISIC2018_CLASS_NAMES = {
    "MEL": "melanoma",
    "NV": "melanocytic_nevus",
    "BCC": "basal_cell_carcinoma",
    "AKIEC": "actinic_keratosis_or_bowen_disease",
    "BKL": "benign_keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
}
ISIC2018_CLASS_TO_INDEX = {
    code: index for index, code in enumerate(ISIC2018_CLASS_CODES)
}


def load_isic2018_frame(dataset_root: str | Path) -> tuple[pd.DataFrame, Path]:
    """Read, validate, and normalize the official ISIC 2018 Task 3 files."""
    root = Path(dataset_root)
    raw_root = root / "raw"
    image_dirs = [
        path
        for path in raw_root.rglob("ISIC2018_Task3_Training_Input")
        if path.is_dir()
    ]
    label_files = list(raw_root.rglob("ISIC2018_Task3_Training_GroundTruth.csv"))
    grouping_files = list(raw_root.rglob("ISIC2018_Task3_Training_LesionGroupings.csv"))
    if len(image_dirs) != 1 or len(label_files) != 1 or len(grouping_files) != 1:
        raise FileNotFoundError(
            "expected exactly one ISIC image directory, ground-truth CSV, and "
            "lesion-grouping CSV under "
            f"{raw_root}; found {len(image_dirs)}, {len(label_files)}, "
            f"{len(grouping_files)}"
        )
    image_dir = image_dirs[0]
    labels_path = label_files[0]
    groupings_path = grouping_files[0]

    labels = pd.read_csv(labels_path)
    groupings = pd.read_csv(groupings_path)
    required_label_columns = {"image", *ISIC2018_CLASS_CODES}
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_label_columns:
        raise KeyError(f"ground-truth CSV missing columns: {sorted(missing_label_columns)}")
    required_grouping_columns = {"image", LESION_ID_COL, "diagnosis_confirm_type"}
    missing_grouping_columns = required_grouping_columns - set(groupings.columns)
    if missing_grouping_columns:
        raise KeyError(f"lesion-grouping CSV missing columns: {sorted(missing_grouping_columns)}")
    if labels["image"].duplicated().any() or groupings["image"].duplicated().any():
        raise ValueError("ISIC image identifiers must be unique in both CSV files")

    one_hot = labels.loc[:, list(ISIC2018_CLASS_CODES)].apply(
        pd.to_numeric, errors="raise"
    )
    values = one_hot.to_numpy(dtype=float)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("ground-truth class columns must contain only 0 or 1")
    if not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("each image must have exactly one active diagnosis class")

    frame = pd.DataFrame(
        {
            SAMPLE_ID_COL: labels["image"].astype(str),
            LABEL_COL: values.argmax(axis=1).astype(int),
        }
    )
    frame["class_code"] = [ISIC2018_CLASS_CODES[index] for index in frame[LABEL_COL]]
    frame["class_name"] = frame["class_code"].map(ISIC2018_CLASS_NAMES)
    frame = frame.merge(
        groupings[["image", LESION_ID_COL, "diagnosis_confirm_type"]],
        left_on=SAMPLE_ID_COL,
        right_on="image",
        how="left",
        validate="one_to_one",
    ).drop(columns="image")

    if frame[LESION_ID_COL].isna().any():
        raise ValueError("some images have no lesion_id after joining the official CSV files")
    frame[LESION_ID_COL] = frame[LESION_ID_COL].astype(str)
    labels_per_lesion = frame.groupby(LESION_ID_COL)[LABEL_COL].nunique()
    if (labels_per_lesion > 1).any():
        raise ValueError("at least one lesion_id is associated with multiple diagnoses")

    image_ids = {path.stem for path in image_dir.glob("*.jpg")}
    metadata_ids = set(frame[SAMPLE_ID_COL])
    missing_images = metadata_ids - image_ids
    unlabeled_images = image_ids - metadata_ids
    if missing_images or unlabeled_images:
        raise ValueError(
            "image/label mismatch: "
            f"missing_images={len(missing_images)} unlabeled_images={len(unlabeled_images)}"
        )

    columns = [
        SAMPLE_ID_COL,
        LESION_ID_COL,
        LABEL_COL,
        "class_code",
        "class_name",
        "diagnosis_confirm_type",
    ]
    return frame.loc[:, columns].sort_values(SAMPLE_ID_COL).reset_index(drop=True), image_dir


def make_lesion_disjoint_split(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    dev_fold: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a stratified split with no lesion appearing on both sides."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 <= dev_fold < n_splits:
        raise ValueError(f"dev_fold must be in [0, {n_splits - 1}]")
    for column in (LABEL_COL, LESION_ID_COL):
        if column not in frame.columns:
            raise KeyError(f"frame missing required column {column!r}")

    # Stratify at the lesion level, not at the image level. Every lesion has one
    # diagnosis, while some lesions have multiple photographs.
    lesions = frame[[LESION_ID_COL, LABEL_COL]].drop_duplicates(LESION_ID_COL)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 / n_splits,
        random_state=seed + dev_fold,
    )
    train_lesion_indices, dev_lesion_indices = next(
        splitter.split(lesions, y=lesions[LABEL_COL])
    )
    train_lesions = set(lesions.iloc[train_lesion_indices][LESION_ID_COL])
    dev_lesions = set(lesions.iloc[dev_lesion_indices][LESION_ID_COL])
    train = (
        frame.loc[frame[LESION_ID_COL].isin(train_lesions)]
        .sort_values(SAMPLE_ID_COL)
        .reset_index(drop=True)
    )
    dev = (
        frame.loc[frame[LESION_ID_COL].isin(dev_lesions)]
        .sort_values(SAMPLE_ID_COL)
        .reset_index(drop=True)
    )

    overlap = set(train[LESION_ID_COL]) & set(dev[LESION_ID_COL])
    if overlap:
        raise AssertionError(f"lesion leakage detected: {len(overlap)} overlapping lesions")
    if len(train) + len(dev) != len(frame):
        raise AssertionError("train/dev split does not cover every image exactly once")
    return train, dev


def _split_summary(frame: pd.DataFrame) -> dict:
    counts = frame[LABEL_COL].value_counts().sort_index()
    return {
        "images": int(len(frame)),
        "lesions": int(frame[LESION_ID_COL].nunique()),
        "class_counts": {
            ISIC2018_CLASS_CODES[index]: int(counts.get(index, 0))
            for index in range(len(ISIC2018_CLASS_CODES))
        },
    }


def prepare_isic2018(
    dataset_root: str | Path,
    *,
    n_splits: int = 5,
    dev_fold: int = 0,
    seed: int = 42,
) -> dict:
    """Create project-ready train/dev CSV files and a reproducibility manifest."""
    root = Path(dataset_root)
    frame, image_dir = load_isic2018_frame(root)
    train, dev = make_lesion_disjoint_split(
        frame,
        n_splits=n_splits,
        dev_fold=dev_fold,
        seed=seed,
    )
    output_dir = root / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_metadata.csv", index=False)
    dev.to_csv(output_dir / "dev_metadata.csv", index=False)

    weights = class_weights_from_labels(
        train[LABEL_COL].tolist(), num_classes=len(ISIC2018_CLASS_CODES)
    )
    manifest = {
        "dataset": "ISIC 2018 Task 3",
        "source": "https://challenge.isic-archive.com/data/",
        "license": "CC BY-NC 4.0",
        "seed": seed,
        "n_splits": n_splits,
        "dev_fold": dev_fold,
        "image_dir": str(image_dir),
        "class_to_index": ISIC2018_CLASS_TO_INDEX,
        "train": _split_summary(train),
        "dev": _split_summary(dev),
        "train_class_weights": {
            code: float(weights[index])
            for index, code in enumerate(ISIC2018_CLASS_CODES)
        },
        "lesion_overlap": 0,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a lesion-disjoint ISIC 2018 seven-class split."
    )
    parser.add_argument("--root", default="data/isic2018")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--dev-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = prepare_isic2018(
        args.root,
        n_splits=args.n_splits,
        dev_fold=args.dev_fold,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
