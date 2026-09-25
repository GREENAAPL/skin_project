"""Small synthetic seven-class fixtures; no ISIC download is needed for tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skin_project.config import Config
from skin_project.isic2018 import ISIC2018_CLASS_CODES


def _make_split(root, split: str, images_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    image_dir = root / f"{split}_images"
    image_dir.mkdir(parents=True)
    for label, code in enumerate(ISIC2018_CLASS_CODES):
        for number in range(images_per_class):
            sample_id = f"{split}_{code}_{number}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "lesion_id": f"lesion_{sample_id}",
                    "label": label,
                    "class_code": code,
                }
            )
            pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels, "RGB").save(image_dir / f"{sample_id}.jpg")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / f"{split}_metadata.csv", index=False)
    return frame


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("isic")
    train = _make_split(root, "train", images_per_class=2, seed=1)
    dev = _make_split(root, "dev", images_per_class=1, seed=2)
    return {"root": root, "train_df": train, "dev_df": dev}


@pytest.fixture
def tiny_config(tiny_dataset, tmp_path):
    return Config(
        {
            "experiment": {"run_name": "test", "seed": 0},
            "data": {
                "root": str(tiny_dataset["root"]),
                "train_metadata": "train_metadata.csv",
                "dev_metadata": "dev_metadata.csv",
                "train_images": "train_images",
                "dev_images": "dev_images",
                "image_size": 64,
                "image_ext": ".jpg",
                "num_workers": 0,
                "class_names": list(ISIC2018_CLASS_CODES),
            },
            "model": {
                "type": "efficientnet_b0",
                "backbone": "efficientnet_b0",
                "num_classes": 7,
                "pretrained": False,
                "freeze_encoder": True,
                "unfreeze_last_blocks": 0,
            },
            "training": {"batch_size": 4, "epochs": 1, "augment": {}},
            "logging": {
                "tensorboard_root": str(tmp_path / "tb"),
                "checkpoint_root": str(tmp_path / "ckpt"),
            },
        }
    )
