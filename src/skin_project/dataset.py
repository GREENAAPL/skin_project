"""PyTorch dataset for image-only ISIC 2018 classification."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transforms
from .utils import LABEL_COL, LESION_ID_COL, SAMPLE_ID_COL


class SkinLesionDataset(Dataset):
    """Load an image by ``sample_id`` and return its seven-class label."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: str | Path,
        image_size: int = 224,
        train: bool = False,
        augment: dict | None = None,
        has_labels: bool = True,
        image_ext: str = ".jpg",
        brightness_factor: float = 1.0,
    ) -> None:
        if SAMPLE_ID_COL not in metadata.columns:
            raise KeyError(f"metadata must contain {SAMPLE_ID_COL!r}")
        if has_labels and LABEL_COL not in metadata.columns:
            raise KeyError(f"metadata must contain {LABEL_COL!r}")
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.has_labels = has_labels
        self.image_ext = str(image_ext)
        self.brightness_factor = float(brightness_factor)
        if train and self.brightness_factor != 1.0:
            raise ValueError("brightness_factor is for deterministic evaluation only")
        self.transform = build_transforms(
            self.image_size,
            train=train,
            augment=augment,
            brightness_factor=self.brightness_factor,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def image_path(self, sample_id: str) -> Path:
        return self.image_dir / f"{sample_id}{self.image_ext}"

    def _load_image(self, sample_id: str) -> Image.Image:
        path = self.image_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(f"image for sample {sample_id!r} not found at {path}")
        with Image.open(path) as image:
            image.load()
            return image.copy()

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        sample_id = str(row[SAMPLE_ID_COL])
        lesion_id = str(row.get(LESION_ID_COL, "unknown"))
        label = int(row[LABEL_COL]) if self.has_labels else -1
        return {
            "image": self.transform(self._load_image(sample_id)),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "lesion_id": lesion_id,
        }
