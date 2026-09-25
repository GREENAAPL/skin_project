"""DataLoader construction for lesion-disjoint train and validation splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SkinLesionDataset
from .utils import LABEL_COL, SAMPLE_ID_COL


def collate_samples(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "sample_id": [sample["sample_id"] for sample in batch],
        "lesion_id": [sample["lesion_id"] for sample in batch],
    }


def build_dataloader(
    dataset: SkinLesionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    worker_options = {}
    if num_workers > 0:
        worker_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        **worker_options,
    )


@dataclass
class DataBundle:
    train_loader: DataLoader
    dev_loader: DataLoader
    train_df: pd.DataFrame
    dev_df: pd.DataFrame


def _resolve(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def build_dataloaders_from_config(cfg: Config) -> DataBundle:
    data = cfg.data
    root = Path(data["root"])
    train_df = pd.read_csv(_resolve(root, data.get("train_metadata", "prepared/train_metadata.csv")))
    dev_df = pd.read_csv(_resolve(root, data.get("dev_metadata", "prepared/dev_metadata.csv")))
    for frame, split in ((train_df, "train"), (dev_df, "dev")):
        missing = {SAMPLE_ID_COL, LABEL_COL} - set(frame.columns)
        if missing:
            raise KeyError(f"{split} metadata missing columns: {sorted(missing)}")

    image_size = int(data.get("image_size", 320))
    image_ext = str(data.get("image_ext", ".jpg"))
    num_workers = int(data.get("num_workers", 0))
    train_dataset = SkinLesionDataset(
        train_df,
        _resolve(root, data.get("train_images", "raw/ISIC2018_Task3_Training_Input")),
        image_size=image_size,
        train=True,
        augment=cfg.training.get("augment", {}),
        image_ext=image_ext,
    )
    dev_dataset = SkinLesionDataset(
        dev_df,
        _resolve(root, data.get("dev_images", "raw/ISIC2018_Task3_Training_Input")),
        image_size=image_size,
        train=False,
        image_ext=image_ext,
    )
    batch_size = int(cfg.training.get("batch_size", 32))
    return DataBundle(
        train_loader=build_dataloader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=len(train_dataset) % batch_size == 1,
        ),
        dev_loader=build_dataloader(
            dev_dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        train_df=train_df,
        dev_df=dev_df,
    )
