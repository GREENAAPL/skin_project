"""Offline inference for unlabeled skin-lesion images."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .checkpoint import load_checkpoint, rebuild_from_checkpoint
from .dataloaders import build_dataloader
from .dataset import SkinLesionDataset
from .evaluation import predict_loader, predictions_dataframe
from .isic2018 import ISIC2018_CLASS_CODES
from .utils import SAMPLE_ID_COL, get_device


@torch.no_grad()
def run_inference(
    checkpoint: str | Path,
    image_list_csv: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    image_ext: str = ".jpg",
) -> pd.DataFrame:
    device = device or get_device()
    payload = load_checkpoint(checkpoint, map_location=device)
    model, image_size = rebuild_from_checkpoint(payload, device=device)
    frame = pd.read_csv(image_list_csv)
    if SAMPLE_ID_COL not in frame.columns:
        raise KeyError(f"input CSV must contain {SAMPLE_ID_COL!r}")
    dataset = SkinLesionDataset(
        frame,
        image_dir,
        image_size=image_size,
        train=False,
        has_labels=False,
        image_ext=image_ext,
    )
    loader = build_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    predictions = predict_loader(model, loader, device)
    num_classes = int(payload["model_config"].get("num_classes", 7))
    class_names = list(payload.get("class_names", []))
    if len(class_names) != num_classes:
        class_names = (
            list(ISIC2018_CLASS_CODES)
            if num_classes == len(ISIC2018_CLASS_CODES)
            else [f"class_{index}" for index in range(num_classes)]
        )
    result = predictions_dataframe(predictions, class_names).drop(columns=["label"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"wrote {len(result)} predictions -> {output_csv}")
    return result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EfficientNet-B0 inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-list", required=True, help="CSV containing sample_id values")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    run_inference(
        args.checkpoint,
        args.image_list,
        args.image_dir,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        image_ext=args.image_ext,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
