"""Validation metrics and prediction export for the seven-class model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, rebuild_from_checkpoint
from .config import Config, add_config_arg
from .dataloaders import build_dataloader
from .dataset import SkinLesionDataset
from .metrics import compute_metrics, format_metrics
from .utils import LABEL_COL, SAMPLE_ID_COL, get_device


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    """Run the model once and retain identifiers, labels and all class scores."""
    model.eval()
    sample_ids: list[str] = []
    lesion_ids: list[str] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    predictions: list[int] = []
    loss_sum = 0.0
    labeled_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        batch_labels = batch["label"].to(device)
        logits = model(images)
        batch_probabilities = F.softmax(logits, dim=1)

        sample_ids.extend(batch["sample_id"])
        lesion_ids.extend(batch["lesion_id"])
        labels.extend(batch_labels.cpu().tolist())
        probabilities.extend(batch_probabilities.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

        if torch.all(batch_labels >= 0):
            batch_size = batch_labels.numel()
            loss_sum += F.cross_entropy(logits, batch_labels).item() * batch_size
            labeled_count += batch_size

    return {
        "sample_id": sample_ids,
        "lesion_id": lesion_ids,
        "label": labels,
        "probabilities": probabilities,
        "pred": predictions,
        "loss": loss_sum / labeled_count if labeled_count else float("nan"),
    }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[dict, dict]:
    out = predict_loader(model, loader, device)
    labeled = [index for index, label in enumerate(out["label"]) if label >= 0]
    if labeled:
        metrics = compute_metrics(
            [out["label"][index] for index in labeled],
            [out["pred"][index] for index in labeled],
            [out["probabilities"][index] for index in labeled],
        )
    else:
        metrics = {}
    metrics["loss"] = out["loss"]
    return metrics, out


def predictions_dataframe(out: dict, class_names: list[str] | None = None) -> pd.DataFrame:
    probabilities = torch.as_tensor(out["probabilities"]).numpy()
    class_names = class_names or [f"class_{index}" for index in range(probabilities.shape[1])]
    if len(class_names) != probabilities.shape[1]:
        raise ValueError("class_names length does not match prediction width")
    frame = pd.DataFrame(
        {
            SAMPLE_ID_COL: out["sample_id"],
            "lesion_id": out["lesion_id"],
            LABEL_COL: out["label"],
            "predicted_class": out["pred"],
            "predicted_label": [class_names[index] for index in out["pred"]],
        }
    )
    for index, name in enumerate(class_names):
        frame[f"probability_{name}"] = probabilities[:, index]
    return frame


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on an ISIC split.")
    add_config_arg(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--predictions-out", default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_file(args.config)
    device = get_device()
    payload = load_checkpoint(args.checkpoint, map_location=device)
    model, image_size = rebuild_from_checkpoint(payload, device=device)
    data = cfg.data
    root = Path(data["root"])
    frame = pd.read_csv(root / data.get(f"{args.split}_metadata", f"prepared/{args.split}_metadata.csv"))
    dataset = SkinLesionDataset(
        frame,
        root / data.get(f"{args.split}_images", "raw/ISIC2018_Task3_Training_Input"),
        image_size=image_size,
        image_ext=str(data.get("image_ext", ".jpg")),
    )
    loader = build_dataloader(
        dataset,
        int(cfg.training.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
    )
    metrics, predictions = evaluate_loader(model, loader, device)
    print(f"[{args.split}] {format_metrics(metrics)}")

    metrics_path = Path(args.metrics_out or f"outputs/predictions/{cfg.run_name}_{args.split}_metrics.json")
    predictions_path = Path(
        args.predictions_out or f"outputs/predictions/{cfg.run_name}_{args.split}_predictions.csv"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_dataframe(predictions, list(data["class_names"])).to_csv(predictions_path, index=False)
    print(f"saved metrics -> {metrics_path}\nsaved predictions -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
