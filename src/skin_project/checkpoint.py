"""Self-contained EfficientNet checkpoint save and load helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .transforms import IMAGENET_MEAN, IMAGENET_STD


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict,
    image_size: int,
    seed: int,
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save everything needed to rebuild the image classifier offline."""
    num_classes = int(model_config.get("num_classes", len(class_names)))
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match model num_classes")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "class_names": list(class_names),
        "class_to_index": {name: index for index, name in enumerate(class_names)},
        "image_config": {
            "image_size": int(image_size),
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "seed": int(seed),
    }
    if extra is not None:
        payload["extra"] = dict(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def rebuild_from_checkpoint(payload: dict, device: str = "cpu"):
    """Return ``(model, image_size)`` without downloading ImageNet weights."""
    from .config import Config
    from .models import build_model

    model_config = dict(payload["model_config"])
    model_config["pretrained"] = False
    cfg = Config({"model": model_config})
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, int(payload["image_config"]["image_size"])
