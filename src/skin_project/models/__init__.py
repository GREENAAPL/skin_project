"""EfficientNet-B0 model factory."""
from __future__ import annotations

from torch import nn

from ..config import Config
from .pretrained import EfficientNetB0

__all__ = ["EfficientNetB0", "build_model", "count_parameters"]


def build_model(cfg: Config) -> nn.Module:
    """Build the image-only EfficientNet-B0 classifier from a config."""
    model_cfg = dict(cfg.model)
    model_type = str(model_cfg.get("type", "efficientnet_b0")).lower()
    backbone = str(model_cfg.get("backbone", "efficientnet_b0")).lower()
    if model_type not in {"efficientnet_b0", "pretrained"} or backbone != "efficientnet_b0":
        raise ValueError("this project supports only the EfficientNet-B0 image model")

    return EfficientNetB0(
        num_classes=int(model_cfg.get("num_classes", 7)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
        unfreeze_last_blocks=int(model_cfg.get("unfreeze_last_blocks", 0)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = (p for p in model.parameters() if p.requires_grad) if trainable_only else model.parameters()
    return sum(parameter.numel() for parameter in parameters)
