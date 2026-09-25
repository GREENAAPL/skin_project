"""Image-only EfficientNet-B0 classifier with partial fine-tuning support."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class EfficientNetB0(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        unfreeze_last_blocks: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        network = models.efficientnet_b0(weights=weights)

        self.embedding_dim = network.classifier[1].in_features
        self.encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
        self._stage_modules = list(network.features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.set_trainable(freeze_encoder, unfreeze_last_blocks)

    def set_trainable(self, freeze_encoder: bool, unfreeze_last_blocks: int = 0) -> None:
        """Freeze the encoder, then optionally fine-tune its final N stages."""
        if unfreeze_last_blocks < 0:
            raise ValueError("unfreeze_last_blocks must be non-negative")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze_encoder
        if freeze_encoder and unfreeze_last_blocks:
            for stage in self._stage_modules[-unfreeze_last_blocks:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "EfficientNetB0":
        """Keep BatchNorm statistics fixed inside frozen encoder stages."""
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                    parameter.requires_grad for parameter in module.parameters(recurse=False)
                ):
                    module.eval()
        return self

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(image)))

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def frozen_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)


# Backward-compatible name for checkpoints created before the repository cleanup.
PretrainedImageModel = EfficientNetB0
