import pytest
import torch

from skin_project.config import Config
from skin_project.models import EfficientNetB0, build_model, count_parameters


def test_efficientnet_outputs_seven_logits():
    model = EfficientNetB0(num_classes=7, pretrained=False, freeze_encoder=True).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7)
    assert count_parameters(model) > 0


def test_partial_fine_tuning_adds_trainable_parameters():
    frozen = EfficientNetB0(pretrained=False, freeze_encoder=True)
    tuned = EfficientNetB0(
        pretrained=False,
        freeze_encoder=True,
        unfreeze_last_blocks=2,
    )
    assert tuned.trainable_parameter_count() > frozen.trainable_parameter_count()


def test_factory_rejects_non_efficientnet_architecture():
    with pytest.raises(ValueError, match="EfficientNet-B0"):
        build_model(Config({"model": {"type": "resnet", "backbone": "resnet18"}}))
