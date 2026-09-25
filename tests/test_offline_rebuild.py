import torch
from torch.nn.modules.batchnorm import _BatchNorm

from skin_project.checkpoint import load_checkpoint, rebuild_from_checkpoint, save_checkpoint
from skin_project.models import EfficientNetB0


CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def test_checkpoint_rebuilds_without_pretrained_download(tmp_path, monkeypatch):
    model = EfficientNetB0(pretrained=False, freeze_encoder=True)
    config = {
        "type": "efficientnet_b0",
        "backbone": "efficientnet_b0",
        "num_classes": 7,
        "pretrained": True,
        "freeze_encoder": True,
    }
    path = save_checkpoint(tmp_path / "model.pt", model, config, 64, 42, CLASSES)

    from torchvision import models

    original = models.efficientnet_b0

    def guarded(weights=None):
        assert weights is None
        return original(weights=None)

    monkeypatch.setattr(models, "efficientnet_b0", guarded)
    rebuilt, image_size = rebuild_from_checkpoint(load_checkpoint(path))
    assert image_size == 64
    assert all(
        torch.equal(value, rebuilt.state_dict()[key])
        for key, value in model.state_dict().items()
    )


def test_frozen_batchnorm_statistics_do_not_change():
    model = EfficientNetB0(pretrained=False, freeze_encoder=True)
    model.train()
    batch_norms = [module for module in model.encoder.modules() if isinstance(module, _BatchNorm)]
    assert batch_norms and all(not module.training for module in batch_norms)
    before = batch_norms[0].running_mean.clone()
    model(torch.randn(2, 3, 64, 64))
    assert torch.equal(before, batch_norms[0].running_mean)
