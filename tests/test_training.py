import torch

from skin_project.config import Config
from skin_project.training import build_optimizer


class SplitLearningRateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 3)
        self.classifier = torch.nn.Linear(3, 7)


def test_optimizer_uses_separate_encoder_and_classifier_rates():
    config = Config(
        {
            "training": {
                "optimizer": "adamw",
                "learning_rate": 2e-4,
                "backbone_learning_rate": 5e-5,
                "classifier_learning_rate": 5e-4,
            }
        }
    )
    optimizer = build_optimizer(SplitLearningRateModel(), config)
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"backbone": 5e-5, "classifier": 5e-4}
