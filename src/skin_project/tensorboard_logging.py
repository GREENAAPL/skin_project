"""TensorBoard logging (README §16).

Thin wrapper over ``torch.utils.tensorboard.SummaryWriter`` that enforces the exact
scalar tag names the project requires, so every team's runs are directly
comparable in one TensorBoard instance.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# Canonical scalar tags (README §16.1).
TRAIN_TAGS = ("train/loss", "train/macro_f1", "train/learning_rate")
DEV_TAGS = (
    "dev/loss",
    "dev/macro_f1",
    "dev/balanced_accuracy",
    "dev/sensitivity",
    "dev/specificity",
    "dev/auroc",
)
# Gradient norm is logged when enabled in the training config.


class TBLogger:
    def __init__(self, tensorboard_root: str | Path, run_name: str) -> None:
        self.log_dir = Path(tensorboard_root) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_train(self, step: int, loss: float, macro_f1: float, lr: float,
                  gradient_norm: float | None = None) -> None:
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/macro_f1", macro_f1, step)
        self.writer.add_scalar("train/learning_rate", lr, step)
        if gradient_norm is not None:
            self.writer.add_scalar("train/gradient_norm", gradient_norm, step)

    def log_dev(self, step: int, metrics: dict[str, float], loss: float | None = None) -> None:
        if loss is not None:
            self.writer.add_scalar("dev/loss", loss, step)
        mapping = {
            "dev/macro_f1": "macro_f1",
            "dev/balanced_accuracy": "balanced_accuracy",
            "dev/sensitivity": "sensitivity",
            "dev/specificity": "specificity",
            "dev/auroc": "auroc",
        }
        for tag, key in mapping.items():
            if key in metrics and metrics[key] == metrics[key]:  # skip NaN
                self.writer.add_scalar(tag, metrics[key], step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
