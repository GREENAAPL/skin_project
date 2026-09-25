"""Config-driven EfficientNet-B0 training and checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint
from .config import Config, add_config_arg, save_effective_config
from .dataloaders import build_dataloaders_from_config
from .evaluation import evaluate_loader
from .metrics import compute_metrics
from .models import build_model, count_parameters
from .preprocessing import class_weights_from_labels
from .tensorboard_logging import TBLogger
from .utils import (
    LABEL_COL,
    RunProvenance,
    ensure_dir,
    get_device,
    git_commit_hash,
    set_seed,
)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build the optimizer, including optional encoder/head parameter groups."""
    learning_rate = float(cfg.training.get("learning_rate", 1e-3))
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    optimizer_name = str(cfg.training.get("optimizer", "adam")).lower()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    separate_learning_rates = any(
        key in cfg.training
        for key in ("backbone_learning_rate", "classifier_learning_rate")
    )
    parameters: list[nn.Parameter] | list[dict]
    if separate_learning_rates:
        encoder = getattr(model, "encoder", None)
        classifier = getattr(model, "classifier", None)
        if encoder is None or classifier is None:
            raise ValueError(
                "separate backbone/classifier learning rates require a model with "
                "encoder and classifier modules"
            )
        backbone_learning_rate = float(
            cfg.training.get("backbone_learning_rate", learning_rate)
        )
        classifier_learning_rate = float(
            cfg.training.get("classifier_learning_rate", learning_rate)
        )
        if backbone_learning_rate <= 0 or classifier_learning_rate <= 0:
            raise ValueError("learning rates must be greater than zero")

        backbone_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        classifier_parameters = [
            parameter
            for parameter in classifier.parameters()
            if parameter.requires_grad
        ]
        grouped_ids = {
            id(parameter)
            for parameter in backbone_parameters + classifier_parameters
        }
        remaining_parameters = [
            parameter
            for parameter in trainable_parameters
            if id(parameter) not in grouped_ids
        ]
        parameters = []
        if backbone_parameters:
            parameters.append(
                {
                    "params": backbone_parameters,
                    "lr": backbone_learning_rate,
                    "name": "backbone",
                }
            )
        if classifier_parameters:
            parameters.append(
                {
                    "params": classifier_parameters,
                    "lr": classifier_learning_rate,
                    "name": "classifier",
                }
            )
        if remaining_parameters:
            parameters.append(
                {
                    "params": remaining_parameters,
                    "lr": learning_rate,
                    "name": "other",
                }
            )
    else:
        parameters = trainable_parameters

    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "sgd":
        momentum = float(cfg.training.get("momentum", 0.9))
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, cfg: Config, epochs: int):
    sched = str(cfg.training.get("scheduler", "none")).lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.training.get("step_size", 10)),
            gamma=float(cfg.training.get("gamma", 0.1)))
    return None


def _grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2) ** 2)
    return float(total ** 0.5)


def train_one_epoch(model, loader, optimizer, criterion, device, log_grad_norm=False):
    """Run one training epoch; return (mean_loss, macro_f1, mean_grad_norm)."""
    model.train()
    losses, y_true, y_pred, grad_norms = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        if log_grad_norm:
            grad_norms.append(_grad_global_norm(model))
        optimizer.step()
        losses.append(float(loss.detach()) * len(label))
        y_true.extend(label.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(1).cpu().tolist())
    n = len(y_true)
    mean_loss = float(np.sum(losses) / n) if n else float("nan")
    macro_f1 = compute_metrics(y_true, y_pred)["macro_f1"] if n else float("nan")
    mean_gn = float(np.mean(grad_norms)) if grad_norms else None
    return mean_loss, macro_f1, mean_gn


def train_from_config(cfg: Config, run_name: str | None = None) -> dict:
    run_name = run_name or cfg.run_name
    seed = cfg.seed
    set_seed(seed)
    device = get_device()

    bundle = build_dataloaders_from_config(cfg)
    model = build_model(cfg).to(device)

    # class-weighted loss from *training* labels only
    weights = None
    num_classes = int(cfg.model.get("num_classes", 7))
    class_names = list(cfg.data.get("class_names", []))
    if len(class_names) != num_classes:
        raise ValueError("data.class_names must contain one name per output class")
    if cfg.training.get("use_class_weights", True):
        w = class_weights_from_labels(
            bundle.train_df[LABEL_COL].tolist(), num_classes=num_classes
        )
        weights = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg.training.get("label_smoothing", 0.0)),
    )

    epochs = int(cfg.training.get("epochs", 10))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    tb_root = cfg.logging.get("tensorboard_root", "runs/tensorboard")
    ckpt_root = ensure_dir(cfg.logging.get("checkpoint_root", "outputs/checkpoints"))
    logger = TBLogger(tb_root, run_name)

    mtype = str(cfg.model.get("type", "efficientnet_b0"))
    log_grad = bool(cfg.training.get("log_gradient_norm", False))

    # persist effective config, seed, provenance next to the checkpoint
    run_dir = ensure_dir(Path(ckpt_root) / run_name)
    save_effective_config(cfg.raw, run_dir / "effective_config.yaml")
    (run_dir / "seed.txt").write_text(str(seed))
    ckpt_path = Path(ckpt_root) / f"{run_name}.pt"
    RunProvenance(
        run_name=run_name, seed=seed, model_type=mtype,
        data_root=str(cfg.data.get("root")), checkpoint_path=str(ckpt_path),
        tensorboard_dir=str(logger.log_dir), git_commit=git_commit_hash(),
    ).save(run_dir / "provenance.json")

    learning_rate_summary = ", ".join(
        f"{group.get('name', f'group_{index}')}={group['lr']:.6g}"
        for index, group in enumerate(optimizer.param_groups)
    )
    print(f"model={mtype}  trainable_params={count_parameters(model):,}  "
          f"device={device}  "
          f"learning_rates=[{learning_rate_summary}]")

    best_f1, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, gn = train_one_epoch(
            model, bundle.train_loader, optimizer, criterion, device, log_grad_norm=log_grad)
        group_learning_rates = {
            str(group.get("name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        lr = max(group_learning_rates.values())
        logger.log_train(epoch, tr_loss, tr_f1, lr, gradient_norm=gn)

        dev_metrics, _ = evaluate_loader(model, bundle.dev_loader, device)
        logger.log_dev(epoch, dev_metrics, loss=dev_metrics.get("loss"))
        if scheduler:
            scheduler.step()

        dev_f1 = dev_metrics.get("macro_f1", float("nan"))
        print(f"epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.4f} train_f1={tr_f1:.4f}  "
              f"dev_f1={dev_f1:.4f} dev_bal_acc={dev_metrics.get('balanced_accuracy', float('nan')):.4f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_macro_f1": tr_f1,
                        **{f"learning_rate_{name}": value
                           for name, value in group_learning_rates.items()},
                        **{f"dev_{k}": v for k, v in dev_metrics.items()}})

        if dev_f1 == dev_f1 and dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            save_checkpoint(
                ckpt_path, model, model_config=dict(cfg.model),
                image_size=int(cfg.data.get("image_size", 224)), seed=seed,
                class_names=class_names,
                extra={"epoch": epoch, "dev_macro_f1": dev_f1, "run_name": run_name},
            )

    logger.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {"run_name": run_name, "best_epoch": best_epoch, "best_dev_macro_f1": best_f1,
               "checkpoint": str(ckpt_path)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best dev macro_f1={best_f1:.4f} @ epoch {best_epoch} -> {ckpt_path}")
    return summary


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    add_config_arg(parser)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    cfg = Config.from_file(args.config)
    train_from_config(cfg, run_name=args.run_name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
