"""Deterministic brightness-robustness evaluation for skin-lesion models.

The same labeled split is evaluated repeatedly with fixed brightness factors. The
clean condition (factor 1.0) is the reference, so all reported drops are paired and
cannot be explained by a different sample composition.

CLI example::

    python -m skin_project.robustness \
        --config configs/isic2018_7class_efficientnet_b0_320.yaml \
        --checkpoint outputs/checkpoints/isic2018_7class_efficientnet_b0_320.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.metrics import recall_score

from .checkpoint import load_checkpoint, rebuild_from_checkpoint
from .config import Config, add_config_arg
from .dataloaders import build_dataloader
from .dataset import SkinLesionDataset
from .evaluation import evaluate_loader, predictions_dataframe
from .utils import get_device

DEFAULT_BRIGHTNESS_FACTORS = (0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50)
PLOTTED_METRICS = ("macro_f1", "balanced_accuracy", "sensitivity", "specificity")


def parse_brightness_factors(value: str | Iterable[float]) -> list[float]:
    """Parse, validate, de-duplicate, and sort brightness factors."""
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        factors = [float(part) for part in raw_values]
    else:
        factors = [float(part) for part in value]

    if not factors:
        raise ValueError("at least one brightness factor is required")
    if any(not math.isfinite(factor) or factor <= 0 for factor in factors):
        raise ValueError("brightness factors must be finite and greater than zero")

    unique = sorted(set(factors))
    if not any(math.isclose(factor, 1.0, abs_tol=1e-9) for factor in unique):
        raise ValueError("brightness factors must include the clean reference 1.0")
    return unique


def add_drop_columns(rows: Sequence[dict]) -> pd.DataFrame:
    """Add absolute and relative drops from the factor-1.0 reference."""
    frame = pd.DataFrame(rows).sort_values("brightness_factor").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no robustness results were supplied")

    baseline_mask = frame["brightness_factor"].map(
        lambda factor: math.isclose(float(factor), 1.0, abs_tol=1e-9)
    )
    if baseline_mask.sum() != 1:
        raise ValueError("results must contain exactly one brightness factor 1.0")

    baseline = frame.loc[baseline_mask].iloc[0]
    for metric in PLOTTED_METRICS + ("auroc", "accuracy"):
        if metric not in frame.columns:
            continue
        baseline_value = float(baseline[metric])
        frame[f"{metric}_drop"] = baseline_value - frame[metric].astype(float)
        relative_name = f"{metric}_relative_drop_pct"
        if math.isfinite(baseline_value) and baseline_value != 0:
            frame[relative_name] = 100.0 * frame[f"{metric}_drop"] / baseline_value
        else:
            frame[relative_name] = float("nan")
    return frame


def build_summary(frame: pd.DataFrame) -> dict:
    """Return compact clean, average-corruption, and worst-case statistics."""
    baseline = frame.loc[
        frame["brightness_factor"].map(
            lambda factor: math.isclose(float(factor), 1.0, abs_tol=1e-9)
        )
    ].iloc[0]
    corrupted = frame.loc[
        ~frame["brightness_factor"].map(
            lambda factor: math.isclose(float(factor), 1.0, abs_tol=1e-9)
        )
    ]
    worst = frame.loc[frame["macro_f1"].astype(float).idxmin()]
    return {
        "clean_macro_f1": float(baseline["macro_f1"]),
        "mean_corrupted_macro_f1": (
            float(corrupted["macro_f1"].mean()) if len(corrupted) else None
        ),
        "worst_brightness_factor": float(worst["brightness_factor"]),
        "worst_macro_f1": float(worst["macro_f1"]),
        "maximum_macro_f1_drop": float(worst["macro_f1_drop"]),
        "maximum_macro_f1_relative_drop_pct": float(
            worst["macro_f1_relative_drop_pct"]
        ),
    }


def plot_brightness_curve(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a compact metric-versus-brightness plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in PLOTTED_METRICS:
        if metric in frame.columns and frame[metric].notna().any():
            ax.plot(
                frame["brightness_factor"],
                frame[metric],
                marker="o",
                linewidth=1.8,
                label=metric.replace("_", " "),
            )
    ax.axvline(1.0, color="0.45", linestyle="--", linewidth=1, label="clean reference")
    ax.set(
        xlabel="Brightness factor",
        ylabel="Score",
        title="Skin-lesion classifier robustness to brightness shift",
        ylim=(0.0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def evaluate_brightness(
    cfg: Config,
    checkpoint_path: str | Path,
    factors: Iterable[float] = DEFAULT_BRIGHTNESS_FACTORS,
    split: str = "dev",
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate one checkpoint across fixed brightness factors and save artifacts."""
    factors = parse_brightness_factors(factors)
    if split not in {"train", "dev"}:
        raise ValueError("split must be 'train' or 'dev'")

    device = get_device()
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model, image_size = rebuild_from_checkpoint(payload, device=device)

    data = cfg.data
    root = Path(data["root"])
    metadata_path = root / data.get(f"{split}_metadata", f"{split}_metadata.csv")
    image_dir = root / data.get(f"{split}_images", f"{split}_images")
    image_ext = str(data.get("image_ext", ".png"))
    frame = pd.read_csv(metadata_path)
    batch_size = int(cfg.training.get("batch_size", 32))
    num_workers = int(data.get("num_workers", 0))

    if output_dir is None:
        output_dir = Path("outputs") / "robustness" / cfg.run_name / split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    num_classes = int(cfg.model.get("num_classes", 2))
    configured_names = list(data.get("class_names", []))
    class_names = (
        configured_names
        if len(configured_names) == num_classes
        else [f"class_{index}" for index in range(num_classes)]
    )
    for factor in factors:
        dataset = SkinLesionDataset(
            frame,
            image_dir,
            image_size=image_size,
            train=False,
            has_labels=True,
            image_ext=image_ext,
            brightness_factor=factor,
        )
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics, predictions = evaluate_loader(model, loader, device)
        recalls = recall_score(
            predictions["label"],
            predictions["pred"],
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        )
        metrics.update(
            {
                f"recall_{class_name}": float(recalls[index])
                for index, class_name in enumerate(class_names)
            }
        )
        metric_rows.append({"brightness_factor": factor, **metrics})
        prediction_frame = predictions_dataframe(predictions, class_names)
        prediction_frame.insert(0, "brightness_factor", factor)
        prediction_frames.append(prediction_frame)
        if num_classes == 2:
            detail = (
                f"sensitivity={metrics['sensitivity']:.4f}  "
                f"specificity={metrics['specificity']:.4f}"
            )
        else:
            detail = f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        print(
            f"brightness={factor:.2f}  macro_f1={metrics['macro_f1']:.4f}  {detail}"
        )

    metrics_frame = add_drop_columns(metric_rows)
    summary = build_summary(metrics_frame)
    metrics_path = output_dir / "brightness_metrics.csv"
    predictions_path = output_dir / "brightness_predictions.csv"
    summary_path = output_dir / "brightness_summary.json"
    plot_path = output_dir / "brightness_curve.png"

    metrics_frame.to_csv(metrics_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_brightness_curve(metrics_frame, plot_path)

    return {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "plot": str(plot_path),
        **summary,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a skin-lesion checkpoint under fixed brightness shifts."
    )
    add_config_arg(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument(
        "--brightness-factors",
        default=",".join(str(value) for value in DEFAULT_BRIGHTNESS_FACTORS),
        help="Comma-separated positive factors; must include 1.0.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_file(args.config)
    result = evaluate_brightness(
        cfg,
        args.checkpoint,
        factors=parse_brightness_factors(args.brightness_factors),
        split=args.split,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
