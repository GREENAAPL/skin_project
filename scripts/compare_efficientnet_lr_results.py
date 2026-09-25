"""Compare the five EfficientNet differential-learning-rate experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "efficientnet_lr"
EXPERIMENTS = (
    {
        "id": "LR-0",
        "label": "LR-0 equal",
        "run": "isic2018_7class_efficientnet_b0_320",
        "backbone_lr": 2e-4,
        "classifier_lr": 2e-4,
    },
    {
        "id": "LR-1",
        "label": "LR-1 low body",
        "run": "isic2018_effnet_lr1_low_body",
        "backbone_lr": 5e-5,
        "classifier_lr": 2e-4,
    },
    {
        "id": "LR-2",
        "label": "LR-2 recommended",
        "run": "isic2018_effnet_lr2_recommended",
        "backbone_lr": 5e-5,
        "classifier_lr": 5e-4,
    },
    {
        "id": "LR-3",
        "label": "LR-3 very low body",
        "run": "isic2018_effnet_lr3_very_low_body",
        "backbone_lr": 1e-5,
        "classifier_lr": 5e-4,
    },
    {
        "id": "LR-4",
        "label": "LR-4 higher body",
        "run": "isic2018_effnet_lr4_higher_body",
        "backbone_lr": 1e-4,
        "classifier_lr": 5e-4,
    },
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    for experiment in EXPERIMENTS:
        run = experiment["run"]
        checkpoint_dir = ROOT / "outputs" / "checkpoints" / run
        robustness_dir = ROOT / "outputs" / "robustness" / run / "dev"
        training_summary = _read_json(checkpoint_dir / "summary.json")
        history = pd.DataFrame(_read_json(checkpoint_dir / "history.json"))
        brightness_summary = _read_json(
            robustness_dir / "brightness_summary.json"
        )
        brightness_metrics = pd.read_csv(
            robustness_dir / "brightness_metrics.csv"
        )
        curves[str(experiment["id"])] = brightness_metrics

        best_epoch = int(training_summary["best_epoch"])
        best_history = history.loc[history["epoch"] == best_epoch].iloc[0]
        clean = brightness_metrics.loc[
            brightness_metrics["brightness_factor"].astype(float).eq(1.0)
        ].iloc[0]
        rows.append(
            {
                **experiment,
                "best_epoch": best_epoch,
                "train_macro_f1_at_best": float(best_history["train_macro_f1"]),
                "dev_macro_f1": float(clean["macro_f1"]),
                "train_dev_macro_f1_gap": float(
                    best_history["train_macro_f1"] - clean["macro_f1"]
                ),
                "dev_balanced_accuracy": float(clean["balanced_accuracy"]),
                "mean_corrupted_macro_f1": float(
                    brightness_summary["mean_corrupted_macro_f1"]
                ),
                "worst_brightness_factor": float(
                    brightness_summary["worst_brightness_factor"]
                ),
                "worst_macro_f1": float(brightness_summary["worst_macro_f1"]),
                "maximum_relative_drop_pct": float(
                    brightness_summary["maximum_macro_f1_relative_drop_pct"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    return frame, curves


def save_plot(frame: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for experiment_id, curve in curves.items():
        axes[0].plot(
            curve["brightness_factor"],
            curve["macro_f1"],
            marker="o",
            linewidth=2,
            label=experiment_id,
        )
    axes[0].axvline(1.0, color="0.45", linestyle="--", linewidth=1)
    curve_minimum = min(float(curve["macro_f1"].min()) for curve in curves.values())
    axes[0].set(
        title="Macro-F1 across brightness factors",
        xlabel="Brightness factor",
        ylabel="Macro-F1",
        ylim=(max(0.0, curve_minimum - 0.04), 0.82),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    positions = range(len(frame))
    width = 0.36
    clean_bars = axes[1].bar(
        [position - width / 2 for position in positions],
        frame["dev_macro_f1"],
        width,
        label="Clean (1.0)",
    )
    corrupted_bars = axes[1].bar(
        [position + width / 2 for position in positions],
        frame["mean_corrupted_macro_f1"],
        width,
        label="Mean shifted brightness",
    )
    axes[1].set_xticks(list(positions), frame["id"])
    bar_minimum = float(
        frame[["dev_macro_f1", "mean_corrupted_macro_f1"]].min().min()
    )
    axes[1].set(
        title="Clean score and brightness-shift average",
        xlabel="Experiment",
        ylabel="Macro-F1",
        ylim=(max(0.0, bar_minimum - 0.04), 0.82),
    )
    axes[1].bar_label(clean_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].bar_label(corrupted_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("EfficientNet-B0 differential learning-rate comparison")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "efficientnet_lr_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, curves = build_comparison()
    comparison_path = OUTPUT_DIR / "efficientnet_lr_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    best = comparison.sort_values(
        ["dev_macro_f1", "mean_corrupted_macro_f1"], ascending=False
    ).iloc[0]
    summary = {
        "selection_metric": "dev_macro_f1",
        "tie_breaker": "mean_corrupted_macro_f1",
        "best_experiment": str(best["id"]),
        "best_run": str(best["run"]),
        "best_backbone_lr": float(best["backbone_lr"]),
        "best_classifier_lr": float(best["classifier_lr"]),
        "best_dev_macro_f1": float(best["dev_macro_f1"]),
        "best_mean_corrupted_macro_f1": float(
            best["mean_corrupted_macro_f1"]
        ),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_path = save_plot(comparison, curves)
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"csv={comparison_path}")
    print(f"plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
