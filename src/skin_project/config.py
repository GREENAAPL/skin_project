"""Configuration loading and merging (README §11).

Configs are YAML. A config may declare `defaults: [base.yaml, ...]` (paths relative
to the config's own directory). Parent values are deep-merged, so each experiment
only needs to override the hyperparameters that change.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a `defaults:` inheritance list first."""
    path = Path(path)
    # encoding= is not optional: without it Python uses the locale encoding, which
    # is cp949 on Korean Windows, and every config carrying a non-ASCII character
    # (most of ours do -- em dashes and section marks in the comments) dies with
    # UnicodeDecodeError before training can start.
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    defaults = raw.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for parent in defaults:
        parent_path = (path.parent / parent).resolve()
        merged = _deep_merge(merged, load_config(parent_path))
    merged = _deep_merge(merged, raw)
    return merged


def save_effective_config(config: dict[str, Any], path: str | Path) -> None:
    """Write the fully-resolved config next to a run (README §11)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


@dataclass
class Config:
    """Typed convenience wrapper with dotted access and sensible defaults."""

    raw: dict[str, Any] = field(default_factory=dict)

    # -- section accessors -------------------------------------------------- #
    @property
    def experiment(self) -> dict:
        return self.raw.get("experiment", {})

    @property
    def data(self) -> dict:
        return self.raw.get("data", {})

    @property
    def model(self) -> dict:
        return self.raw.get("model", {})

    @property
    def training(self) -> dict:
        return self.raw.get("training", {})

    @property
    def logging(self) -> dict:
        return self.raw.get("logging", {})

    # -- common fields ------------------------------------------------------ #
    @property
    def seed(self) -> int:
        return int(self.experiment.get("seed", 42))

    @property
    def run_name(self) -> str:
        return str(self.experiment.get("run_name", "run"))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        return cls(load_config(path))


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Standard `--config` argument shared by every CLI module."""
    parser.add_argument(
        "--config", required=True, type=str, help="Path to a YAML configuration file."
    )


def _main(argv: list[str] | None = None) -> int:
    """`python -m skin_project.config --config <f>` prints the effective config."""
    parser = argparse.ArgumentParser(description="Inspect an effective configuration.")
    add_config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
