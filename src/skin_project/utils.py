"""Shared constants and reproducibility helpers."""
from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SAMPLE_ID_COL = "sample_id"
LESION_ID_COL = "lesion_id"
LABEL_COL = "label"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit_hash(default: str = "unknown") -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode().strip()
    except Exception:
        return default


@dataclass
class RunProvenance:
    run_name: str
    seed: int
    model_type: str
    data_root: str
    checkpoint_path: str
    tensorboard_dir: str
    git_commit: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(prefer_cuda: bool = True) -> str:
    """Select CUDA when available, otherwise use CPU."""
    override = os.environ.get("SKIN_PROJECT_DEVICE", "").strip().lower()
    if override:
        return override
    if prefer_cuda:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"
