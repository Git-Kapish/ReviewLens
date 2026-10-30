"""Config loading and seeding helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_SEED = 42


def load_config(name: str) -> dict[str, Any]:
    """Load `configs/<name>.yaml` (name given with or without extension)."""
    path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / rel


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python, numpy and torch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
