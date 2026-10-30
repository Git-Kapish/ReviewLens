"""Classification metrics and the metrics-JSON writer shared by every model."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from reviewlens.config import resolve_path
from reviewlens.data.clean import LABELS

logger = logging.getLogger(__name__)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over the three classes (the primary metric)."""
    ids = range(len(LABELS))
    return float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1/support and the confusion matrix.

    Inputs are label ids (indices into `LABELS`). Matrix rows are true, columns predicted.
    """
    ids = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                   "support": int(s[i])}
            for i, name in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "labels": LABELS,
    }


def write_metrics(path: str | Path, metrics: dict[str, Any], config: dict[str, Any]) -> Path:
    """Write `metrics` plus the config used and a UTC timestamp to a JSON file."""
    path = resolve_path(str(path)) if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "config": config}
    path.write_text(json.dumps(payload | metrics, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path
