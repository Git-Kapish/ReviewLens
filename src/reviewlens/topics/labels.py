"""Hand-written topic labels, stored in `reports/topic_labels.json`."""

from __future__ import annotations

import json
from pathlib import Path


def load_labels(path: str | Path, topic_ids: list[int]) -> dict[int, str]:
    """Read `{topic_id: {"label": ...}}` and check it covers exactly the real topics.

    Raises ValueError if a topic has no label or a label refers to a topic that does not exist,
    so stale labels cannot silently attach to the wrong topic after a refit.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = {int(k): v["label"] for k, v in raw["topics"].items()}
    real = {t for t in topic_ids if t != -1}
    if set(labels) != real:
        raise ValueError(
            f"labels cover {sorted(labels)} but the model has topics {sorted(real)}"
        )
    return labels
