"""Stream Amazon Reviews 2023 category files from the Hugging Face Hub."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from itertools import islice
from typing import Any

import pandas as pd
from huggingface_hub import HfFileSystem

from reviewlens.config import load_config

logger = logging.getLogger(__name__)


def stream_reviews(category: str | None = None, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw review dicts from the head of a category .jsonl, without a full download."""
    cfg = load_config("data")["dataset"]
    category = category or cfg["category"]
    path = f"datasets/{cfg['hf_repo']}/{cfg['file_template'].format(category=category)}"
    with HfFileSystem().open(path, "r", encoding="utf-8") as f:
        for line in islice(f, max_rows):
            yield json.loads(line)


def load_reviews(category: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Load (the head of) a category into a DataFrame with a datetime column."""
    df = pd.DataFrame(stream_reviews(category, max_rows))
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    logger.info("loaded %d rows from %s", len(df), category)
    return df


def describe(df: pd.DataFrame) -> dict[str, Any]:
    """Columns, row count, date range and star/class balance."""
    labels = load_config("data")["labels"]
    star_to_label = {s: name for name, stars in labels.items() for s in stars}
    classes = df["rating"].round().astype(int).map(star_to_label)
    return {
        "columns": list(df.columns),
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "stars": df["rating"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "classes": classes.value_counts(normalize=True).round(4).to_dict(),
        "empty_text_frac": round(float((df["text"].str.strip() == "").mean()), 4),
        "median_chars": int(df["text"].str.len().median()),
    }
