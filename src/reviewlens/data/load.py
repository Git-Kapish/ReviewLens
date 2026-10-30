"""Stream Amazon Reviews 2023 category files from the Hugging Face Hub."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfFileSystem

from reviewlens.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _hub_path(category: str | None) -> str:
    cfg = load_config("data")["dataset"]
    category = category or cfg["category"]
    return f"datasets/{cfg['hf_repo']}/{cfg['file_template'].format(category=category)}"


def stream_lines(category: str | None = None, max_rows: int | None = None) -> Iterator[str]:
    """Yield raw .jsonl lines from the head of a category file, without a full download."""
    with HfFileSystem().open(_hub_path(category), "r", encoding="utf-8") as f:
        yield from islice(f, max_rows)


def stream_reviews(
    category: str | None = None, max_rows: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield parsed review dicts from the head of a category file."""
    for line in stream_lines(category, max_rows):
        yield json.loads(line)


def reservoir_sample_lines(k: int, seed: int, category: str | None = None) -> list[str]:
    """Uniform random sample of k lines over the whole file (algorithm R, one pass)."""
    rng = random.Random(seed)
    sample: list[str] = []
    n = 0
    for n, line in enumerate(stream_lines(category), start=1):
        if n <= k:
            sample.append(line)
        elif (j := rng.randint(0, n - 1)) < k:
            sample[j] = line
        if n % 500_000 == 0:
            logger.info("scanned %d lines", n)
    logger.info("scanned %d lines in total", n)
    return sample


def load_raw_sample(category: str | None = None) -> pd.DataFrame:
    """Raw sample cached at `data/raw/<category>_sample_<n>.jsonl`; streamed on first use."""
    cfg = load_config("data")
    category = category or cfg["dataset"]["category"]
    n = cfg["sample"]["raw_rows"]
    path = resolve_path(cfg["paths"]["raw"]) / f"{category}_sample_{n}.jsonl"
    if not path.exists():
        logger.info("no cached sample, streaming %s (one full pass)", category)
        _write_lines(path, reservoir_sample_lines(n, cfg["seed"], category))
    return pd.read_json(path, lines=True, dtype=False)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)


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
