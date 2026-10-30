"""Cleaning, language filtering and label mapping for raw reviews."""

from __future__ import annotations

import logging
import re

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0  # langdetect is otherwise non-deterministic

LABELS = ["negative", "neutral", "positive"]

_HTML = re.compile(r"<[^>]+>")  # Amazon text contains literal <br /> tags
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")


def clean_text(text: str) -> str:
    """Light cleaning kept for every model: drop HTML tags, collapse whitespace."""
    return _WS.sub(" ", _HTML.sub(" ", text)).strip()


def normalize_for_tfidf(text: str) -> str:
    """Heavier normalization for the TF-IDF path only: lowercase, URL token, squeeze repeats."""
    text = _URL.sub(" url ", text.lower())
    return _WS.sub(" ", _REPEAT.sub(r"\1\1", text)).strip()


def is_english(text: str, min_chars: int, max_chars: int) -> bool:
    """Language filter.

    Texts under `min_chars` are too short to identify, so only an alphabet check applies
    (reject if most letters are non-ASCII).
    """
    if len(text) < min_chars:
        letters = [c for c in text if c.isalpha()]
        return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.5
    try:
        return detect(text[:max_chars]) == "en"
    except LangDetectException:  # no detectable features (e.g. only digits/symbols)
        return False


def rating_to_label(rating: float, labels_cfg: dict[str, list[int]]) -> str:
    """Map a star rating to negative / neutral / positive using the config label map."""
    stars = int(round(rating))
    for name, group in labels_cfg.items():
        if stars in group:
            return name
    raise ValueError(f"rating {rating} has no label")


def clean_reviews(
    df: pd.DataFrame, cfg: dict, funnel: dict[str, int] | None = None
) -> pd.DataFrame:
    """Clean, dedupe, language-filter, cap and label a raw sample.

    Row counts after each step are written into `funnel` if given.
    """
    funnel = {} if funnel is None else funnel
    funnel["raw"] = len(df)
    df = df.assign(text=df["text"].fillna("").map(clean_text), title=df["title"].fillna(""))
    df = df[df["text"] != ""]
    funnel["non_empty"] = len(df)

    df = df.sort_values("timestamp")  # keep the earliest copy of a duplicate
    df = df[~df["text"].str.lower().duplicated()]
    funnel["deduped"] = len(df)

    c = cfg["clean"]
    df = df[df["text"].map(lambda t: is_english(t, c["lang_min_chars"], c["lang_max_chars"]))]
    funnel["english"] = len(df)

    target = cfg["sample"]["target_rows"]
    if len(df) > target:
        df = df.sample(target, random_state=cfg["seed"])
    funnel["final"] = len(df)

    df = df.assign(label=df["rating"].map(lambda r: rating_to_label(r, cfg["labels"])))
    df = df.assign(label_id=df["label"].map(LABELS.index))
    df = df.assign(date=pd.to_datetime(df["timestamp"], unit="ms"))
    logger.info("cleaning funnel: %s", funnel)
    return df.sort_values("timestamp").reset_index(drop=True)
