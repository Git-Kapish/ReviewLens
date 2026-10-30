"""Sentence splitting, embedding-similarity aspect mapping and labeling-set sampling."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Split on sentence punctuation followed by a capital, and on line breaks.

    Fragments with fewer than `min_words` words are dropped. Text without punctuation (common in
    short reviews) stays one sentence; a sentence that holds two opinions is not split further.
    """
    parts = (p.strip() for p in _SPLIT.split(text.strip()))
    return [p for p in parts if len(p.split()) >= min_words]


def embedding_aspects(
    sentence_emb: np.ndarray, centroids: np.ndarray, aspects: list[str],
    threshold: float, top_k: int,
) -> list[list[str]]:
    """For each sentence, up to `top_k` aspects whose centroid cosine similarity >= threshold.

    Embeddings must be L2-normalized, so the dot product is the cosine. Best match first.
    """
    sims = sentence_emb @ centroids.T
    out = []
    for row in sims:
        order = np.argsort(-row)[:top_k]
        out.append([aspects[i] for i in order if row[i] >= threshold])
    return out


def sample_sentences(
    df: pd.DataFrame, per_rating: int, seed: int, min_words: int, max_words: int
) -> pd.DataFrame:
    """One random sentence from each of `per_rating` random reviews per star rating.

    Returns `review_id` (the row index of `df`), `rating` and `text`, so the draw is reproducible
    and unbiased towards aspects (no keyword filtering).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rating, g in df.groupby("rating"):
        order = rng.permutation(len(g))
        taken = 0
        for i in order:
            sents = [s for s in split_sentences(g["text"].iloc[i], min_words)
                     if len(s.split()) <= max_words]
            if not sents:
                continue
            rows.append({"review_id": g.index[i], "rating": int(rating),
                         "text": sents[int(rng.integers(len(sents)))]})
            taken += 1
            if taken == per_rating:
                break
    return pd.DataFrame(rows)
