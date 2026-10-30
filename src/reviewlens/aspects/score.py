"""Aspect-level sentiment: map sentences to aspects, score them, aggregate per review."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from reviewlens.aspects.extract import embedding_aspects, split_sentences
from reviewlens.aspects.lexicon import build_patterns, keyword_aspects
from reviewlens.data.clean import LABELS

logger = logging.getLogger(__name__)


class AspectAnalyzer:
    """Keyword + embedding aspect mapping with a sentence-level sentiment model.

    `predict_proba` maps a list of sentences to class probabilities (columns follow `LABELS`).
    `encoder` has an `encode(list[str], ...)` method returning embeddings (sentence-transformers).
    """

    def __init__(self, cfg: dict[str, Any], encoder,
                 predict_proba: Callable[[list[str]], np.ndarray],
                 threshold: float | None = None, use_embedding: bool = True) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.predict_proba = predict_proba
        self.use_embedding = use_embedding
        self.threshold = cfg["similarity"]["threshold"] if threshold is None else threshold
        self.patterns = build_patterns(cfg["lexicon"])
        self.aspects = list(cfg["lexicon"])
        seeds = [s for a in self.aspects for s in cfg["lexicon"][a]["seeds"]]
        emb = self._encode(seeds)
        sizes = [len(cfg["lexicon"][a]["seeds"]) for a in self.aspects]
        cuts = np.cumsum([0, *sizes])
        cents = np.stack([emb[cuts[i]:cuts[i + 1]].mean(axis=0) for i in range(len(self.aspects))])
        self.centroids = cents / np.linalg.norm(cents, axis=1, keepdims=True)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(
            texts, batch_size=self.cfg["embedding"]["batch_size"], show_progress_bar=False,
            normalize_embeddings=True))

    def map_aspects(self, sentences: list[str]) -> list[list[str]]:
        """Aspects per sentence: keyword hits first, then embedding matches not already found."""
        found = [keyword_aspects(s, self.patterns) for s in sentences]
        if self.use_embedding and sentences:
            sims = embedding_aspects(
                self._encode(sentences), self.centroids, self.aspects, self.threshold,
                self.cfg["similarity"]["max_aspects_per_sentence"])
            found = [kw + [a for a in emb if a not in kw]
                     for kw, emb in zip(found, sims, strict=True)]
        return found

    def analyze_sentences(self, sentences: list[str]) -> list[dict[str, Any]]:
        """Per sentence: its aspects and class probabilities."""
        aspects = self.map_aspects(sentences)
        probs = self.predict_proba(sentences) if sentences else np.zeros((0, len(LABELS)))
        return [{"sentence": s, "aspects": a, "probs": p}
                for s, a, p in zip(sentences, aspects, probs, strict=True)]

    def analyze_reviews(self, texts: list[str]) -> list[dict[str, dict[str, Any]]]:
        """For each review, `{aspect: {sentiment, probs, n_sentences}}` for the aspects it mentions.

        The sentence sentiments of one aspect are averaged (probabilities), then argmax is taken.
        """
        split = [split_sentences(t, self.cfg["split"]["min_words"]) for t in texts]
        flat = [s for ss in split for s in ss]
        rows = self.analyze_sentences(flat)
        out, pos = [], 0
        for ss in split:
            acc: dict[str, list[np.ndarray]] = defaultdict(list)
            for row in rows[pos:pos + len(ss)]:
                for a in row["aspects"]:
                    acc[a].append(row["probs"])
            pos += len(ss)
            out.append({
                a: {"sentiment": LABELS[int(np.mean(p, axis=0).argmax())],
                    "probs": np.mean(p, axis=0).tolist(), "n_sentences": len(p)}
                for a, p in acc.items()
            })
        return out


def aggregate_aspects(per_review: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Dataset-level table: for each aspect, how many reviews mention it and the sentiment mix."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(LABELS, 0))
    for review in per_review:
        for a, info in review.items():
            counts[a][info["sentiment"]] += 1
    out = {}
    for a, c in counts.items():
        n = sum(c.values())
        out[a] = {"n_reviews": n, **{f"share_{k}": v / n for k, v in c.items()},
                  "net_sentiment": (c["positive"] - c["negative"]) / n}
    return out
