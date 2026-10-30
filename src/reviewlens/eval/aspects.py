"""Evaluation of aspect mapping and aspect sentiment against hand-labeled sentences."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def parse_labels(cell: Any) -> dict[str, str]:
    """`"graphics:positive|story:negative"` -> {aspect: sentiment}; empty or NaN -> {}."""
    if not isinstance(cell, str) or not cell.strip():
        return {}
    return dict(item.strip().split(":") for item in cell.split("|"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate(gold: list[dict[str, str]], pred_aspects: list[list[str]],
             pred_sentiment: list[str]) -> dict[str, Any]:
    """Score aspect detection and aspect sentiment.

    `gold[i]` maps aspect to sentiment for sentence i; `pred_aspects[i]` is the predicted aspect
    list and `pred_sentiment[i]` the one sentence-level sentiment the system gives every aspect
    of that sentence. Returns:
    - detection: micro precision/recall/F1 over (sentence, aspect) pairs, and the share of
      sentences whose predicted aspect set equals the gold set exactly;
    - sentiment: accuracy on gold pairs the system also detected, and end-to-end accuracy over
      all gold pairs (a missed aspect counts as wrong), both with Wilson intervals.
    """
    tp = fp = fn = exact = 0
    both = both_ok = 0
    n_pairs = 0
    for g, pa, ps in zip(gold, pred_aspects, pred_sentiment, strict=True):
        gs, ps_set = set(g), set(pa)
        tp += len(gs & ps_set)
        fp += len(ps_set - gs)
        fn += len(gs - ps_set)
        exact += gs == ps_set
        n_pairs += len(gs)
        for a in gs & ps_set:
            both += 1
            both_ok += g[a] == ps
    n = len(gold)
    return {
        "n_sentences": n,
        "n_gold_pairs": n_pairs,
        "detection": _prf(tp, fp, fn) | {"tp": tp, "fp": fp, "fn": fn,
                                         "exact_set_match": exact / n if n else 0.0},
        "sentiment_on_detected": {
            "n": both, "accuracy": both_ok / both if both else 0.0,
            "ci95": wilson(both_ok, both)},
        "sentiment_end_to_end": {
            "n": n_pairs, "accuracy": both_ok / n_pairs if n_pairs else 0.0,
            "ci95": wilson(both_ok, n_pairs)},
    }


def load_gold(path: str) -> pd.DataFrame:
    """Read a labeled CSV and add a `gold` column of parsed {aspect: sentiment} dicts."""
    df = pd.read_csv(path)
    df["gold"] = df["labels"].map(parse_labels)
    return df
