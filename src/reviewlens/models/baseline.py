"""TF-IDF + logistic regression baseline."""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from reviewlens.data.clean import normalize_for_tfidf
from reviewlens.eval.metrics import macro_f1

logger = logging.getLogger(__name__)


def _vectorizer(spec: dict[str, Any]) -> TfidfVectorizer:
    spec = dict(spec)
    spec["ngram_range"] = tuple(spec["ngram_range"])
    return TfidfVectorizer(preprocessor=normalize_for_tfidf, **spec)


def build_pipeline(cfg: dict, use_char: bool, C: float) -> Pipeline:
    """Unfitted pipeline that takes raw review text (normalization is part of the vectorizer)."""
    feats = [("word", _vectorizer(cfg["tfidf"]["word"]))]
    if use_char:
        feats.append(("char", _vectorizer(cfg["tfidf"]["char"])))
    lr = LogisticRegression(
        C=C,
        class_weight=cfg["lr"]["class_weight"],
        max_iter=cfg["lr"]["max_iter"],
        random_state=cfg["seed"],
    )
    return Pipeline([("tfidf", FeatureUnion(feats)), ("lr", lr)])


def select_baseline(
    train: pd.DataFrame, val: pd.DataFrame, cfg: dict
) -> tuple[Pipeline, list[dict[str, Any]]]:
    """Grid search on validation macro-F1. Vectorizers are fit on `train` only.

    Returns the best fitted pipeline (trained on train, not refit on train+val) and one result
    row per grid point.
    """
    results, best, best_f1 = [], None, -1.0
    grid = itertools.product(cfg["grid"]["use_char"], cfg["grid"]["C"])
    for use_char, C in grid:
        t0 = time.perf_counter()
        pipe = build_pipeline(cfg, use_char, C).fit(train["text"], train["label_id"])
        f1 = macro_f1(val["label_id"], pipe.predict(val["text"]))
        results.append({
            "use_char": use_char, "C": C, "val_macro_f1": f1,
            "n_features": len(pipe["tfidf"].get_feature_names_out()),
            "fit_seconds": round(time.perf_counter() - t0, 1),
        })
        logger.info("use_char=%s C=%s val macro-F1=%.4f", use_char, C, f1)
        if f1 > best_f1:
            best, best_f1 = pipe, f1
    return best, results
