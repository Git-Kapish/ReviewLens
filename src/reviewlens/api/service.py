"""Inference service: loads every model once and analyzes reviews.

Kept free of FastAPI so the same code can be used from scripts and tests.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import joblib
import numpy as np
from safetensors.numpy import load_file

from reviewlens.aspects.score import AspectAnalyzer
from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.explain.token_attr import explain_text, top_words
from reviewlens.models.predict import load_model, predict_proba
from reviewlens.topics.labels import load_labels

logger = logging.getLogger(__name__)


def _probs(p: np.ndarray) -> dict[str, float]:
    return {name: round(float(v), 6) for name, v in zip(LABELS, p, strict=True)}


class ReviewLensService:
    """DistilBERT + baseline sentiment, nearest topic, aspects and token attributions."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.cfg = load_config("api")
        self.tcfg = load_config("transformer")
        topics_cfg = load_config("topics")
        logger.info("loading sentiment models")
        self.model, self.tok = load_model(resolve_path(self.tcfg["paths"]["model_dir"]))
        # our own artifact from `make baseline`, not untrusted input
        baseline_dir = resolve_path(load_config("baseline")["paths"]["model_dir"])
        self.baseline = joblib.load(baseline_dir / "baseline.joblib")

        logger.info("loading sentence encoder and topic centroids")
        self.encoder = SentenceTransformer(topics_cfg["embedding"]["model_name"])
        self.encoder.max_seq_length = topics_cfg["embedding"]["max_seq_length"]
        topic_dir = resolve_path(self.cfg["topic"]["model_dir"])
        sizes = json.loads((topic_dir / "topics.json").read_text(encoding="utf-8"))["topic_sizes"]
        all_ids = sorted(int(k) for k in sizes)  # rows of the centroid matrix, outlier -1 first
        cents = load_file(str(topic_dir / "topic_embeddings.safetensors"))["topic_embeddings"]
        if cents.shape[0] != len(all_ids):
            raise ValueError(f"{cents.shape[0]} centroids but {len(all_ids)} topic ids")
        keep = [i for i, t in enumerate(all_ids) if t != -1]  # new reviews never get 'unassigned'
        self.topic_ids = [all_ids[i] for i in keep]
        cents = cents[keep]
        self.centroids = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        self.topic_labels = load_labels(resolve_path(self.cfg["topic"]["labels"]), self.topic_ids)

        self.aspects = AspectAnalyzer(load_config("aspects"), self.encoder, self._sentence_proba)
        self._lock = threading.Lock()  # Captum hooks the embedding layer: one explanation at a time
        logger.info("service ready")

    def _sentence_proba(self, sentences: list[str]) -> np.ndarray:
        return predict_proba(self.model, self.tok, sentences, self.tcfg["max_len"], batch_size=64)

    def health(self) -> dict[str, str]:
        """Names of the loaded models."""
        return {"sentiment": self.tcfg["model_name"], "baseline": "tfidf_logreg",
                "encoder": load_config("topics")["embedding"]["model_name"],
                "topics": f"{len(self.topic_ids)} topics"}

    def _topic(self, text: str) -> dict[str, Any]:
        emb = np.asarray(self.encoder.encode([text], normalize_embeddings=True))[0]
        sims = self.centroids @ emb
        best = int(sims.argmax())
        t = self.topic_ids[best]
        return {"id": t, "label": self.topic_labels[t], "similarity": round(float(sims[best]), 4)}

    def analyze(self, text: str, explain: bool = True) -> dict[str, Any]:
        """Analyze one review (already validated and stripped)."""
        t0 = time.perf_counter()
        probs = predict_proba(self.model, self.tok, [text], self.tcfg["max_len"])[0]
        label = LABELS[int(probs.argmax())]
        base = self.baseline.predict_proba([text])[0]
        aspects = self.aspects.analyze_reviews([text])[0]
        out: dict[str, Any] = {
            "sentiment": label,
            "probabilities": _probs(probs),
            "baseline": {"label": LABELS[int(base.argmax())], "probabilities": _probs(base)},
            "topic": self._topic(text) if label == "negative" else None,
            "aspects": {a: {"sentiment": v["sentiment"], "n_sentences": v["n_sentences"],
                            "probabilities": _probs(np.asarray(v["probs"]))}
                        for a, v in aspects.items()},
            "explanation": None,
            "truncated": len(self.tok(text, truncation=False)["input_ids"]) > self.tcfg["max_len"],
        }
        if explain:
            with self._lock:
                r = explain_text(self.model, self.tok, text, self.tcfg["max_len"],
                                 n_steps=self.cfg["explain"]["n_steps"])
            out["explanation"] = {
                "tokens": [{"token": w, "score": round(s, 4)} for w, s, sp in
                           zip(r["words"], r["scores"], r["is_special"], strict=True) if not sp],
                "top_tokens": [{"token": w, "score": round(s, 4)}
                               for w, s in top_words(r, self.cfg["explain"]["top_k"])],
            }
        out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out

    def analyze_batch(self, texts: list[str], explain: bool = False) -> list[dict[str, Any]]:
        """Analyze several reviews; attributions are off by default (they dominate the cost)."""
        return [self.analyze(t, explain=explain) for t in texts]
