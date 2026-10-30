"""BERTopic on negative reviews: cached embeddings, clustering, topic words, coherence.

Heavy libraries (sentence-transformers, bertopic, umap, hdbscan, gensim) are imported inside the
functions that need them, so the pure helpers can be imported and tested without them.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reviewlens.config import set_seed
from reviewlens.data.clean import LABELS

logger = logging.getLogger(__name__)
NEGATIVE_ID = LABELS.index("negative")


def select_negatives(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All negative reviews from every split, with a unique `doc_id` and their `split`."""
    parts = []
    for name, df in splits.items():
        neg = df[df["label_id"] == NEGATIVE_ID]
        parts.append(pd.DataFrame({
            "doc_id": [f"{name}-{i}" for i in neg.index],
            "split": name,
            "text": neg["text"].to_numpy(),
            "rating": neg["rating"].to_numpy(),
            "timestamp": neg["timestamp"].to_numpy(),
            "date": neg["date"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def embed_texts(texts: list[str], cfg: dict[str, Any], cache_path: str | Path) -> np.ndarray:
    """Sentence embeddings cached on disk; reused only if the texts and model setting match."""
    emb_cfg = cfg["embedding"]
    key = hashlib.sha1(
        (emb_cfg["model_name"] + str(emb_cfg["max_seq_length"]) + "\n".join(texts)).encode()
    ).hexdigest()
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["key"]) == key:
            logger.info("embeddings loaded from %s", cache_path)
            return cached["embeddings"]
        logger.info("embedding cache is stale, recomputing")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(emb_cfg["model_name"])
    model.max_seq_length = emb_cfg["max_seq_length"]
    emb = model.encode(texts, batch_size=emb_cfg["batch_size"], show_progress_bar=False,
                       normalize_embeddings=True).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=emb, key=np.array(key))
    return emb


def fit_topics(docs: list[str], embeddings: np.ndarray, cfg: dict[str, Any],
               min_cluster_size: int):
    """Fit BERTopic on precomputed embeddings; returns `(model, topics)` aligned with `docs`."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    set_seed(cfg["seed"])
    u, h, v = cfg["umap"], cfg["hdbscan"], cfg["vectorizer"]
    model = BERTopic(
        embedding_model=cfg["embedding"]["model_name"],
        umap_model=UMAP(n_neighbors=u["n_neighbors"], n_components=u["n_components"],
                        min_dist=u["min_dist"], metric=u["metric"], random_state=cfg["seed"]),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, min_samples=h["min_samples"],
                              metric=h["metric"],
                              cluster_selection_method=h["cluster_selection_method"],
                              prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=v["stop_words"],
                                         ngram_range=tuple(v["ngram_range"]), min_df=v["min_df"]),
        nr_topics=cfg["nr_topics"],
        top_n_words=cfg["top_n_words"],
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    return model, list(model.topics_)


def topic_words(model, topic_id: int, n: int, unigrams_only: bool = False) -> list[str]:
    """Top words of a topic from c-TF-IDF; optionally only single words (for coherence)."""
    words = [w for w, _ in model.get_topic(topic_id)]
    if unigrams_only:
        words = [w for w in words if " " not in w]
    return words[:n]


def coherence_cv(model, docs: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    """c_v coherence of the topics' top unigram words over `docs` (gensim, single process).

    Returns the mean and the per-topic values keyed by topic id. Outlier topic -1 is skipped.
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    from sklearn.feature_extraction.text import CountVectorizer

    analyzer = CountVectorizer(stop_words=cfg["vectorizer"]["stop_words"]).build_analyzer()
    tokens = [analyzer(d) for d in docs]
    dictionary = Dictionary(tokens)
    ids = sorted(t for t in model.get_topics() if t != -1)
    topics = [
        [w for w in topic_words(model, t, cfg["coherence"]["n_words"], unigrams_only=True)
         if w in dictionary.token2id]
        for t in ids
    ]
    cm = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary, coherence="c_v",
                        processes=1)
    per_topic = cm.get_coherence_per_topic()
    return {"mean": float(np.mean(per_topic)),
            "per_topic": {int(t): float(c) for t, c in zip(ids, per_topic, strict=True)}}


def representative_docs(embeddings: np.ndarray, topics: np.ndarray, n: int) -> dict[int, list[int]]:
    """Row indices of the `n` documents closest (cosine) to each topic's centroid."""
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    out: dict[int, list[int]] = {}
    for t in sorted(set(topics.tolist()) - {-1}):
        rows = np.flatnonzero(topics == t)
        centroid = unit[rows].mean(axis=0)
        sims = unit[rows] @ (centroid / np.linalg.norm(centroid))
        out[int(t)] = [int(rows[i]) for i in np.argsort(-sims)[:n]]
    return out
