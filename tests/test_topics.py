import json

import numpy as np
import pandas as pd
import pytest

from reviewlens.topics.bertopic_runner import representative_docs, select_negatives
from reviewlens.topics.labels import load_labels


def _split(label_ids, start=0):
    n = len(label_ids)
    return pd.DataFrame(
        {"text": [f"t{i}" for i in range(n)], "label_id": label_ids, "rating": [1] * n,
         "timestamp": range(n), "date": ["2020-01-01"] * n},
        index=range(start, start + n),
    )


def test_select_negatives_keeps_only_negatives_with_unique_ids():
    out = select_negatives({"train": _split([0, 1, 2, 0]), "val": _split([0, 2])})
    assert out["split"].tolist() == ["train", "train", "val"]
    assert out["doc_id"].is_unique and out["doc_id"].tolist() == ["train-0", "train-3", "val-0"]


def test_representative_docs_are_closest_to_the_topic_centroid():
    emb = np.array([[1, 0], [0.9, 0.1], [0.2, 1.0], [0, 1], [1, 1]], dtype=float)
    topics = np.array([0, 0, 1, 1, -1])
    reps = representative_docs(emb, topics, 1)
    assert set(reps) == {0, 1}  # the outlier topic is skipped
    assert reps[0][0] in (0, 1) and reps[1][0] in (2, 3)
    assert 4 not in sum(reps.values(), [])


def test_load_labels_requires_exact_topic_coverage(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"topics": {"0": {"label": "a"}, "1": {"label": "b"}}}))
    assert load_labels(path, [-1, 0, 1]) == {0: "a", 1: "b"}
    with pytest.raises(ValueError):
        load_labels(path, [0, 1, 2])  # topic 2 has no label
    with pytest.raises(ValueError):
        load_labels(path, [0])  # label for a topic that does not exist
