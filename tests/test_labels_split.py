import numpy as np
import pandas as pd
import pytest

from reviewlens.config import load_config
from reviewlens.data.clean import rating_to_label
from reviewlens.data.split import split_dataset


@pytest.mark.parametrize(
    "rating,label",
    [(1.0, "negative"), (2.0, "negative"), (3.0, "neutral"), (4.0, "positive"), (5.0, "positive")],
)
def test_label_map(rating, label):
    assert rating_to_label(rating, load_config("data")["labels"]) == label


def test_label_map_rejects_unknown_rating():
    with pytest.raises(ValueError):
        rating_to_label(0.0, load_config("data")["labels"])


def _frame(n=2000):
    rng = np.random.default_rng(0)
    label = rng.choice(["negative", "neutral", "positive"], size=n, p=[0.15, 0.08, 0.77])
    return pd.DataFrame({"text": [f"review {i}" for i in range(n)], "label": label,
                         "timestamp": rng.permutation(n)})


def test_split_is_disjoint_complete_and_time_ordered():
    cfg = load_config("data")
    df = _frame()
    parts = split_dataset(df, cfg)
    assert sum(len(p) for p in parts.values()) == len(df)
    texts = [set(p["text"]) for p in parts.values()]
    assert sum(len(t) for t in texts) == len(set().union(*texts))  # no overlap
    # time hold-out is strictly the latest window
    assert parts["timehold"]["timestamp"].min() > max(
        parts[k]["timestamp"].max() for k in ("train", "val", "test")
    )
    assert len(parts["timehold"]) == pytest.approx(0.10 * len(df), abs=1)


def test_split_is_stratified_and_deterministic():
    cfg = load_config("data")
    df = _frame()
    a, b = split_dataset(df, cfg), split_dataset(df, cfg)
    assert a["train"]["text"].tolist() == b["train"]["text"].tolist()
    overall = df["label"].value_counts(normalize=True)
    for k in ("train", "val", "test"):
        share = a[k]["label"].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.02
    rest = len(df) - len(a["timehold"])
    assert len(a["train"]) == pytest.approx(0.8 * rest, abs=2)
    assert len(a["val"]) == pytest.approx(0.1 * rest, abs=2)
