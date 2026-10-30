import numpy as np
import pandas as pd
import pytest

from reviewlens.eval.bootstrap import bootstrap_macro_f1
from reviewlens.eval.errors import sample_errors, summarize_tags
from reviewlens.eval.metrics import macro_f1
from reviewlens.eval.robustness import add_typos, append_emoji, insert_negation, truncate


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    good = np.where(rng.random(n) < 0.8, y, rng.integers(0, 3, n))
    bad = np.where(rng.random(n) < 0.5, y, rng.integers(0, 3, n))
    return y, good, bad


def test_bootstrap_point_matches_sklearn_and_ci_brackets_it():
    y, good, bad = _data()
    r = bootstrap_macro_f1(y, {"bad": bad, "good": good}, n_resamples=300, seed=1)
    assert r["good"]["point"] == pytest.approx(macro_f1(y, good))
    assert r["good"]["ci_low"] <= r["good"]["point"] <= r["good"]["ci_high"]


def test_bootstrap_difference_is_paired_and_detects_the_better_model():
    y, good, bad = _data()
    r = bootstrap_macro_f1(y, {"bad": bad, "good": good}, n_resamples=300, seed=1)
    d = r["diff_good_minus_bad"]
    assert d["point"] > 0 and d["ci_low"] > 0
    same = bootstrap_macro_f1(y, {"a": good, "b": good}, n_resamples=100, seed=1)["diff_b_minus_a"]
    assert same["ci_low"] == same["ci_high"] == 0.0  # identical predictions: zero width


def test_bootstrap_is_reproducible():
    y, good, bad = _data()
    a = bootstrap_macro_f1(y, {"bad": bad, "good": good}, n_resamples=50, seed=3)
    b = bootstrap_macro_f1(y, {"bad": bad, "good": good}, n_resamples=50, seed=3)
    assert a == b


def test_typos_keep_word_shape_and_rate_zero_is_identity():
    text = "this keyboard feels absolutely wonderful today"
    assert add_typos(text, 0.0, np.random.default_rng(0)) == text
    out = add_typos(text, 1.0, np.random.default_rng(0))
    for a, b in zip(text.split(), out.split(), strict=True):
        assert sorted(a) == sorted(b) and a[0] == b[0] and a[-1] == b[-1]


def test_truncate_and_emoji_and_negation():
    assert truncate("a b c d e", 3) == "a b c"
    assert truncate("a b", 8) == "a b"
    assert append_emoji("ok", ["🎮"], np.random.default_rng(0)).startswith("ok 🎮")
    assert insert_negation("I love this game", ["love"]) == "I not love this game"
    assert insert_negation("Arrived on Tuesday", ["love"]) is None


def test_sample_errors_only_returns_misclassified_rows_and_summary_rejects_bad_tags():
    df = pd.DataFrame({"text": list("abcdef"), "label_id": [0, 1, 2, 0, 1, 2], "rating": [1] * 6})
    pred = pd.Series([0, 1, 0, 1, 1, 1])  # wrong on rows 2, 3, 5
    out = sample_errors(df, {"m": pred}, "m", n=10, seed=0)
    assert out["row"].tolist() == [2, 3, 5]
    assert out["true"].tolist() == ["positive", "negative", "positive"]
    with pytest.raises(ValueError):
        summarize_tags(pd.DataFrame({"tag": ["mixed", "other"]}), ["other"])
    s = summarize_tags(pd.DataFrame({"tag": ["other", "other"]}), ["other", "sarcasm"])
    assert s["counts"] == {"other": 2, "sarcasm": 0}
