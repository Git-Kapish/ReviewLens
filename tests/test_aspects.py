import numpy as np
import pandas as pd
import pytest

from reviewlens.aspects.extract import embedding_aspects, sample_sentences, split_sentences
from reviewlens.aspects.lexicon import build_patterns, keyword_aspects
from reviewlens.aspects.score import AspectAnalyzer, aggregate_aspects
from reviewlens.config import load_config
from reviewlens.eval.aspects import evaluate, parse_labels, wilson

LEX = {"price_value": {"keywords": ["price", "worth", "money"]},
       "battery": {"keywords": ["battery life", "charge"]}}


def test_split_sentences_splits_on_punctuation_and_drops_fragments():
    text = "Great game. The graphics are lovely! Ok.\nBattery dies fast"
    assert split_sentences(text, min_words=3) == ["The graphics are lovely!", "Battery dies fast"]
    assert split_sentences(text, min_words=2)[0] == "Great game."
    assert split_sentences("no punctuation here at all") == ["no punctuation here at all"]


def test_keyword_match_is_whole_word_case_insensitive_and_allows_plurals():
    pats = build_patterns(LEX)
    assert keyword_aspects("The PRICE is fair", pats) == ["price_value"]
    assert keyword_aspects("Not worth the moneys", pats) == ["price_value"]
    assert keyword_aspects("Great pricey thing", pats) == []  # 'price' inside 'pricey' is not a hit
    assert keyword_aspects("Battery life is bad, hard to charge", pats) == ["battery"]


def test_embedding_aspects_respects_threshold_and_top_k():
    cents = np.eye(4)[:3]
    emb = np.array([[1.0, 0, 0, 0], [0.6, 0.6, 0.0, 0.0], [0.1, 0.1, 0.1, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    out = embedding_aspects(emb, cents, ["a", "b", "c"], threshold=0.5, top_k=2)
    assert out[0] == ["a"] and set(out[1]) == {"a", "b"} and out[2] == []
    top1 = embedding_aspects(emb, cents, ["a", "b", "c"], threshold=0.5, top_k=1)
    assert top1[1] in (["a"], ["b"])


def test_sample_sentences_is_reproducible_and_stratified_by_rating():
    df = pd.DataFrame({"rating": [1, 1, 2, 2], "text": ["Bad thing here. Very bad indeed.",
                       "Awful stuff overall", "Not good at all", "It is meh really"]})
    a = sample_sentences(df, 1, seed=1, min_words=3, max_words=10)
    assert a.equals(sample_sentences(df, 1, seed=1, min_words=3, max_words=10))
    assert sorted(a["rating"]) == [1, 2]


def test_parse_labels_and_evaluate():
    assert parse_labels("graphics:positive|story:negative") == {
        "graphics": "positive", "story": "negative"}
    assert parse_labels(np.nan) == {} and parse_labels("") == {}
    gold = [{"graphics": "positive"}, {}, {"price_value": "negative", "story": "negative"}]
    pred_a = [["graphics"], ["story"], ["price_value"]]
    pred_s = ["positive", "positive", "positive"]
    r = evaluate(gold, pred_a, pred_s)
    assert (r["detection"]["tp"], r["detection"]["fp"], r["detection"]["fn"]) == (2, 1, 1)
    assert r["sentiment_on_detected"]["n"] == 2 and r["sentiment_on_detected"]["accuracy"] == 0.5
    e2e = r["sentiment_end_to_end"]
    assert e2e["n"] == 3 and e2e["accuracy"] == pytest.approx(1 / 3)
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8


class _FakeEncoder:
    """Maps a text to a one-hot over a few trigger words, so similarity is predictable."""

    def encode(self, texts, **_):
        keys = ["battery", "graphics"]
        v = np.array([[float(k in t.lower()) + 0.01 for k in keys] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_analyzer_maps_scores_and_aggregates_per_review():
    cfg = load_config("aspects")
    cfg["lexicon"] = {"battery": {"keywords": ["battery"], "seeds": ["battery"]},
                      "graphics": {"keywords": ["graphics"], "seeds": ["graphics"]}}
    cfg["similarity"]["threshold"] = 0.9

    def proba(ss):  # negative if 'bad' in the sentence, else positive
        return np.array([[0.8, 0.1, 0.1] if "bad" in s else [0.1, 0.1, 0.8] for s in ss])

    an = AspectAnalyzer(cfg, _FakeEncoder(), proba)
    reviews = ["The graphics are great. The battery is bad.", "Nothing relevant here"]
    out = an.analyze_reviews(reviews)
    assert out[0]["graphics"]["sentiment"] == "positive"
    assert out[0]["battery"]["sentiment"] == "negative"
    assert out[1] == {}
    agg = aggregate_aspects(out)
    assert agg["battery"]["n_reviews"] == 1 and agg["battery"]["net_sentiment"] == -1.0
