import numpy as np
import pandas as pd
import pytest

from reviewlens.explain.lr_coefs import top_ngrams
from reviewlens.explain.render import render_example
from reviewlens.explain.token_attr import merge_wordpieces
from reviewlens.models.baseline import build_pipeline
from reviewlens.topics.trends import episodes, flag_rising, monthly_counts, n_tests


def _frame(months, topic_by_month):
    rows = []
    for m, (n, k) in zip(months, topic_by_month, strict=True):
        rows += [{"date": f"{m}-15", "t": "a"}] * k + [{"date": f"{m}-15", "t": "b"}] * (n - k)
    return pd.DataFrame(rows)


MONTHS = [f"2020-{m:02d}" for m in range(1, 13)]


def test_monthly_counts_fills_empty_months():
    df = pd.DataFrame({"date": ["2020-01-03", "2020-03-09"], "t": ["a", "b"]})
    c = monthly_counts(df, "t", ["a", "b"])
    assert list(c.index.astype(str)) == ["2020-01", "2020-02", "2020-03"]
    assert c.loc["2020-02"].sum() == 0


def test_flag_rising_flags_a_real_jump_and_ignores_a_flat_series():
    flat = monthly_counts(_frame(MONTHS, [(100, 20)] * 12), "t", ["a", "b"])
    kw = {"min_total": 50, "baseline_months": 6, "z_threshold": 3.0, "min_rise": 0.05,
          "min_count": 5}
    assert flag_rising(flat, **kw).empty
    jump = monthly_counts(_frame(MONTHS, [(100, 20)] * 8 + [(100, 40)] + [(100, 20)] * 3),
                          "t", ["a", "b"])
    f = flag_rising(jump, **kw)
    assert f["month"].tolist() == ["2020-09"] and f["category"].tolist() == ["a"]
    assert f["z"].iloc[0] > 3 and f["baseline_share"].iloc[0] == pytest.approx(0.2)


def test_flag_rising_skips_months_with_too_few_reviews_and_counts_tests():
    sparse = monthly_counts(_frame(MONTHS, [(100, 20)] * 6 + [(10, 8)] + [(100, 20)] * 5),
                            "t", ["a", "b"])
    f = flag_rising(sparse, 50, 6, 3.0, 0.05, 5)
    assert f.empty  # the 10-review month is too small to test, and poisons later baselines
    assert n_tests(sparse, 50, 6) == 0  # no month has six valid months before it


def test_episodes_merge_consecutive_months():
    flags = pd.DataFrame({"month": ["2020-03", "2020-04", "2020-08"], "category": ["a"] * 3,
                          "count": 9, "n": 100, "share": [.3, .4, .3], "baseline_share": .1,
                          "z": [3.5, 5.0, 3.2]})
    e = episodes(flags)
    assert e["n_months"].tolist() == [2, 1] and e["peak_z"].tolist() == [5.0, 3.2]


def test_merge_wordpieces_joins_continuations_but_not_special_tokens():
    toks = ["[CLS]", "un", "##believ", "##able", "game", "[SEP]"]
    assert merge_wordpieces(toks) == [[0], [1, 2, 3], [4], [5]]


def test_top_ngrams_returns_largest_and_smallest_coefficients():
    cfg = {"tfidf": {"word": {"ngram_range": [1, 1], "min_df": 1, "max_features": 100,
                              "sublinear_tf": True}}, "lr": {"max_iter": 500,
           "class_weight": "balanced"}, "seed": 0}
    texts = ["great fun game", "great love it", "awful broken thing", "awful waste money",
             "okay average fine", "okay fine average"]
    pipe = build_pipeline(cfg, use_char=False, C=1.0).fit(texts, [2, 2, 0, 0, 1, 1])
    top = top_ngrams(pipe, 2)
    assert "awful" in [f for f, _ in top["negative"]["positive"]]
    assert "great" in [f for f, _ in top["positive"]["positive"]]
    assert top["negative"]["positive"][0][1] >= top["negative"]["positive"][1][1]


def test_render_example_escapes_html_and_hides_special_tokens():
    ex = {"id": "t-1", "true": "negative", "pred": 0, "probs": [0.7, 0.2, 0.1], "tag": "x",
          "words": ["[CLS]", "<b>bad</b>", "[SEP]"], "scores": [0.0, 0.5, 0.0],
          "is_special": [True, False, True]}
    out = render_example(ex)
    assert "&lt;b&gt;bad&lt;/b&gt;" in out and "<b>bad</b>" not in out and "[CLS]" not in out


def test_integrated_gradients_and_deletion_check_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("captum")
    from reviewlens.explain.token_attr import deletion_check, explain_text, top_words

    try:
        tok = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except OSError:
        pytest.skip("tokenizer not available offline")
    torch.manual_seed(0)
    model = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(
        vocab_size=30522, dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=3)).eval()
    r = explain_text(model, tok, "this unbelievable game is great fun", 32, n_steps=20)
    assert len(r["words"]) == len(r["scores"]) == len(r["is_special"])
    assert r["words"][0] == "[CLS]" and np.isfinite(r["scores"]).all() and np.isfinite(r["delta"])
    assert "[CLS]" not in [w for w, _ in top_words(r, 3)]
    check = deletion_check(model, [r], k=2, n_random=3, seed=0)
    assert check["n_reviews"] == 1 and 0.0 <= check["share_top_beats_random"] <= 1.0
