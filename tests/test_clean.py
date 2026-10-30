import pandas as pd
import pytest

from reviewlens.config import load_config
from reviewlens.data.clean import clean_reviews, clean_text, is_english, normalize_for_tfidf


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("Great<br />game!<br/>  Loved\n\nit ") == "Great game! Loved it"


def test_normalize_for_tfidf_keeps_negation_and_squeezes():
    out = normalize_for_tfidf("NOT good!!! sooooo bad http://x.co/a")
    assert out == "not good!! soo bad url"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This game is really fun and the controls feel great to me.", True),
        ("Este juego es muy divertido y los controles son excelentes.", False),
        ("Great", True),
        ("这个游戏很好", False),
        ("12345 !!!! 99999 ???? 000000 ,,,,,,,,,,", False),  # nothing to detect
    ],
)
def test_is_english(text, expected):
    assert is_english(text, min_chars=30, max_chars=500) is expected


def _raw(rows):
    base = {"title": "t", "timestamp": 0, "verified_purchase": True}
    return pd.DataFrame([base | r for r in rows])


def test_clean_reviews_funnel_dedupe_and_labels():
    cfg = load_config("data")
    en = "This game is really fun and the controls feel great to me."
    df = _raw(
        [
            {"text": en, "rating": 5.0, "timestamp": 2},
            {"text": en.upper(), "rating": 1.0, "timestamp": 1},  # case-insensitive duplicate
            {"text": "  ", "rating": 3.0},
            {"text": None, "rating": 2.0},
            {"text": "Este juego es muy divertido y los controles son excelentes.", "rating": 4.0},
            {"text": "Broke after a week", "rating": 2.0, "timestamp": 3},
        ]
    )
    funnel: dict[str, int] = {}
    out = clean_reviews(df, cfg, funnel)
    assert funnel == {"raw": 6, "non_empty": 3 + 1, "deduped": 3, "english": 2, "final": 2}
    # the earliest copy of the duplicate survives, so its rating (1.0 -> negative) is kept
    assert sorted(out["label"]) == ["negative", "negative"]
    assert set(out["label_id"]) == {0}
