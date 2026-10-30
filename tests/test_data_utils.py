import pandas as pd

from reviewlens.data import load
from reviewlens.data.profile import profile_split


def test_reservoir_sample_has_k_unique_lines_and_is_seeded(monkeypatch):
    lines = [f"line{i}" for i in range(500)]
    monkeypatch.setattr(load, "stream_lines", lambda category=None: iter(lines))
    a = load.reservoir_sample_lines(50, seed=3)
    assert len(a) == len(set(a)) == 50
    assert a == load.reservoir_sample_lines(50, seed=3)
    assert a != load.reservoir_sample_lines(50, seed=4)
    assert max(int(x[4:]) for x in a) > 250  # lines from the whole stream, not just the head


def test_reservoir_sample_returns_everything_when_k_exceeds_the_stream(monkeypatch):
    monkeypatch.setattr(load, "stream_lines", lambda category=None: iter(["a", "b", "c"]))
    assert sorted(load.reservoir_sample_lines(10, seed=1)) == ["a", "b", "c"]


def test_profile_split_reports_shares_lengths_and_dates():
    df = pd.DataFrame({
        "text": ["bad", "ok then fine", "great game with a long story here today"],
        "label": ["negative", "neutral", "positive"], "rating": [1, 3, 5],
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "verified_purchase": [True, False, True],
    })
    p = profile_split(df, short_tokens=4)
    assert p["rows"] == 3 and p["date_min"] == "2020-01-01" and p["date_max"] == "2021-01-01"
    assert p["class_count"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert p["share_under_4_tokens"] == round(2 / 3, 4)
    assert p["verified_purchase_share"] == round(2 / 3, 4)
