from pathlib import Path

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PROBS = {"negative": 0.8, "neutral": 0.15, "positive": 0.05}
RESPONSE = {
    "sentiment": "negative", "probabilities": PROBS,
    "baseline": {"label": "neutral", "probabilities": PROBS},
    "topic": {"id": 0, "label": "Controller and accessory defects", "similarity": 0.63},
    "aspects": {"durability": {"sentiment": "negative", "n_sentences": 1, "probabilities": PROBS}},
    "explanation": {"tokens": [{"token": "<broke>", "score": 0.5},
                               {"token": "fine", "score": -0.1}],
                    "top_tokens": [{"token": "<broke>", "score": 0.5}]},
    "truncated": False, "latency_ms": 12.0,
}


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.ok = status, body, status < 400

    def json(self):
        return self._body


def _open(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    return at.sidebar.radio[0].set_value(page).run() if page != "Analyze" else at


def test_model_comparison_and_themes_pages_render_from_reports():
    for page in ("Model Comparison", "Themes & Trends"):
        at = _open(page)
        assert not at.exception, at.exception
        assert at.dataframe, f"{page} shows no tables"


def test_analyze_shows_results_and_escapes_the_review_words(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, RESPONSE))
    at = _open("Analyze")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric][:2] == ["negative", "neutral"]
    html = " ".join(m.value for m in at.markdown if 'class="w"' in m.value)  # the unsafe-HTML block
    assert "&lt;broke&gt;" in html and "<broke>" not in html


def test_analyze_reports_an_unreachable_api(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", down)
    at = _open("Analyze")
    at.button[0].click().run()
    assert at.error and "Could not reach the API" in at.error[0].value


def test_analyze_shows_the_apis_validation_message(monkeypatch):
    body = {"detail": [{"msg": "String should have at most 5000 characters"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(422, body))
    at = _open("Analyze")
    at.button[0].click().run()
    assert at.error and "at most 5000" in at.error[0].value
