import pytest
from fastapi.testclient import TestClient

from reviewlens.api.main import create_app
from reviewlens.config import resolve_path

PROBS = {"negative": 0.7, "neutral": 0.2, "positive": 0.1}


class FakeService:
    """Stands in for ReviewLensService so the HTTP layer is tested without loading models."""

    def health(self):
        return {"sentiment": "fake"}

    def analyze(self, text, explain=True):
        neg = "bad" in text
        return {
            "sentiment": "negative" if neg else "positive",
            "probabilities": PROBS,
            "baseline": {"label": "negative", "probabilities": PROBS},
            "topic": {"id": 1, "label": "t", "similarity": 0.5} if neg else None,
            "aspects": {"price_value": {"sentiment": "negative", "n_sentences": 1,
                                        "probabilities": PROBS}},
            "explanation": {"tokens": [{"token": "bad", "score": 0.4}],
                            "top_tokens": [{"token": "bad", "score": 0.4}]} if explain else None,
            "truncated": False,
            "latency_ms": 1.0,
        }

    def analyze_batch(self, texts, explain=False):
        return [self.analyze(t, explain) for t in texts]


@pytest.fixture()
def client():
    with TestClient(create_app(FakeService())) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_returns_the_documented_fields(client):
    r = client.post("/analyze", json={"text": "  this is bad  "})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"sentiment", "probabilities", "baseline", "topic", "aspects",
                         "explanation", "truncated", "latency_ms"}
    assert body["topic"]["label"] == "t" and body["explanation"]["top_tokens"][0]["token"] == "bad"
    pos = client.post("/analyze", json={"text": "this is great", "explain": False}).json()
    assert pos["topic"] is None and pos["explanation"] is None  # topic only for negative reviews


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"text": "x" * 5001}, {}, {"text": 5}, {"texts": ["a"]},
])
def test_analyze_rejects_bad_input_with_422(client, payload):
    r = client.post("/analyze", json=payload)
    assert r.status_code == 422 and r.json()["detail"]


def test_analyze_accepts_exactly_the_character_limit(client):
    assert client.post("/analyze", json={"text": "a" * 5000}).status_code == 200


def test_batch_validates_size_and_each_text(client):
    ok = client.post("/analyze/batch", json={"texts": ["bad one", "good one"]})
    assert ok.status_code == 200 and ok.json()["n"] == 2
    assert ok.json()["results"][0]["explanation"] is None  # attributions are off by default
    assert client.post("/analyze/batch", json={"texts": []}).status_code == 422
    assert client.post("/analyze/batch", json={"texts": ["a"] * 33}).status_code == 422
    too_long = client.post("/analyze/batch", json={"texts": ["fine", "x" * 5001]})
    assert too_long.status_code == 422 and "5000" in too_long.text
    assert client.post("/analyze/batch", json={"texts": ["fine", "  "]}).status_code == 422


def test_unexpected_errors_become_a_500_without_details():
    class Broken(FakeService):
        def analyze(self, text, explain=True):
            raise RuntimeError("secret internals")

    with TestClient(create_app(Broken()), raise_server_exceptions=False) as c:
        r = c.post("/analyze", json={"text": "x"})
    assert r.status_code == 500 and "secret" not in r.text


@pytest.mark.skipif(not resolve_path("models/distilbert/model.safetensors").exists()
                    or not resolve_path("models/bertopic/topics.json").exists(),
                    reason="model artifacts not built")
def test_real_service_end_to_end():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("captum")
    with TestClient(create_app()) as c:  # loads the real models once
        r = c.post("/analyze", json={
            "text": "The controller stick broke after two weeks. Waste of money, terrible."})
        assert r.status_code == 200
        b = r.json()
        assert sum(b["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert b["sentiment"] == "negative" and b["topic"] is not None
        assert b["explanation"]["top_tokens"] and b["latency_ms"] > 0
        long = c.post("/analyze", json={"text": "great game " * 400, "explain": False}).json()
        assert long["truncated"] is True and long["topic"] is None
        assert c.get("/health").json()["status"] == "ok"
