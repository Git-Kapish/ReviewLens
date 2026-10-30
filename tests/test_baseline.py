import numpy as np
import pytest

from reviewlens.config import load_config
from reviewlens.data.clean import LABELS
from reviewlens.eval.metrics import compute_metrics, macro_f1, write_metrics
from reviewlens.models.baseline import build_pipeline, select_baseline

pd = pytest.importorskip("pandas")


def test_metrics_perfect_and_confusion_layout():
    y = np.array([0, 0, 1, 2, 2, 2])
    m = compute_metrics(y, y)
    assert m["accuracy"] == m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    assert m["per_class"]["positive"]["support"] == 3


def test_macro_f1_penalizes_ignoring_a_class():
    y_true = np.array([0, 1, 2, 2, 2, 2])
    y_pred = np.array([2, 2, 2, 2, 2, 2])  # always the majority class
    assert macro_f1(y_true, y_pred) < 0.35
    assert compute_metrics(y_true, y_pred)["per_class"]["negative"]["recall"] == 0.0


def test_missing_class_does_not_crash():
    m = compute_metrics(np.array([0, 0, 2]), np.array([0, 2, 2]))
    assert m["per_class"]["neutral"]["support"] == 0


def test_write_metrics_adds_config_and_timestamp(tmp_path):
    path = write_metrics(tmp_path / "m.json", {"macro_f1": 0.5}, {"seed": 1})
    import json

    payload = json.loads(path.read_text())
    assert payload["config"] == {"seed": 1}
    assert payload["macro_f1"] == 0.5 and "timestamp" in payload


def _toy(n_per_class=20):
    words = {"negative": "terrible broken refund", "neutral": "okay average fine",
             "positive": "great love excellent"}
    rows = [{"text": f"{w} game number {i}", "label_id": LABELS.index(k)}
            for k, w in words.items() for i in range(n_per_class)]
    return pd.DataFrame(rows)


def test_baseline_learns_toy_data_and_vectorizer_sees_train_only():
    cfg = load_config("baseline")
    cfg["grid"] = {"use_char": [False, True], "C": [1.0]}
    train, val = _toy(), _toy(5)
    val.loc[0, "text"] = "zzzuniqueval token only in val"
    pipe, results = select_baseline(train, val, cfg)
    assert len(results) == 2 and max(r["val_macro_f1"] for r in results) > 0.9
    vocab = pipe["tfidf"].get_feature_names_out()
    assert not any("zzzuniqueval" in v for v in vocab)  # fit on train only
    proba = pipe.predict_proba(["terrible broken refund"])
    assert proba.shape == (1, 3) and proba.argmax() == LABELS.index("negative")


def test_pipeline_uses_balanced_class_weight():
    pipe = build_pipeline(load_config("baseline"), use_char=False, C=1.0)
    assert pipe["lr"].class_weight == "balanced"
