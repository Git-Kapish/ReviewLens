import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from reviewlens.config import load_config  # noqa: E402
from reviewlens.data.clean import LABELS  # noqa: E402
from reviewlens.models.predict import load_model, predict_proba  # noqa: E402
from reviewlens.models.transformer import class_weights, train  # noqa: E402


def test_class_weights_are_balanced_inverse_frequency():
    y = np.array([0] * 20 + [1] * 10 + [2] * 70)
    w = class_weights(y).numpy()
    assert w[1] > w[0] > w[2]  # rarer class, larger weight
    assert w == pytest.approx(100 / (3 * np.array([20, 10, 70])))
    # weighted class totals are equal, which is what "balanced" means
    assert (w * np.bincount(y)) == pytest.approx([100 / 3] * 3)


def test_class_weights_handle_missing_class():
    assert torch.isfinite(class_weights(np.array([0, 0, 2]))).all()


def _tiny_model():
    cfg = transformers.DistilBertConfig(
        vocab_size=30522, dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=len(LABELS)
    )
    return transformers.DistilBertForSequenceClassification(cfg)


def _toy(n=16):
    words = {0: "terrible broken refund", 1: "okay average fine", 2: "great love excellent"}
    return pd.DataFrame(
        [{"text": f"{w} number {i}", "label_id": y} for y, w in words.items() for i in range(n)]
    )


def test_train_saves_best_epoch_and_predictions_keep_input_order(tmp_path):
    cfg = load_config("transformer") | {"epochs": 2, "batch_size": 8, "lr": 5e-3}
    try:
        history = train(cfg, _toy(), _toy(6), tmp_path / "m", model=_tiny_model())
    except OSError:
        pytest.skip("tokenizer not available offline")
    assert [h["epoch"] for h in history["history"]] == [1, 2]
    assert history["best_epoch"] in (1, 2)
    assert all(np.isfinite(h["train_loss"]) for h in history["history"])

    model, tok = load_model(tmp_path / "m")
    texts = _toy(3)["text"].tolist()
    p = predict_proba(model, tok, texts, cfg["max_len"], batch_size=4)
    assert p.shape == (len(texts), 3) and p.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
    rev = predict_proba(model, tok, texts[::-1], cfg["max_len"], batch_size=4)
    assert rev[::-1] == pytest.approx(p, abs=1e-4)  # length-sorted batching restores order
