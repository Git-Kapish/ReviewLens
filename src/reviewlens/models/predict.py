"""Inference for the fine-tuned transformer: device choice, loading, batched probabilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """`cuda` if available, else `cpu`."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str | Path, device: torch.device | None = None):
    """Load a saved model and tokenizer, moved to `device` and set to eval mode."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return model, tokenizer


@torch.no_grad()
def predict_logits(
    model,
    tokenizer,
    texts: list[str],
    max_len: int,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Logits of shape (len(texts), n_classes), in the order of `texts`.

    Texts are batched by length to cut padding, then put back in input order.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out[idx] = model(**enc).logits.float().cpu().numpy()
    model.train(was_training)
    return out


def predict_proba(model, tokenizer, texts: list[str], max_len: int, **kwargs) -> np.ndarray:
    """Class probabilities, shape (len(texts), n_classes)."""
    logits = predict_logits(model, tokenizer, texts, max_len, **kwargs)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
