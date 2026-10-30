"""Paired bootstrap confidence intervals for macro-F1 and for the difference between two models."""

from __future__ import annotations

import numpy as np

from reviewlens.data.clean import LABELS


def _macro_f1_from_counts(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Macro-F1 from a flat confusion count (fast path for resampling loops)."""
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)))


def bootstrap_macro_f1(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile CIs on macro-F1 for each model and for every model's difference to the first.

    Every resample draws the same row indices for all models (paired), so the difference CI
    reflects the models disagreeing on the same reviews, not sampling noise on separate sets.
    Returns `{name: {point, ci_low, ci_high}}` plus `diff_<name>_minus_<first>` entries.
    """
    k = len(LABELS)
    y = np.asarray(y_true, dtype=int)
    preds = {name: np.asarray(p, dtype=int) for name, p in preds.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_resamples, len(y)))
    boot = {
        name: np.array([_macro_f1_from_counts(y[i], p[i], k) for i in idx])
        for name, p in preds.items()
    }
    names = list(preds)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def summarize(point: float, draws: np.ndarray) -> dict:
        return {
            "point": point,
            "ci_low": float(np.percentile(draws, lo)),
            "ci_high": float(np.percentile(draws, hi)),
        }

    point = {n: _macro_f1_from_counts(y, p, k) for n, p in preds.items()}
    out = {n: summarize(point[n], boot[n]) for n in names}
    for n in names[1:]:
        key = f"diff_{n}_minus_{names[0]}"
        out[key] = summarize(point[n] - point[names[0]], boot[n] - boot[names[0]])
        out[key]["share_resamples_above_zero"] = float(np.mean(boot[n] > boot[names[0]]))
    out["n_resamples"] = n_resamples
    out["n"] = int(len(y))
    return out
