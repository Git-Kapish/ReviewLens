"""Top n-gram coefficients of the TF-IDF + logistic-regression baseline."""

from __future__ import annotations

import numpy as np

from reviewlens.data.clean import LABELS


def top_ngrams(pipe, n: int) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Per class, the `n` features with the largest and the smallest coefficient.

    A large positive coefficient raises that class's score, a large negative one lowers it (the
    other classes compete through the softmax). Feature names have their vectorizer prefix removed.
    """
    names = np.array([f.split("__", 1)[-1] for f in pipe["tfidf"].get_feature_names_out()])
    coef = pipe["lr"].coef_
    out = {}
    for k, label in enumerate(LABELS):
        order = np.argsort(coef[k])
        out[label] = {
            "positive": [(str(names[i]), float(coef[k, i])) for i in order[::-1][:n]],
            "negative": [(str(names[i]), float(coef[k, i])) for i in order[:n]],
        }
    return out
