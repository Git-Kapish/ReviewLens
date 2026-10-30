"""Input perturbations for robustness checks. All are deterministic given the seed."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np


def add_typos(text: str, rate: float, rng: np.random.Generator) -> str:
    """Swap two adjacent inner letters in a `rate` share of the words that have 4+ letters."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            j = int(rng.integers(1, len(w) - 2))  # keep first and last letter in place
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def truncate(text: str, n_tokens: int) -> str:
    """Keep the first `n_tokens` whitespace tokens."""
    return " ".join(text.split()[:n_tokens])


def append_emoji(text: str, emojis: list[str], rng: np.random.Generator) -> str:
    """Append one or two sentiment-neutral emoji."""
    picks = rng.choice(emojis, size=int(rng.integers(1, 3)))
    return text + " " + " ".join(picks)


def insert_negation(text: str, words: list[str]) -> str | None:
    """Put 'not' before the first polar word, or return None if the review has none."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    m = pattern.search(text)
    return None if m is None else text[: m.start()] + "not " + text[m.start() :]


def build_perturbations(cfg: dict, seed: int) -> dict[str, Callable[[str], str | None]]:
    """Name -> function. A function returns None where the perturbation does not apply."""
    rng = np.random.default_rng(seed)
    return {
        "typos": lambda t: add_typos(t, cfg["typo_rate"], rng),
        "uppercase": lambda t: t.upper(),
        "lowercase": lambda t: t.lower(),
        "emoji": lambda t: append_emoji(t, cfg["emojis"], rng),
        "truncate": lambda t: truncate(t, cfg["truncate_tokens"]),
        "negation": lambda t: insert_negation(t, cfg["negation_words"]),
    }
