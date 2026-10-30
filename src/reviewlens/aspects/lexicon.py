"""Aspect lexicon: whole-word keyword patterns and seed phrases for the embedding map."""

from __future__ import annotations

import re
from typing import Any


def build_patterns(lexicon: dict[str, dict[str, Any]]) -> dict[str, re.Pattern[str]]:
    """One case-insensitive regex per aspect; whole words, plural -s/-es allowed."""
    patterns = {}
    for aspect, spec in lexicon.items():
        terms = sorted(spec["keywords"], key=len, reverse=True)  # longest first
        alt = "|".join(re.escape(t) for t in terms)
        patterns[aspect] = re.compile(rf"(?<!\w)(?:{alt})(?:s|es)?(?!\w)", re.IGNORECASE)
    return patterns


def keyword_aspects(sentence: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    """Aspects with at least one keyword in `sentence`, in lexicon order."""
    return [a for a, p in patterns.items() if p.search(sentence)]
