"""Data profile: class balance, length distribution, date range, per split."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _lengths(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    out = {f"p{int(k * 100)}": float(v) for k, v in q.items()}
    return out | {"mean": round(float(s.mean()), 1)}


def profile_split(df: pd.DataFrame, short_tokens: int) -> dict[str, Any]:
    """Profile one split."""
    tokens = df["text"].str.split().str.len()
    return {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "class_share": df["label"].value_counts(normalize=True).round(4).to_dict(),
        "class_count": df["label"].value_counts().to_dict(),
        "star_share": df["rating"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "chars": _lengths(df["text"].str.len()),
        "tokens": _lengths(tokens),
        f"share_under_{short_tokens}_tokens": round(float((tokens < short_tokens).mean()), 4),
        "share_under_5_tokens": round(float((tokens < 5).mean()), 4),
        "verified_purchase_share": round(float(df["verified_purchase"].mean()), 4),
    }


def build_profile(
    splits: dict[str, pd.DataFrame], funnel: dict[str, int], short_tokens: int
) -> dict[str, Any]:
    """Profile of the whole cleaned set, each split, and the cleaning funnel."""
    full = pd.concat(splits.values(), ignore_index=True)
    return {
        "funnel": funnel,
        "overall": profile_split(full, short_tokens),
        "splits": {k: profile_split(v, short_tokens) for k, v in splits.items()},
        "timehold_starts": str(splits["timehold"]["date"].min()),
    }
