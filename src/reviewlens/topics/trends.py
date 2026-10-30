"""Monthly shares and a simple rule that flags rising shares (sentiment or topics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_counts(
    df: pd.DataFrame, group_col: str, categories: list, date_col: str = "date"
) -> pd.DataFrame:
    """Counts per calendar month (rows, including empty months) and category (columns)."""
    month = pd.to_datetime(df[date_col]).dt.to_period("M")
    counts = pd.crosstab(month, df[group_col]).reindex(columns=categories, fill_value=0)
    full = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    return counts.reindex(full, fill_value=0)


def flag_rising(
    counts: pd.DataFrame, min_total: int, baseline_months: int, z_threshold: float,
    min_rise: float, min_count: int,
) -> pd.DataFrame:
    """Flag (month, category) pairs whose share jumped above the trailing baseline.

    For month t, the baseline share p0 is the pooled share of the previous `baseline_months`
    months. The month is flagged when the binomial z-score
    `(p - p0) / sqrt(p0 (1 - p0) / n)` is at least `z_threshold`, the absolute rise `p - p0` is at
    least `min_rise`, and the category has at least `min_count` reviews that month. Months (and
    baselines) where any month has fewer than `min_total` reviews are skipped.
    """
    totals = counts.sum(axis=1)
    valid = totals >= min_total
    rows = []
    for i in range(baseline_months, len(counts)):
        month = counts.index[i]
        window = slice(i - baseline_months, i)
        if not valid.iloc[i] or not valid.iloc[window].all():
            continue
        n, n0 = totals.iloc[i], totals.iloc[window].sum()
        for cat in counts.columns:
            k, k0 = counts[cat].iloc[i], counts[cat].iloc[window].sum()
            p, p0 = k / n, k0 / n0
            if p0 <= 0 or p0 >= 1:
                continue
            z = (p - p0) / np.sqrt(p0 * (1 - p0) / n)
            if z >= z_threshold and p - p0 >= min_rise and k >= min_count:
                rows.append({"month": str(month), "category": cat, "count": int(k), "n": int(n),
                             "share": float(p), "baseline_share": float(p0), "z": float(z)})
    return pd.DataFrame(rows, columns=["month", "category", "count", "n", "share",
                                       "baseline_share", "z"])


def n_tests(counts: pd.DataFrame, min_total: int, baseline_months: int) -> int:
    """How many (month, category) pairs `flag_rising` can test, for the false-alarm estimate."""
    valid = counts.sum(axis=1) >= min_total
    ok = sum(
        valid.iloc[i] and valid.iloc[i - baseline_months:i].all()
        for i in range(baseline_months, len(counts))
    )
    return int(ok * counts.shape[1])


def episodes(flags: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive flagged months of one category into episodes (first, last, peak z)."""
    out = []
    for cat, g in flags.groupby("category"):
        months = pd.PeriodIndex(g["month"], freq="M")
        start = 0
        for j in range(1, len(g) + 1):
            if j == len(g) or months[j].ordinal - months[j - 1].ordinal != 1:
                seg = g.iloc[start:j]
                out.append({"category": cat, "first_month": seg["month"].iloc[0],
                            "last_month": seg["month"].iloc[-1], "n_months": j - start,
                            "peak_z": float(seg["z"].max()),
                            "peak_share": float(seg["share"].max()),
                            "baseline_share": float(seg["baseline_share"].iloc[0])})
                start = j
    return pd.DataFrame(out)
