"""Error-analysis helpers: draw a reproducible sample of misclassified reviews and tally tags."""

from __future__ import annotations

import pandas as pd

from reviewlens.data.clean import LABELS


def sample_errors(
    df: pd.DataFrame, preds: dict[str, pd.Series], focus: str, n: int, seed: int
) -> pd.DataFrame:
    """Random `n` rows that model `focus` got wrong, with every model's prediction as a column.

    `df` needs `text` and `label_id`; `preds` maps model name to predicted label ids aligned
    with `df`. Rows keep their position in `df` as `row`, so tags can be joined back.
    """
    wrong = df.index[preds[focus].to_numpy() != df["label_id"].to_numpy()]
    picked = pd.Series(wrong).sample(n=min(n, len(wrong)), random_state=seed).sort_values()
    out = pd.DataFrame({"row": picked.to_numpy()})
    out["true"] = [LABELS[i] for i in df.loc[out["row"], "label_id"]]
    for name, p in preds.items():
        out[f"pred_{name}"] = [LABELS[i] for i in p.loc[out["row"]]]
    out["rating"] = df.loc[out["row"], "rating"].to_numpy()
    out["text"] = df.loc[out["row"], "text"].to_numpy()
    return out


def summarize_tags(tags: pd.DataFrame, allowed: list[str]) -> dict:
    """Counts and shares per failure tag; raises on a tag outside `allowed`."""
    bad = set(tags["tag"]) - set(allowed)
    if bad:
        raise ValueError(f"unknown tags: {sorted(bad)}")
    counts = tags["tag"].value_counts().reindex(allowed, fill_value=0)
    return {
        "n": int(len(tags)),
        "counts": {t: int(c) for t, c in counts.items()},
        "share": {t: round(float(c) / len(tags), 4) for t, c in counts.items()},
    }
