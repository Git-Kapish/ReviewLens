"""Stratified train/val/test split plus a time-based hold-out of the latest window."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Return train/val/test/timehold frames.

    The latest `time_holdout_frac` of reviews (by timestamp) become `timehold` and are kept out
    of the other three. The rest is split stratified by label with the config seed.
    """
    s, seed = cfg["split"], cfg["seed"]
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * (1 - s["time_holdout_frac"]))
    rest, timehold = df.iloc[:cut], df.iloc[cut:]

    held = s["val_frac"] + s["test_frac"]
    train, temp = train_test_split(rest, test_size=held, stratify=rest["label"], random_state=seed)
    val, test = train_test_split(
        temp, test_size=s["test_frac"] / held, stratify=temp["label"], random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "timehold": timehold.reset_index(drop=True),
    }
