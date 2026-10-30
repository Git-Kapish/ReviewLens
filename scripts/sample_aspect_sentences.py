"""Phase 6: draw the sentences to hand-label for aspect evaluation (seeded, no keyword filter).

Dev sentences come from the validation split and tune the similarity threshold; eval sentences
come from the test split and are scored once. Writes CSVs with an empty `labels` column; fill it
as `aspect:sentiment|aspect:sentiment` (sentiment: negative, neutral, positive; empty = no aspect).
Refuses to overwrite a file that already has labels.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from reviewlens.aspects.extract import sample_sentences
from reviewlens.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sample_aspect_sentences")

cfg = load_config("aspects")
lab = cfg["labeled"]
proc = resolve_path(load_config("data")["paths"]["processed"])

for split, n_key, path_key in (("val", "dev_per_rating", "dev_labels"),
                               ("test", "eval_per_rating", "eval_labels")):
    out = resolve_path(cfg["paths"][path_key])
    if out.exists() and pd.read_csv(out)["labels"].notna().any():
        log.info("%s already has labels, not overwriting", out)
        continue
    df = pd.read_parquet(proc / f"{split}.parquet")
    s = sample_sentences(df, lab[n_key], cfg["seed"], lab["min_words"], lab["max_words"])
    s.insert(0, "id", [f"{split}-{i}" for i in range(len(s))])
    s.insert(1, "split", split)
    s["labels"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False, encoding="utf-8")
    log.info("wrote %d sentences to %s", len(s), out)
