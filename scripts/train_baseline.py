"""Phase 2: select and train the TF-IDF + LR baseline, evaluate on val and test."""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd

from reviewlens.config import load_config, resolve_path, set_seed
from reviewlens.eval.metrics import compute_metrics, write_metrics
from reviewlens.models.baseline import select_baseline

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_baseline")

cfg = load_config("baseline")
data_cfg = load_config("data")
set_seed(cfg["seed"])

proc = resolve_path(data_cfg["paths"]["processed"])
train, val, test = (pd.read_parquet(proc / f"{k}.parquet") for k in ("train", "val", "test"))

pipe, grid_results = select_baseline(train, val, cfg)
best = max(grid_results, key=lambda r: r["val_macro_f1"])
log.info("selected %s", best)

model_dir = resolve_path(cfg["paths"]["model_dir"])
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_dir / "baseline.joblib")

# the test set is scored once, after selection on val
metrics = {
    "model": "tfidf_logreg",
    "selected": {"use_char": best["use_char"], "C": best["C"]},
    "grid": grid_results,
    "val": compute_metrics(val["label_id"], pipe.predict(val["text"])),
    "test": compute_metrics(test["label_id"], pipe.predict(test["text"])),
}
write_metrics(cfg["paths"]["metrics"], metrics, cfg)
log.info("test macro-F1 = %.4f", metrics["test"]["macro_f1"])
