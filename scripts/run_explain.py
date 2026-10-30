"""Phase 7b: explainability. LR top n-grams, a ~10-example DistilBERT attribution gallery, and a
deletion check on whether the attributions track the model's behavior."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.eval.metrics import write_metrics
from reviewlens.eval.plots import plot_lr_ngrams
from reviewlens.explain.lr_coefs import top_ngrams
from reviewlens.explain.render import render_gallery
from reviewlens.explain.token_attr import deletion_check, explain_text, top_words
from reviewlens.models.predict import load_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_explain")

cfg = load_config("explain")
paths, seed = cfg["paths"], cfg["seed"]
tcfg, ecfg = load_config("transformer"), load_config("eval")
test = pd.read_parquet(resolve_path(load_config("data")["paths"]["processed"]) / "test.parquet")

# 1. logistic regression: top n-gram coefficients per class
# our own artifact from `make baseline`, not untrusted input
baseline_dir = resolve_path(load_config("baseline")["paths"]["model_dir"])
top = top_ngrams(joblib.load(baseline_dir / "baseline.joblib"), cfg["lr"]["top_n"])
out = resolve_path(paths["lr_json"])
out.write_text(json.dumps(top, indent=2), encoding="utf-8")
plot_lr_ngrams(top, cfg["lr"]["plot_n"], resolve_path(paths["lr_figure"]))
for cls in LABELS:
    log.info("LR toward %s: %s", cls, ", ".join(f for f, _ in top[cls]["positive"][:8]))

# 2. DistilBERT attributions for a static gallery
model, tok = load_model(resolve_path(tcfg["paths"]["model_dir"]))
pred = pd.read_parquet(resolve_path(ecfg["paths"]["predictions_dir"]) / "distilbert_test.parquet")
short = test["text"].str.split().str.len() <= cfg["gallery"]["max_words"]
rng = np.random.default_rng(seed)
picks: list[tuple[int, str]] = []
for cls_id, cls in enumerate(LABELS):
    ok = np.flatnonzero((pred["label_id"] == cls_id) & (pred["pred"] == cls_id) & short)
    for i in rng.choice(ok, size=cfg["gallery"]["n_correct_per_class"], replace=False):
        picks.append((int(i), f"correct {cls}"))
errors = pd.read_csv(resolve_path(ecfg["paths"]["error_sample"]))
tags = pd.read_csv(resolve_path(ecfg["paths"]["error_tags"])).set_index("row")["tag"]
errors = errors[errors["row"].map(short)].sample(frac=1, random_state=seed)
for row in errors["row"].head(cfg["gallery"]["n_errors"]):
    picks.append((int(row), f"error, {tags[row]}"))

gallery = []
for row, kind in picks:
    r = explain_text(model, tok, test["text"].iloc[row], tcfg["max_len"],
                     n_steps=cfg["attribution"]["n_steps"])
    r.update(id=f"test-{row}", true=LABELS[int(test["label_id"].iloc[row])], tag=kind)
    gallery.append(r)
    log.info("%s | true %s pred %s | top: %s", r["id"], r["true"], LABELS[r["pred"]],
             [w for w, _ in top_words(r, 4)])
gallery_out = [
    {"id": g["id"], "kind": g["tag"], "true": g["true"], "pred": LABELS[g["pred"]],
     "probs": g["probs"], "words": g["words"], "scores": np.round(g["scores"], 4).tolist(),
     "convergence_delta": g["delta"]}
    for g in gallery
]
resolve_path(paths["gallery_json"]).write_text(json.dumps(gallery_out, indent=1), encoding="utf-8")
resolve_path(paths["gallery_html"]).write_text(
    render_gallery(gallery, "DistilBERT token attributions (test reviews)"), encoding="utf-8")

# 3. does removing the top-attributed words change the prediction more than random words?
sample = rng.choice(len(test), size=cfg["deletion"]["n_reviews"], replace=False)
results = [explain_text(model, tok, test["text"].iloc[int(i)], tcfg["max_len"],
                        n_steps=cfg["attribution"]["n_steps"]) for i in sample]
dcfg = cfg["deletion"]
check = deletion_check(model, results, dcfg["k_words"], dcfg["n_random"], seed)
deltas = [g["delta"] for g in gallery] + [r["delta"] for r in results]
check["convergence_delta_mean"] = float(np.mean(deltas))
check["convergence_delta_max"] = float(np.max(deltas))
write_metrics(paths["metrics"], {"deletion_check": check, "n_gallery": len(gallery)}, cfg)
log.info("deletion check: %s", check)
