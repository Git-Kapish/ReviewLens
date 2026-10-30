"""Phase 4 (fast part): bootstrap CIs, time hold-out drift, error sample and tag summary.

Needs `make baseline` and the DistilBERT predictions from `make transformer` (no model is run).
Robustness is in scripts/run_robustness.py because it runs DistilBERT inference.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.eval.bootstrap import bootstrap_macro_f1
from reviewlens.eval.errors import sample_errors, summarize_tags
from reviewlens.eval.metrics import compute_metrics, write_metrics
from reviewlens.eval.plots import plot_ci_comparison, plot_robustness

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_eval")

cfg = load_config("eval")
proc = resolve_path(load_config("data")["paths"]["processed"])
pred_dir = resolve_path(cfg["paths"]["predictions_dir"])
# our own artifact from `make baseline`, not untrusted input
baseline_dir = resolve_path(load_config("baseline")["paths"]["model_dir"])
baseline = joblib.load(baseline_dir / "baseline.joblib")
b = cfg["bootstrap"]

splits = {k: pd.read_parquet(proc / f"{k}.parquet") for k in ("test", "timehold")}
preds: dict[str, dict[str, pd.Series]] = {}
for name, df in splits.items():
    probs = baseline.predict_proba(df["text"])
    out = pd.DataFrame(probs, columns=[f"p_{c}" for c in LABELS])
    out.insert(0, "label_id", df["label_id"].to_numpy())
    out.insert(1, "pred", probs.argmax(axis=1))
    out.to_parquet(pred_dir / f"baseline_{name}.parquet", index=False)
    d = pd.read_parquet(pred_dir / f"distilbert_{name}.parquet")
    assert (d["label_id"] == df["label_id"]).all(), "prediction rows misaligned"
    preds[name] = {"baseline": out["pred"], "distilbert": d["pred"]}

metrics: dict = {"bootstrap": {}, "per_split": {}, "drift": {}}
for name, df in splits.items():
    y = df["label_id"].to_numpy()
    metrics["bootstrap"][name] = bootstrap_macro_f1(
        y, {m: p.to_numpy() for m, p in preds[name].items()},
        b["n_resamples"], b["alpha"], cfg["seed"],
    )
    metrics["per_split"][name] = {
        m: compute_metrics(y, p.to_numpy()) for m, p in preds[name].items()
    } | {"class_share": {c: float((y == i).mean()) for i, c in enumerate(LABELS)},
         "date_min": str(df["date"].min()), "date_max": str(df["date"].max())}
    for m, ci in metrics["bootstrap"][name].items():
        if isinstance(ci, dict):
            log.info("%s %s: %s", name, m, {k: round(v, 4) for k, v in ci.items()})

# drift: same model on the time hold-out vs the random test split (different reviews, so unpaired)
for m in ("baseline", "distilbert"):
    t, h = (metrics["bootstrap"][s][m] for s in ("test", "timehold"))
    metrics["drift"][m] = {
        "test_macro_f1": t["point"], "timehold_macro_f1": h["point"],
        "change": h["point"] - t["point"],
        "cis_overlap": bool(h["ci_low"] <= t["ci_high"] and t["ci_low"] <= h["ci_high"]),
    }
write_metrics(cfg["paths"]["metrics"], metrics, cfg)
fig_dir = resolve_path(cfg["paths"]["figures_dir"])
plot_ci_comparison(metrics["bootstrap"], fig_dir / "bootstrap_ci.png")

# robustness figure, once both models have been run (see scripts/run_robustness.py)
rob_files = {m: resolve_path(cfg["paths"]["robustness"].format(model=m))
             for m in ("baseline", "distilbert")}
if all(p.exists() for p in rob_files.values()):
    rob = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in rob_files.items()}
    plot_robustness({m: {k: v["delta"] for k, v in r["perturbations"].items()}
                     for m, r in rob.items()}, fig_dir / "robustness.png")

# error sample: random misclassified test reviews from DistilBERT, hand-tagged afterwards
sample = sample_errors(splits["test"], preds["test"], "distilbert", cfg["errors"]["n_sample"],
                       cfg["seed"])
sample.to_csv(resolve_path(cfg["paths"]["error_sample"]), index=False, encoding="utf-8")
n_wrong = int((preds["test"]["distilbert"] != splits["test"]["label_id"]).sum())
log.info("error sample: %d of %d DistilBERT test errors", len(sample), n_wrong)

tags_path = resolve_path(cfg["paths"]["error_tags"])
if tags_path.exists():
    tags = pd.read_csv(tags_path)
    assert set(tags["row"]) == set(sample["row"]), "tags do not match the current sample"
    summary = summarize_tags(tags, cfg["errors"]["tags"]) | {
        "n_distilbert_test_errors": n_wrong, "sample_seed": cfg["seed"],
        "baseline_also_wrong": int(
            (sample["pred_baseline"] != sample["true"]).sum()),
    }
    write_metrics(cfg["paths"]["error_summary"], summary, cfg)
    log.info("tag counts %s", summary["counts"])
else:
    log.info("no tags file at %s yet; tag the sample, then rerun", tags_path)
