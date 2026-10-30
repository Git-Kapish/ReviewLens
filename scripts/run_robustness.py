"""Phase 4 (slow part): robustness of one model to input perturbations on the test set.

Usage:  python scripts/run_robustness.py --model baseline     seconds on CPU
        python scripts/run_robustness.py --model distilbert   needs a GPU for speed (Colab)

Writes reports/robustness_<model>.json. Perturbed texts depend only on the seed, so the two
runs score exactly the same texts. Only rows whose text changed are re-scored.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.eval.bootstrap import bootstrap_macro_f1
from reviewlens.eval.metrics import write_metrics
from reviewlens.eval.robustness import build_perturbations

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_robustness")

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["baseline", "distilbert"], required=True)
args = parser.parse_args()

cfg = load_config("eval")
rb, b = cfg["robustness"], cfg["bootstrap"]
test = pd.read_parquet(resolve_path(load_config("data")["paths"]["processed"]) / "test.parquet")
texts, y = test["text"].tolist(), test["label_id"].to_numpy()

if args.model == "baseline":
    import joblib

    # our own artifact from `make baseline`, not untrusted input
    baseline_dir = resolve_path(load_config("baseline")["paths"]["model_dir"])
    pipe = joblib.load(baseline_dir / "baseline.joblib")

    def predict(batch: list[str]) -> np.ndarray:
        """Baseline class predictions."""
        return np.asarray(pipe.predict(batch))
else:
    from reviewlens.models.predict import get_device, load_model, predict_proba

    tcfg = load_config("transformer")
    model, tok = load_model(resolve_path(tcfg["paths"]["model_dir"]))
    log.info("DistilBERT on %s", get_device())

    def predict(batch: list[str]) -> np.ndarray:
        """DistilBERT class predictions."""
        probs = predict_proba(model, tok, batch, tcfg["max_len"],
                              batch_size=rb["eval_batch_size"])
        return probs.argmax(axis=1)

# clean predictions are recomputed here so clean and perturbed runs share hardware and precision
clean = predict(texts)
results: dict = {"model": args.model, "perturbations": {}, "examples": {}}
if args.model == "distilbert":  # sanity check against the Phase 3 predictions (Colab GPU)
    saved = pd.read_parquet(
        resolve_path(cfg["paths"]["predictions_dir"]) / "distilbert_test.parquet")["pred"]
    results["clean_agreement_with_phase3_predictions"] = float((clean == saved.to_numpy()).mean())
    log.info("agreement with Phase 3 predictions: %.4f",
             results["clean_agreement_with_phase3_predictions"])

for name, fn in build_perturbations(rb, cfg["seed"]).items():
    perturbed = [fn(t) for t in texts]
    applies = np.array([p is not None for p in perturbed])
    changed = np.array([p is not None and p != t for p, t in zip(perturbed, texts, strict=True)])
    idx = np.flatnonzero(changed)
    log.info("%s: applies to %d, changes %d of %d", name, applies.sum(), changed.sum(), len(texts))
    after = clean.copy()
    if len(idx):
        after[idx] = predict([perturbed[i] for i in idx])
    # negation applies only to reviews with a polar word; the others use every review
    boot = bootstrap_macro_f1(
        y[applies], {"clean": clean[applies], "perturbed": after[applies]},
        b["n_resamples"], b["alpha"], cfg["seed"],
    )
    c, a = clean[applies], after[applies]
    flipped = ((c == 2) & (a == 0)) | ((c == 0) & (a == 2))  # positive <-> negative
    results["examples"][name] = [{"before": texts[i], "after": perturbed[i]} for i in idx[:3]]
    results["perturbations"][name] = {
        "n_applicable": int(applies.sum()),
        "n_changed": int(changed.sum()),
        "clean_macro_f1": boot["clean"]["point"],
        "perturbed_macro_f1": boot["perturbed"]["point"],
        "delta": boot["diff_perturbed_minus_clean"],
        "share_predictions_changed": float((c != a).mean()),
        "share_polarity_flipped": float(flipped.mean()),
    }
    log.info("  %.4f -> %.4f", boot["clean"]["point"], boot["perturbed"]["point"])

write_metrics(cfg["paths"]["robustness"].format(model=args.model), results, cfg)
