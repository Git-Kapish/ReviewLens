"""Phase 3: fine-tune DistilBERT, evaluate on val/test, save metrics, curves and predictions.

Usage:  python scripts/train_transformer.py            full run (GPU recommended)
        python scripts/train_transformer.py --smoke    tiny CPU run, outputs in models/
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path even without editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.eval.metrics import compute_metrics, write_metrics
from reviewlens.eval.plots import plot_curves
from reviewlens.models.predict import load_model, predict_proba
from reviewlens.models.transformer import train

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_transformer")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline")
args = parser.parse_args()

cfg = load_config("transformer")
proc = resolve_path(load_config("data")["paths"]["processed"])
splits = {k: pd.read_parquet(proc / f"{k}.parquet") for k in ("train", "val", "test", "timehold")}

if args.smoke:
    s = cfg["smoke"]
    cfg["epochs"] = s["epochs"]
    model_dir = resolve_path(s["model_dir"])
    splits["train"] = splits["train"].sample(s["train_rows"], random_state=cfg["seed"])
    for k in ("val", "test", "timehold"):
        splits[k] = splits[k].sample(s["val_rows"], random_state=cfg["seed"])
else:
    model_dir = resolve_path(cfg["paths"]["model_dir"])

result = train(cfg, splits["train"], splits["val"], model_dir)

# score the best epoch as reloaded from disk; the test set is scored once, after selection on val
model, tokenizer = load_model(model_dir)
probs = {
    k: predict_proba(model, tokenizer, v["text"].tolist(), cfg["max_len"],
                     batch_size=cfg["eval_batch_size"])
    for k, v in splits.items() if k != "train"
}
metrics = {
    "model": cfg["model_name"],
    **result,
    "val": compute_metrics(splits["val"]["label_id"], probs["val"].argmax(axis=1)),
    "test": compute_metrics(splits["test"]["label_id"], probs["test"].argmax(axis=1)),
}
log.info("val macro-F1 %.4f | test macro-F1 %.4f", metrics["val"]["macro_f1"],
         metrics["test"]["macro_f1"])

# a smoke run goes through the same writing code, but into the gitignored smoke model folder
out_root = model_dir / "reports" if args.smoke else resolve_path("reports")
rel = {k: Path(v).relative_to("reports") for k, v in cfg["paths"].items() if k != "model_dir"}
write_metrics(out_root / rel["metrics"], metrics, cfg)
plot_curves(result["history"], out_root / rel["curves"])
pred_dir = out_root / rel["predictions_dir"]
pred_dir.mkdir(parents=True, exist_ok=True)
for name in ("test", "timehold"):  # probabilities only; time hold-out is scored in Phase 4
    out = pd.DataFrame(probs[name], columns=[f"p_{c}" for c in LABELS])
    out.insert(0, "label_id", splits[name]["label_id"].to_numpy())
    out.insert(1, "pred", np.argmax(probs[name], axis=1))
    out.to_parquet(pred_dir / f"distilbert_{name}.parquet", index=False)
log.info("wrote outputs under %s", out_root)
