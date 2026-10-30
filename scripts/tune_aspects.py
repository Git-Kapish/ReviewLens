"""Phase 6: choose the embedding-similarity threshold on the 40 dev sentences (validation split).

Rule, fixed before the run: highest micro-F1 of aspect detection (keyword + embedding), ties go to
the higher threshold. The eval sentences are not read here. Set the result in configs/aspects.yaml.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from reviewlens.aspects.score import AspectAnalyzer
from reviewlens.config import load_config, resolve_path
from reviewlens.eval.aspects import evaluate, load_gold
from reviewlens.eval.metrics import write_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("tune_aspects")

from sentence_transformers import SentenceTransformer  # noqa: E402

cfg = load_config("aspects")
gold = load_gold(resolve_path(cfg["paths"]["dev_labels"]))
sentences = gold["text"].tolist()
encoder = SentenceTransformer(cfg["embedding"]["model_name"])
no_sentiment = lambda ss: np.zeros((len(ss), 3))  # noqa: E731  (detection only here)

results = {}
kw = AspectAnalyzer(cfg, encoder, no_sentiment, use_embedding=False)
results["keyword_only"] = evaluate(
    gold["gold"].tolist(), kw.map_aspects(sentences), ["negative"] * len(sentences))["detection"]
log.info("keyword only: %s", {k: round(v, 3) for k, v in results["keyword_only"].items()})

grid = {}
for tau in np.round(np.arange(0.30, 0.76, 0.05), 2):
    an = AspectAnalyzer(cfg, encoder, no_sentiment, threshold=float(tau))
    d = evaluate(gold["gold"].tolist(), an.map_aspects(sentences),
                 ["negative"] * len(sentences))["detection"]
    grid[float(tau)] = d
    log.info("tau=%.2f: P %.3f R %.3f F1 %.3f", tau, d["precision"], d["recall"], d["f1"])

best_f1 = max(d["f1"] for d in grid.values())
best_tau = max(t for t, d in grid.items() if d["f1"] == best_f1)
log.info("selected threshold %.2f (dev F1 %.3f)", best_tau, best_f1)
write_metrics(cfg["paths"]["dev_report"], {
    "n_sentences": len(gold), "keyword_only": results["keyword_only"],
    "threshold_grid": {str(t): d for t, d in grid.items()}, "selected_threshold": best_tau,
}, cfg)
