"""Phase 6: score the hand-labeled eval sentences once, then aggregate aspects on the test split.

Run after tuning the threshold on the dev set (scripts/tune_aspects.py) and freezing the config.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from reviewlens.aspects.score import AspectAnalyzer, aggregate_aspects
from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.eval.aspects import evaluate, load_gold
from reviewlens.eval.metrics import write_metrics
from reviewlens.eval.plots import plot_aspect_sentiment
from reviewlens.models.predict import load_model, predict_proba

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_aspects")

cfg = load_config("aspects")
tcfg = load_config("transformer")
paths = cfg["paths"]
encoder = SentenceTransformer(cfg["embedding"]["model_name"])
model, tok = load_model(resolve_path(tcfg["paths"]["model_dir"]))
# our own artifact from `make baseline`, not untrusted input
baseline_dir = resolve_path(load_config("baseline")["paths"]["model_dir"])
baseline = joblib.load(baseline_dir / "baseline.joblib")


def distilbert_proba(sentences: list[str]) -> np.ndarray:
    """Sentence-level DistilBERT probabilities."""
    return predict_proba(model, tok, sentences, tcfg["max_len"], batch_size=128)


gold = load_gold(resolve_path(paths["eval_labels"]))
sentences, gold_pairs = gold["text"].tolist(), gold["gold"].tolist()

analyzers = {
    "keyword_only": AspectAnalyzer(cfg, encoder, distilbert_proba, use_embedding=False),
    "keyword_plus_embedding": AspectAnalyzer(cfg, encoder, distilbert_proba),
}
sent_probs = {"distilbert": distilbert_proba(sentences),
              "baseline": baseline.predict_proba(sentences)}

metrics: dict = {"threshold": cfg["similarity"]["threshold"], "n_sentences": len(gold),
                 "variants": {}}
for name, an in analyzers.items():
    pred_aspects = an.map_aspects(sentences)
    for sm, probs in sent_probs.items():
        pred_sent = [LABELS[i] for i in probs.argmax(axis=1)]
        metrics["variants"][f"{name}/{sm}"] = evaluate(gold_pairs, pred_aspects, pred_sent)
        e = metrics["variants"][f"{name}/{sm}"]
        log.info("%s/%s: detection F1 %.3f | sentiment on detected %.3f (n=%d) | end-to-end %.3f",
                 name, sm, e["detection"]["f1"], e["sentiment_on_detected"]["accuracy"],
                 e["sentiment_on_detected"]["n"], e["sentiment_end_to_end"]["accuracy"])

# every miss, false alarm and wrong sentiment of the main configuration, for the failure analysis
main_aspects = analyzers["keyword_plus_embedding"].map_aspects(sentences)
main_sent = [LABELS[i] for i in sent_probs["distilbert"].argmax(axis=1)]
cases = []
for (_, row), pa, ps in zip(gold.iterrows(), main_aspects, main_sent, strict=True):
    g = row["gold"]
    issues = ([f"missed:{a}" for a in g if a not in pa] + [f"extra:{a}" for a in pa if a not in g]
              + [f"sentiment:{a} gold {g[a]} pred {ps}" for a in g if a in pa and g[a] != ps])
    if issues:
        cases.append({"id": row["id"], "text": row["text"], "gold": g, "pred_aspects": pa,
                      "pred_sentiment": ps, "issues": issues})
metrics["main_configuration"] = "keyword_plus_embedding/distilbert"
metrics["cases_with_issues"] = cases
write_metrics(paths["metrics"], metrics, cfg)

# dataset-level aspect sentiment on the test split (descriptive; accuracy is bounded by the above)
split = pd.read_parquet(resolve_path(load_config("data")["paths"]["processed"])
                        / f"{cfg['aggregate_split']}.parquet")
an = analyzers["keyword_plus_embedding"]
per_review = an.analyze_reviews(split["text"].tolist())
table = aggregate_aspects(per_review)
by_label = {}
for lab_id, name in enumerate(LABELS):
    sub = [r for r, y in zip(per_review, split["label_id"], strict=True) if y == lab_id]
    by_label[name] = aggregate_aspects(sub)
summary = {"split": cfg["aggregate_split"], "n_reviews": len(split),
           "n_reviews_with_any_aspect": int(sum(bool(r) for r in per_review)),
           "aspects": table, "by_review_label": by_label}
write_metrics(paths["summary"], summary, cfg)
plot_aspect_sentiment(table, resolve_path(paths["figures_dir"]) / "aspect_sentiment.png")
log.info("aggregated %d reviews; %d mention at least one aspect",
         len(split), summary["n_reviews_with_any_aspect"])
