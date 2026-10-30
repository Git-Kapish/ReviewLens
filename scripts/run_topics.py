"""Phase 5: BERTopic on negative reviews.

Fits on the train-split negatives only, assigns topics to the negatives of the other splits
(val, test, time hold-out), scores c_v coherence, saves the model and the assignments.
Topic labels are written by hand to reports/topic_labels.json and picked up on the next run.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.eval.metrics import write_metrics
from reviewlens.eval.plots import plot_topic_sizes
from reviewlens.topics.bertopic_runner import (
    coherence_cv,
    embed_texts,
    fit_topics,
    representative_docs,
    select_negatives,
    topic_words,
)
from reviewlens.topics.labels import load_labels

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_topics")

cfg = load_config("topics")
paths = cfg["paths"]
proc = resolve_path(load_config("data")["paths"]["processed"])
splits = {k: pd.read_parquet(proc / f"{k}.parquet") for k in ("train", "val", "test", "timehold")}

neg = select_negatives(splits)
log.info("negative reviews per split: %s", neg["split"].value_counts().to_dict())
emb = embed_texts(neg["text"].tolist(), cfg, resolve_path(paths["embeddings"]))

# fit on train negatives only; the other splits are assigned afterwards
fit_idx = np.flatnonzero(neg["split"] == "train")[: cfg["fit"]["max_docs"]]
fit_docs = neg["text"].iloc[fit_idx].tolist()

grid, models = [], {}
for mcs in cfg["grid"]["min_cluster_size"]:
    model, topics = fit_topics(fit_docs, emb[fit_idx], cfg, mcs)
    topics = np.array(topics)
    real = sorted(set(topics.tolist()) - {-1})
    coh = coherence_cv(model, fit_docs, cfg)
    grid.append({
        "min_cluster_size": mcs, "n_topics": len(real),
        "outlier_share": float((topics == -1).mean()), "coherence_cv": coh["mean"],
    })
    models[mcs] = (model, topics)
    log.info("min_cluster_size=%d: %s", mcs, grid[-1])

# selection rule (set before the run, see docs/DECISIONS.md): highest c_v among the settings
# whose final topic count is in the PRD range of 8-12; if none qualify, stop and look.
ok = [g for g in grid if 8 <= g["n_topics"] <= 12]
if not ok:
    sys.exit(f"no grid setting gave 8-12 topics: {grid}")
best = max(ok, key=lambda g: g["coherence_cv"])
model, fit_topics_arr = models[best["min_cluster_size"]]
log.info("selected %s", best)

neg["topic"] = -1
neg.loc[neg.index[fit_idx], "topic"] = fit_topics_arr
rest = np.flatnonzero(neg["split"] != "train")
rest_topics, _ = model.transform(neg["text"].iloc[rest].tolist(), embeddings=emb[rest])
neg.loc[neg.index[rest], "topic"] = rest_topics

coh = coherence_cv(model, fit_docs, cfg)
reps = representative_docs(emb[fit_idx], fit_topics_arr, cfg["representative_docs"])
real = sorted(t for t in set(neg["topic"]) if t != -1)
labels_path = resolve_path(paths["labels"])
labels = load_labels(labels_path, real) if labels_path.exists() else {}

topic_rows = []
for t in real:
    mask = neg["topic"] == t
    topic_rows.append({
        "topic": t,
        "label": labels.get(t),
        "n_total": int(mask.sum()),
        "n_by_split": neg.loc[mask, "split"].value_counts().to_dict(),
        "mean_rating": float(neg.loc[mask, "rating"].mean()),
        "coherence_cv": coh["per_topic"][t],
        "top_words": topic_words(model, t, cfg["top_n_words"]),
        "representative": [
            {"doc_id": neg["doc_id"].iloc[fit_idx[i]], "text": neg["text"].iloc[fit_idx[i]][:400]}
            for i in reps[t]
        ],
    })
split_stats = {
    s: {"n": int(len(g)), "outlier_share": float((g["topic"] == -1).mean())}
    for s, g in neg.groupby("split")
}
metrics = {
    "n_negative_reviews": int(len(neg)), "n_fit_docs": int(len(fit_idx)),
    "grid": grid, "selected": best, "coherence_cv_mean": coh["mean"],
    "by_split": split_stats, "topics": topic_rows,
    "outliers_n": int((neg["topic"] == -1).sum()),
}
write_metrics(paths["metrics"], metrics, cfg)

neg.to_parquet(resolve_path(paths["assignments"]), index=False)
model_dir = resolve_path(paths["model_dir"])
model_dir.mkdir(parents=True, exist_ok=True)
model.save(str(model_dir), serialization="safetensors", save_ctfidf=True,
           save_embedding_model=cfg["embedding"]["model_name"])
sizes = {(labels.get(r["topic"]) or f"topic {r['topic']}"): r["n_total"] for r in topic_rows}
fig_dir = resolve_path(paths["figures_dir"])
plot_topic_sizes(sizes, metrics["outliers_n"], fig_dir / "topic_sizes.png")
log.info("%d topics, c_v %.4f, outliers %.1f%% of fit docs", len(real), coh["mean"],
         100 * best["outlier_share"])
