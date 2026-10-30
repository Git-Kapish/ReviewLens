"""Phase 7a: monthly sentiment and topic shares, and themes flagged as rising.

Sentiment shares use the star-rating labels of all 60,000 reviews. Topic shares are among the
11,671 negative reviews. Topics are assigned with the saved BERTopic model (nearest topic
centroid, the same rule the API will use) so every month is assigned the same way; the fit-time
HDBSCAN labels are rerun as a sensitivity check.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from bertopic import BERTopic

from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import LABELS
from reviewlens.eval.metrics import write_metrics
from reviewlens.eval.plots import plot_trends
from reviewlens.topics.labels import load_labels
from reviewlens.topics.trends import episodes, flag_rising, monthly_counts, n_tests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_trends")

cfg = load_config("trends")
paths, rule = cfg["paths"], cfg["rising"]
proc = resolve_path(load_config("data")["paths"]["processed"])
reviews = pd.concat([pd.read_parquet(proc / f"{k}.parquet") for k in
                     ("train", "val", "test", "timehold")], ignore_index=True)
reviews["sentiment"] = reviews["label_id"].map(dict(enumerate(LABELS)))

negs = pd.read_parquet(resolve_path(paths["assignments"]))
emb = np.load(resolve_path(paths["embeddings"]), allow_pickle=False)["embeddings"]
model = BERTopic.load(str(resolve_path(paths["model_dir"])))
negs["topic_centroid"], _ = model.transform(negs["text"].tolist(), embeddings=emb)
real = sorted(t for t in set(negs["topic"]) if t != -1)
labels = load_labels(resolve_path(paths["labels"]), real)
name = {**labels, -1: "unassigned"}
agree = float((negs["topic_centroid"] == negs["topic"])[negs["topic"] != -1].mean())
log.info("nearest-centroid vs fit-time labels agree on %.1f%% of fit-assigned reviews", 100 * agree)


def run(df: pd.DataFrame, col: str, cats: list, min_total: int, only: list | None = None):
    """Counts, flags and episodes for one grouping column."""
    counts = monthly_counts(df, col, cats)
    flags = flag_rising(counts, min_total, rule["baseline_months"], rule["z_threshold"],
                        rule["min_rise"], rule["min_count"])
    if only is not None:
        flags = flags[flags["category"].isin(only)]
    return counts, flags, n_tests(counts, min_total, rule["baseline_months"])


# sentiment: is the negative share rising?
min_rev = cfg["sentiment"]["min_reviews_per_month"]
s_counts, s_flags, s_tests = run(reviews, "sentiment", LABELS, min_rev, only=["negative"])
by_year = (reviews.assign(year=pd.to_datetime(reviews["date"]).dt.year)
           .groupby("year")["sentiment"].value_counts(normalize=True).unstack().round(4))
counts_year = reviews.assign(year=pd.to_datetime(reviews["date"]).dt.year).groupby("year").size()

# topics among negatives
min_neg = cfg["topics"]["min_negatives_per_month"]
t_counts, t_flags, t_tests = run(negs, "topic_centroid", [-1, *real], min_neg, only=real)
f_counts, f_flags, f_tests = run(negs, "topic", [-1, *real], min_neg, only=real)
t_eps, f_eps = episodes(t_flags), episodes(f_flags)
for df in (t_flags, t_eps, f_flags, f_eps):
    if len(df):
        df["category"] = df["category"].map(name)

metrics = {
    "rule": rule, "n_months_sentiment": int((s_counts.sum(axis=1) >= min_rev).sum()),
    "sentiment_by_year": {int(y): {**by_year.loc[y].to_dict(), "n": int(counts_year[y])}
                          for y in by_year.index},
    "sentiment_flags_negative_rising": s_flags.to_dict("records"), "sentiment_tests": s_tests,
    "assignment": {"agreement_with_fit_labels": agree,
                   "outlier_share_centroid": float((negs["topic_centroid"] == -1).mean()),
                   "outlier_share_fit": float((negs["topic"] == -1).mean())},
    "topic_tests": t_tests,
    "expected_false_flags_if_binomial": t_tests * 0.00135,
    "topic_flags_centroid": t_flags.to_dict("records"),
    "topic_episodes_centroid": t_eps.to_dict("records"),
    "topic_flags_fit_labels": f_flags.to_dict("records"),
    "topic_episodes_fit_labels": f_eps.to_dict("records"),
    "topic_share_by_year_centroid": (
        negs.assign(year=pd.to_datetime(negs["date"]).dt.year,
                    name=negs["topic_centroid"].map(name))
        .query("year >= 2013").groupby("year")["name"].value_counts(normalize=True).unstack()
        .round(4).fillna(0).to_dict("index")),
}
write_metrics(paths["metrics"], metrics, cfg)

valid_s = s_counts.sum(axis=1) >= min_rev
valid_t = t_counts.sum(axis=1) >= min_neg
w = cfg["plot"]["smooth_months"]
s_share = s_counts.div(s_counts.sum(axis=1), axis=0).where(valid_s)
t_share = t_counts.rename(columns=name).pipe(lambda c: c.div(c.sum(axis=1), axis=0)).where(valid_t)
start = pd.Period(cfg["plot"]["start"], freq="M")
plot_trends(s_share.loc[start:].rolling(w, min_periods=1).mean(),
            t_share.loc[start:].rolling(w, min_periods=1).mean(),
            w, resolve_path(paths["figures_dir"]) / "trends.png")
log.info("sentiment: %d negative-share flags in %d tests", len(s_flags), s_tests)
log.info("topics (centroid): %d flagged months, %d episodes, %d tests; fit labels: %d months",
         len(t_flags), len(t_eps), t_tests, len(f_flags))
if len(t_eps):
    log.info("episodes:\n%s", t_eps.to_string())
