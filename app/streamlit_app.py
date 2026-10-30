"""ReviewLens Streamlit app: Analyze, Themes & Trends, Model Comparison.

The Analyze page calls the FastAPI service at API_URL. The other two pages read the committed
result files in reports/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from reviewlens.explain.render import render_words

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REPORTS = Path(os.environ.get("REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports"))
MAX_CHARS = 5000
EXAMPLES = {
    "Mixed: good graphics, broken controller": (
        "The graphics are great but the controller stick broke after two weeks and customer "
        "service never answered. Waste of money."),
    "Positive: headset": (
        "Comfortable, lightweight and the sound quality is excellent. Easy to set up and "
        "great value for the price."),
    "Neutral-ish: short": "It does the job. Nothing special, arrived on time.",
}


@st.cache_data
def read_json(name: str) -> dict:
    """A JSON file from reports/."""
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def call_api(text: str, explain: bool) -> tuple[dict | None, str | None]:
    """POST /analyze; returns (response, error message)."""
    try:
        r = requests.post(f"{API_URL}/analyze", json={"text": text, "explain": explain},
                          timeout=120)
    except requests.RequestException as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"
    if r.status_code == 422:
        detail = r.json().get("detail", [])
        msg = "; ".join(d.get("msg", "invalid input") for d in detail) or "invalid input"
        return None, f"The API rejected the input: {msg}"
    if not r.ok:
        return None, f"API error {r.status_code}"
    return r.json(), None


def page_analyze() -> None:
    """Analyze a single review through the API."""
    st.header("Analyze a review")
    choice = st.selectbox("Example", ["(write your own)", *EXAMPLES], index=1)
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Review text", value=default, height=140, max_chars=MAX_CHARS,
                        key=f"text-{choice}")
    explain = st.checkbox("Show word-level explanation (slower)", value=True)
    if not st.button("Analyze", type="primary"):
        return
    if not text.strip():
        st.warning("Enter some text first.")
        return
    with st.spinner("Analyzing..."):
        res, err = call_api(text, explain)
    if err:
        st.error(err)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("DistilBERT", res["sentiment"], f"{res['probabilities'][res['sentiment']]:.0%}")
    c2.metric("TF-IDF + LR baseline", res["baseline"]["label"],
              f"{res['baseline']['probabilities'][res['baseline']['label']]:.0%}")
    c3.metric("Response time", f"{res['latency_ms']:.0f} ms")
    if res["truncated"]:
        st.warning("This review is longer than 128 word pieces; the model reads only the start.")
    st.bar_chart(pd.Series(res["probabilities"], name="probability"), height=200)

    if res["topic"]:
        t = res["topic"]
        st.subheader("Complaint theme")
        st.write(f"**{t['label']}** (topic {t['id']}, similarity {t['similarity']:.2f}). "
                 "This is the nearest of the nine topics, so a weak match is still labeled.")
    st.subheader("Aspects")
    if res["aspects"]:
        st.dataframe(pd.DataFrame(
            [{"aspect": a, "sentiment": v["sentiment"], "sentences": v["n_sentences"],
              **{f"p({k})": round(p, 2) for k, p in v["probabilities"].items()}}
             for a, v in res["aspects"].items()]), hide_index=True)
        st.caption("One sentiment is given per sentence, so a sentence with two opinions gets the "
                   "same label for both aspects.")
    else:
        st.write("No aspect keywords or close matches found.")
    if res["explanation"]:
        st.subheader("Which words drove the prediction")
        tokens = res["explanation"]["tokens"]
        html = render_words([t["token"] for t in tokens], [t["score"] for t in tokens])
        st.markdown(f"<div style='line-height:2'>{html}</div>", unsafe_allow_html=True)
        st.caption("Green supports the predicted class, red argues against it (Integrated "
                   "Gradients). Reliable on short reviews; noisy on long ones.")
        st.write("Top words: " + ", ".join(
            f"{t['token']} ({t['score']:+.2f})" for t in res["explanation"]["top_tokens"]))


def page_themes() -> None:
    """Topics, trends and aspect sentiment from the saved results."""
    st.header("Themes and trends")
    topics = read_json("topics.json")
    st.subheader("Complaint themes in negative reviews")
    rows = [{"topic": t["topic"], "theme (draft label)": t["label"], "reviews": t["n_total"],
             "mean rating": round(t["mean_rating"], 2), "top words": ", ".join(t["top_words"][:8])}
            for t in topics["topics"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"{topics['n_negative_reviews'] - topics['outliers_n']:,} of "
               f"{topics['n_negative_reviews']:,} negative reviews were assigned a topic when the "
               "model was fit; labels are drafts.")
    pick = st.selectbox("Example reviews for", [t["label"] for t in topics["topics"]])
    for t in topics["topics"]:
        if t["label"] == pick:
            for rep in t["representative"]:
                st.markdown(f"> {rep['text']}")

    st.subheader("Sentiment over time")
    monthly = pd.read_csv(REPORTS / "trends_monthly.csv", index_col="month")
    smooth = st.slider("Smoothing (months)", 1, 12, 6)
    sent = monthly[["share_negative", "share_neutral", "share_positive"]]
    st.line_chart(sent.rolling(smooth, min_periods=1).mean())
    st.subheader("Topic share among negative reviews")
    topic_cols = [c for c in monthly.columns if c.startswith("topic_share_")]
    share = monthly[topic_cols].rename(columns=lambda c: c.removeprefix("topic_share_"))
    st.area_chart(share.rolling(smooth, min_periods=1).mean())
    st.caption("The negative share of reviews rose from about 14% (2014) to 24% (2022), while "
               "complaints moved from game topics to hardware topics (D8 in docs/DECISIONS.md).")

    st.subheader("Themes flagged as rising")
    trends = read_json("trends.json")
    eps = pd.DataFrame(trends["topic_episodes_centroid"])
    if not eps.empty:
        st.dataframe(eps.round(3), hide_index=True)
    st.caption("Rule: share at least 3 binomial z-scores and 5 points above the previous six "
               "months. Most flags are single-month spikes, not sustained trends.")

    st.subheader("Aspect sentiment (test split)")
    aspects = pd.DataFrame(read_json("aspect_summary.json")["aspects"]).T
    st.bar_chart(aspects[["share_negative", "share_neutral", "share_positive"]])
    st.caption("Aspect detection has about 83% precision and 54% recall on a hand-labeled set of "
               "100 sentences, so treat small differences between aspects as noise.")


def page_models() -> None:
    """Baseline vs DistilBERT results with confidence intervals and robustness."""
    st.header("Model comparison")
    boot = read_json("eval_metrics.json")["bootstrap"]
    base, bert = read_json("baseline_metrics.json"), read_json("transformer_metrics.json")
    rows = []
    for split, label in (("test", "Test"), ("timehold", "Time hold-out")):
        for m, name in (("baseline", "TF-IDF + LR"), ("distilbert", "DistilBERT")):
            b = boot[split][m]
            rows.append({"split": label, "model": name, "macro-F1": round(b["point"], 4),
                         "95% CI": f"{b['ci_low']:.4f} to {b['ci_high']:.4f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    d = boot["test"]["diff_distilbert_minus_baseline"]
    st.write(f"Difference on the test split: **{d['point']:+.4f}** macro-F1 "
             f"(95% CI {d['ci_low']:+.4f} to {d['ci_high']:+.4f}, paired bootstrap, "
             "1,000 resamples). "
             "Each model was trained once, so seed variation is not included.")

    st.subheader("Per-class F1 on the test split")
    classes = base["test"]["per_class"]
    st.dataframe(pd.DataFrame({
        "class": list(classes),
        "TF-IDF + LR": [round(classes[c]["f1"], 3) for c in classes],
        "DistilBERT": [round(bert["test"]["per_class"][c]["f1"], 3) for c in classes],
        "support": [classes[c]["support"] for c in classes]}), hide_index=True)
    st.caption("Neutral (3-star) is the weak class for both models; accuracy overstates quality "
               "because 73% of reviews are positive.")

    st.subheader("Robustness: change in macro-F1 after perturbing the test reviews")
    rb = {m: read_json(f"robustness_{m}.json")["perturbations"] for m in ("baseline", "distilbert")}
    st.dataframe(pd.DataFrame({
        "perturbation": list(rb["baseline"]),
        "TF-IDF + LR": [round(rb["baseline"][p]["delta"]["point"], 4) for p in rb["baseline"]],
        "DistilBERT": [round(rb["distilbert"][p]["delta"]["point"], 4) for p in rb["baseline"]]}),
        hide_index=True)
    st.caption("Case and emoji changes are zero by construction (both models lowercase; the "
               "baseline "
               "drops emoji). Negation insertion does not preserve the label, so it measures "
               "sensitivity rather than accuracy.")
    st.image(str(REPORTS / "figures" / "robustness.png"))


PAGES = {"Analyze": page_analyze, "Themes & Trends": page_themes, "Model Comparison": page_models}


def main() -> None:
    """Page router."""
    st.set_page_config(page_title="ReviewLens", layout="wide")
    st.sidebar.title("ReviewLens")
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption(f"API: {API_URL}")
    PAGES[page]()


main()
