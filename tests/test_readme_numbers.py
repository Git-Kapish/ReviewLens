"""Every headline number in the README must match the metrics files in reports/."""

import json

import pytest

from reviewlens.config import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
REPORTS = PROJECT_ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def ci(d: dict, sign: bool = False) -> str:
    point = f"{d['point']:+.4f}" if sign else f"{d['point']:.4f}"
    return f"{point} ({d['ci_low']:.4f} to {d['ci_high']:.4f})"


def test_bootstrap_table_matches_eval_metrics():
    boot = load("eval_metrics.json")["bootstrap"]
    for split in ("test", "timehold"):
        for key in ("baseline", "distilbert"):
            assert ci(boot[split][key]) in README
        assert ci(boot[split]["diff_distilbert_minus_baseline"], sign=True) in README


def test_accuracy_and_per_class_f1_match():
    base, bert = load("baseline_metrics.json")["test"], load("transformer_metrics.json")["test"]
    assert f"{base['accuracy']:.4f} for the baseline and {bert['accuracy']:.4f}" in README
    for c in ("negative", "neutral", "positive"):
        row = f"| {c} | {base['per_class'][c]['f1']:.4f} | {bert['per_class'][c]['f1']:.4f} |"
        assert row in README


def test_error_counts_and_topic_sizes_match():
    summary = load("error_summary.json")
    assert summary["n_distilbert_test_errors"] == 840 and "840 of the 5,400" in README
    for tag, n in summary["counts"].items():
        assert f"| {n} |" in README, tag
    for t in load("topics.json")["topics"]:
        assert f"| {t['label']} | {t['n_total']:,} |" in README


def test_aspect_results_match():
    v = load("aspects_metrics.json")["variants"]["keyword_plus_embedding/distilbert"]
    d, s, e = v["detection"], v["sentiment_on_detected"], v["sentiment_end_to_end"]
    assert f"{d['precision']:.3f} / {d['recall']:.3f} / {d['f1']:.3f}" in README
    for m in (s, e):
        lo, hi = m["ci95"]
        assert f"{m['accuracy']:.3f} (95% CI {lo:.3f} to {hi:.3f}, {m['n']} pairs)" in README
    support = load("aspect_summary.json")["aspects"]["support_returns"]
    assert f"{100 * support['share_negative']:.1f}% of reviews" in README
    assert f"{100 * support['share_positive']:.1f}% positive" in README


@pytest.mark.parametrize("pert", ["typos", "truncate", "negation"])
def test_robustness_deltas_match(pert):
    for model in ("baseline", "distilbert"):
        delta = load(f"robustness_{model}.json")["perturbations"][pert]["delta"]["point"]
        assert f"{delta:+.4f}" in README


def test_data_shares_trends_and_latency_match():
    profile = load("data_profile.json")["overall"]
    assert f"{profile['rows']:,}" in README
    for cls in ("positive", "negative", "neutral"):
        share = 100 * profile["class_count"][cls] / profile["rows"]  # counts, not the rounded share
        assert f"{share:.1f}% {cls}" in README
    years = load("trends.json")["sentiment_by_year"]
    assert f"{100 * years['2014']['negative']:.1f}%" in README
    assert f"{100 * years['2022']['negative']:.1f}%" in README
    lat = load("api_latency_http.json")
    assert f"{lat['without_explanation']['median_ms']:.0f} ms" in README
    assert f"{lat['with_explanation']['p95_ms']:,.0f} ms" in README
    deletion = load("explain_metrics.json")["deletion_check"]
    assert f"{100 * deletion['share_top_beats_random']:.1f}% of {deletion['n_reviews']}" in README
