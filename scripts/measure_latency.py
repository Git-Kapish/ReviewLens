"""Phase 8: CPU latency of the analysis, in-process or against a running API.

Usage:  python scripts/measure_latency.py                      in-process service
        python scripts/measure_latency.py --url http://localhost:8000   over HTTP (e.g. Docker)
Times 50 random test reviews (seeded), after 3 warm-up calls, with and without explanations.
"""

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from reviewlens.config import load_config, resolve_path
from reviewlens.eval.metrics import write_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("measure_latency")

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="base URL of a running API; omit to time the service directly")
parser.add_argument("--n", type=int, default=50)
args = parser.parse_args()

test = pd.read_parquet(resolve_path(load_config("data")["paths"]["processed"]) / "test.parquet")
rng = np.random.default_rng(load_config("api").get("seed", 42))
texts = [t.strip()[:5000] for t in test["text"].iloc[rng.choice(len(test), args.n, replace=False)]]

if args.url:
    import requests

    def call(text: str, explain: bool) -> None:
        r = requests.post(f"{args.url.rstrip('/')}/analyze",
                          json={"text": text, "explain": explain}, timeout=120)
        r.raise_for_status()
    mode = "http"
else:
    from reviewlens.api.service import ReviewLensService

    service = ReviewLensService()

    def call(text: str, explain: bool) -> None:
        service.analyze(text, explain=explain)
    mode = "inprocess"


def run(explain: bool) -> dict:
    """Median and 95th percentile wall time in ms over the sampled reviews."""
    for t in texts[:3]:
        call(t, explain)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        call(t, explain)
        times.append(1000 * (time.perf_counter() - t0))
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(np.max(times)), "n": len(times)}


result = {"mode": mode, "cpu": platform.processor(), "threads": __import__("os").cpu_count(),
          "without_explanation": run(False), "with_explanation": run(True)}
write_metrics(f"reports/api_latency_{mode}.json", result, {"n_reviews": args.n, "url": args.url})
log.info("%s", {k: v for k, v in result.items() if k.endswith("explanation")})
