"""Phase 0 check: stream the head of each category, log its profile, save to reports/."""

import json
import logging
import sys
from datetime import UTC, datetime

from reviewlens.config import load_config, resolve_path
from reviewlens.data.load import describe, load_reviews

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("check_dataset")

cfg = load_config("data")
cats = sys.argv[1:] or [cfg["dataset"]["category"]]
out = {
    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    "config": cfg,
    "note": "profile of the first check.max_rows lines of each file, not the full category",
    "categories": {c: describe(load_reviews(c, cfg["check"]["max_rows"])) for c in cats},
}
path = resolve_path(cfg["paths"]["reports"]) / "phase0_dataset_check.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
log.info(json.dumps(out["categories"], indent=2))
log.info("saved %s", path)
