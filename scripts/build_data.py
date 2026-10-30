"""Phase 1: sample, clean, label, split, save parquet and the data profile."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reviewlens.config import load_config, resolve_path, set_seed
from reviewlens.data.clean import clean_reviews
from reviewlens.data.load import load_raw_sample
from reviewlens.data.profile import build_profile
from reviewlens.data.split import split_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("build_data")

COLS = ["text", "title", "rating", "label", "label_id", "timestamp", "date", "parent_asin",
        "user_id", "verified_purchase"]

cfg = load_config("data")
set_seed(cfg["seed"])
funnel: dict[str, int] = {}
df = clean_reviews(load_raw_sample(), cfg, funnel)
splits = split_dataset(df, cfg)

out_dir = resolve_path(cfg["paths"]["processed"])
out_dir.mkdir(parents=True, exist_ok=True)
for name, part in splits.items():
    part[COLS].to_parquet(out_dir / f"{name}.parquet", index=False)

profile = build_profile(splits, funnel, cfg["clean"]["short_tokens"])
profile = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "config": cfg} | profile
path = resolve_path(cfg["paths"]["reports"]) / "data_profile.json"
path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
log.info("saved %s and %s", out_dir, path)
