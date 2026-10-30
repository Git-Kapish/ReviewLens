# Decisions

## D1. Dataset: Amazon Reviews 2023, `Video_Games` (Phase 0)

**Choice:** `McAuley-Lab/Amazon-Reviews-2023`, category `Video_Games`, subsampled to ~60K reviews in Phase 1.
**Fallback:** `yelp_review_full` (no timestamps; trend view becomes topic share by rating).

### What was verified
Source: `reports/phase0_dataset_check.json`, the first 200,000 lines of each category file streamed from the Hub.
This is a head-of-file profile, not the full category; full-category numbers come from Phase 1.

| | Video_Games | All_Beauty |
|---|---|---|
| File size (Hub) | 2.68 GB | 0.33 GB |
| Date range (head 200K) | 1999-11-13 to 2023-03-21 | 2001-01-30 to 2023-04-03 |
| Negative (1-2 stars) | 14.1% | 17.5% |
| Neutral (3 stars) | 8.0% | 8.9% |
| Positive (4-5 stars) | 77.9% | 73.6% |
| Median review length | 134 chars | 111 chars |
| Empty text | 0.02% | 0.04% |

Fields present: `rating`, `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`, `timestamp` (ms), `helpful_vote`, `verified_purchase`.
`rating`, `text` and `timestamp` cover everything the PRD needs, including the trend view.

### Why Video_Games
- It is the PRD's first example, and the date range spans over 20 years, enough for monthly trends and a held-out latest window.
- Complaint themes (bugs/crashes, price, activation/DRM, controls, shipping/packaging) map well onto the planned aspect lexicon.
- Both categories are about equally imbalanced (~74-78% positive, 8-9% neutral), so neither is easier on class balance. At 60K reviews, neutral is roughly 4.8K, enough to train and evaluate on.
- Cost: the file is 2.68 GB against 0.33 GB for All_Beauty. Phase 1 will stream it once and reservoir-sample ~60K rather than keep it on disk. If that turns out to be a problem, All_Beauty is the swap (needs your OK per CLAUDE.md).

### License
The HF dataset card has no `license` field and no license text. The dataset is published by McAuley Lab (UCSD) for research and recommendation work.
Treat it as research-use: no raw data or derived text is committed (`data/` is gitignored), and the README will cite the authors.
Verify the terms at the project page before any redistribution.

### Open caveats
- Head-of-file stats may differ from the full file; the real class balance and date range are recomputed in Phase 1.
- The HF loader script for this dataset is rejected by recent `datasets` versions, so `data/load.py` streams the raw `.jsonl` through `huggingface_hub` instead.
