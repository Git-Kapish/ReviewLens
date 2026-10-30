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
- Head-of-file stats differ from the full file. Phase 1 on a uniform sample of the whole file gives 19.5% negative (head of file: 14.1%) and a date range ending 2023-09-05 (head: 2023-03-21). The head-of-file table above is kept as the record of what drove the category choice; `docs/RESULTS.md` has the real numbers.
- The HF loader script for this dataset is rejected by recent `datasets` versions, so `data/load.py` streams the raw `.jsonl` through `huggingface_hub` instead.

## D2. Phase 1 data pipeline choices

- **Sampling:** uniform reservoir sample of 75,000 raw lines over the whole file (seed 42), cached under `data/raw/`; cleaned and capped to 60,000. Sampling before cleaning means duplicates are only found within the sample, not across the full 4.6M reviews.
- **Text field:** the model input is the review `text`; `title` is kept as a column but not used. Literal HTML tags (`<br />`) are replaced by spaces and whitespace is collapsed. Nothing else is changed for the transformer.
- **Deduplication:** exact match on lowercased text, keeping the earliest review. This removed 7.3% of non-empty reviews. Many of them are short generic reviews ("Great", "Fun game") that different people wrote independently, so dedup also thins the shortest, noisiest reviews. Kept because the PRD asks for dedup, and the effect is documented rather than hidden.
- **Language filter:** `langdetect` (seeded) on the first 500 characters when the text is at least 30 characters. Shorter texts are too short to identify, so they are kept unless most of their letters are non-ASCII. Short non-English text written in Latin letters will get through.
- **Short reviews are kept** (20.6% are under 10 tokens). They are a real part of the data and the robustness phase tests truncation.
- **TF-IDF normalization** lives in `clean.normalize_for_tfidf` (lowercase, URL token, squeeze repeated characters) and is used only by the baseline. Negation words are not removed.
- **Time-based hold-out:** the latest 10% by timestamp, removed before the random split, so train/val/test never contain data from the hold-out window.
- **Dependencies added:** `scikit-learn` and `langdetect` (both light). `matplotlib`, `ipykernel`, `nbconvert` and `nbformat` are in an optional `notebooks` extra.

## D3. Phase 2 baseline choices

- **Normalization lives inside the vectorizer** (`preprocessor=normalize_for_tfidf`), so the saved pipeline takes raw review text and the API can use it directly.
- **No refit on train+val.** The selected pipeline is the one trained on train only, so the validation score stays an honest estimate and nothing is tuned on test. Refitting would add data at the cost of a model that was never scored as trained.
- **Grid is small on purpose** (char n-grams on/off x 3 values of C) to keep CPU time reasonable. Word features alone won, so the char features are not used in the saved model.
- **Test set:** the script scores it once per run, after selection. Rerunning `make baseline` reproduces the same numbers (fixed seed, deterministic solver), so it is not a second look with changed choices.
- **Metrics JSON:** `eval/metrics.write_metrics` adds the config and a UTC timestamp; later models reuse it.
