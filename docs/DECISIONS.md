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

## D4. Phase 3 transformer choices

- **Plain PyTorch loop, no `Trainer` and no `accelerate`.** A class-weighted loss, per-epoch train/val curves and best-epoch saving are about 80 lines of loop, and the weighted loss would need a `Trainer` subclass anyway. `accelerate` from the PRD stack list is therefore not a dependency. `torch` and `transformers` are an optional extra (`pip install -e ".[transformer]"`).
- **Python range widened to `>=3.11,<3.13`.** Colab runs 3.12, and the earlier `<3.12` cap would stop `pip install -e .` there. The project still targets and is tested on 3.11.
- **Settings:** `distilbert-base-uncased`, max_len 128, 3 epochs, batch 32, AdamW lr 2e-5, weight decay 0.01, 10% linear warmup, grad-clip 1.0, fp16 only on CUDA, seed 42. These are the PRD ranges and common defaults; there is no hyperparameter search, so the transformer gets no more tuning effort than the one-shot defaults.
- **Class-weighted loss:** inverse-frequency weights from the train split (`n / (3 * count)`), the same idea as `class_weight="balanced"` in the baseline. The same weights are used for the validation loss, so the two curves are comparable.
- **Model selection:** best epoch by validation macro-F1 (the primary metric), saved at that epoch. The final model is reloaded from disk before it is scored, so the scored model is the saved artifact. Validation and test are scored once, after training.
- **Full train split, no subsample.** 43,200 reviews fit a free T4 comfortably, so the subsample fallback is only used by `--smoke`.
- **Predictions saved for later:** the script writes class probabilities for the test set and the time hold-out to `reports/predictions/`, so Phase 4 can run bootstrap CIs and the drift check on CPU without the weights. Only the test metrics are computed in Phase 3; the time hold-out is scored in Phase 4.
- **Weights are not committed** (`models/` is gitignored). The way to get them is to run `notebooks/02_train_distilbert_colab.ipynb` (or `make transformer` on a GPU machine).
- **Checks done on CPU:** unit tests with a tiny random-weight DistilBERT (training loop, best-epoch saving, input-order-preserving batched inference) and a `--smoke` run of the real script on 256 reviews. These show the pipeline runs; they say nothing about model quality.

## D5. Phase 4 evaluation choices

- **Bootstrap:** 1,000 paired resamples, percentile 95% intervals, seed 42. Pairing means both models see the same resampled reviews, so the CI on the difference is the one that answers "is DistilBERT better". Seed variance of training is not covered (one run per model).
- **Error sample:** 60 random errors of DistilBERT on the test split (it is the final model). The baseline's predictions for the same rows are in the sample file. Tags were assigned by me reading each review, with rules written in `docs/ERROR_ANALYSIS.md`. There was one tagger, so counts are a judgement, and a second reader should spot-check `reports/error_tags.csv`.
- **Robustness on the test set:** the perturbations are fixed rules with no tuning, so they do not use the test set to choose anything. Perturbed texts depend only on the seed, so the baseline (local) and DistilBERT (Colab) were scored on identical texts. Parameters are in `configs/eval.yaml`: 15% typo rate, truncation to 8 whitespace tokens (the PRD asks for fewer than 10), two neutral emoji appended, upper and lower case.
- **Negation insertion** is the one perturbation that does not preserve the label. I report macro-F1 on the 3,659 reviews that contain a listed polar word, plus the share of predictions that changed and that flipped between positive and negative, and say in the results that it measures sensitivity, not accuracy.
- **Only changed rows are re-scored.** Rows whose text is unchanged reuse the clean prediction. Clean predictions are recomputed in the same run (and checked against the Phase 3 file for DistilBERT) so hardware differences do not show up as perturbation effects.
- **Robustness is run per model** (`scripts/run_robustness.py --model ...`) because DistilBERT inference on the full set takes too long on this CPU; the DistilBERT run is in `notebooks/02b_robustness_colab.ipynb`. `make eval` runs both and then rebuilds the summary figure.

## D6. Phase 5 topic model choices

- **Which reviews:** "negative" means the star-rating label (1-2 stars), not the model's prediction. All 11,671 negative reviews from every split are embedded and given a topic. The model is fit on the 8,193 train negatives only; val, test and the time hold-out negatives are assigned with `transform`, so no held-out review shapes the topics. The API in Phase 8 will apply the same `transform` to reviews the sentiment model predicts as negative. No document cap was needed (the cap in `configs/topics.yaml` is 20,000).
- **Embeddings:** `all-MiniLM-L6-v2` (normalized, max 256 word pieces), cached in `data/processed/negative_embeddings.npz` with a hash of the texts and model, so a changed input recomputes instead of silently reusing stale vectors.
- **Pipeline:** UMAP (15 neighbours, 5 dimensions, cosine, seed 42) then HDBSCAN (min_samples 10, `eom`), topic words from c-TF-IDF on a 1-2-gram CountVectorizer with English stop words. `nr_topics=10` merges down to 10 topics counting the outlier group, which gives 9 real topics here. PRD range is 8-12.
- **Choosing `min_cluster_size` (rule fixed before the run):** grid of 20, 40 and 80, pick the highest c_v coherence among settings with 8-12 final topics. Result: 80 had the highest c_v (0.566) but only 7 topics, so it was excluded; 40 was selected (0.534, 9 topics); 20 gave 0.432. I did not change the rule after seeing this. Setting 80 also leaves only 4.8% outliers, so a coarser model with fewer, broader topics was available and was not used.
- **Outliers stay outliers (topic -1).** Not reassigned to the nearest topic, because that would label reviews that fit no cluster.
- **Coherence:** c_v from gensim over the top 10 unigram topic words, computed on the same train negatives the model was fit on (not held out), single process. Bigram topic words are excluded from this score because c_v needs words present in the tokenized corpus.
- **Labels:** drafted by the assistant from the top words, three representative reviews (closest to the topic centroid) and five random reviews per topic, stored in `reports/topic_labels.json`. The file says its status. They have not been checked by the project owner yet, and `load_labels` fails if the labels and topics stop matching after a refit.
- **Dependencies added** (approved): optional `topics` extra with `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`, `gensim`. The fitted model is saved to `models/bertopic/` (gitignored); topic assignments for every negative review to `data/processed/negatives_topics.parquet` (gitignored, rebuilt by `make topics`) for the Phase 7 trends.

## D7. Phase 6 aspect sentiment choices

- **Aspects (13), adapted to video games and peripherals:** gameplay, graphics, story, sound, controls, durability, price_value, battery, comfort_fit, delivery_packaging, support_returns, compatibility_setup, performance_bugs. The PRD's examples (delivery, quality, price, support, usability, packaging) are covered; hardware and game aspects are added because this category mixes both. The lexicon and seed phrases are in `configs/aspects.yaml` and were written from domain knowledge and the Phase 5 topic words, before I looked at any labeled sentence.
- **Mapping:** sentences are split on sentence punctuation followed by a capital letter and on line breaks (no extra dependency). A sentence gets an aspect if a keyword matches as a whole word (plural -s/-es allowed), or if its MiniLM embedding has cosine similarity of at least 0.50 to the aspect's seed-phrase centroid (at most two embedding aspects per sentence). Sentences are not split on "but", so one sentence with two opinions gets one sentiment for all its aspects. That is a known limit, not something I fixed.
- **Sentiment:** the Phase 3 DistilBERT (the baseline is scored alongside for comparison) applied to each sentence. It was trained on whole reviews, not sentences. The label for an aspect in a review is the argmax of the mean probabilities over that review's sentences that mention the aspect.
- **Labeled data, in two parts so the threshold is not tuned on the evaluation set:** 40 dev sentences from the validation split (8 per star rating, drawn at random with no keyword filter) tune the similarity threshold; 100 eval sentences from the test split (20 per rating) are scored once. Each sentence has zero or more `aspect:sentiment` labels. Order of work: lexicon and code frozen and committed (47d0c5d), eval sentences labeled before the system was run on them, then one run. I did not change the lexicon after seeing eval results, although the results suggest changes (see RESULTS.md).
- **Threshold rule (fixed before the run):** highest micro-F1 of aspect detection on the dev set, ties to the higher value. Result: 0.50 (dev F1 0.667; keyword only 0.635), scanning 0.30 to 0.75.
- **The labeler is also the lexicon author.** I (the assistant) wrote the lexicon and labeled all 140 sentences, so the gold labels are consistent with my own notion of the aspects and probably flatter the system a little. Some labels are judgement calls (for example a power-cord fault counted as `battery`, an overall opinion about a game counted as no aspect). The project owner should spot-check them; the files are `data/labeled/aspect_dev.csv` and `aspect_eval.csv` (committed; `.gitignore` lets `data/labeled/` through).
- **Dataset-level aggregation** is computed on the test split only (5,400 reviews), as a descriptive table; running it on all 60,000 reviews would take much longer on this CPU without adding evaluation value.

## D8. Phase 7 trend and explanation choices

- **Sentiment trend uses the star-rating labels** of all 60,000 reviews, by calendar month, because model predictions exist only for the test and time hold-out splits. Months with fewer than 100 reviews (before 2013, plus a few gaps) are not tested; the figure starts in 2014, when topic counts per month become readable.
- **Topic trend uses the saved BERTopic model's nearest-centroid assignment for every negative review**, so all months are assigned the same way. The fit-time labels (HDBSCAN) are not usable for this: train reviews carry fit-time labels while held-out ones carry `approximate_predict` labels, and they have different outlier rates. A saved BERTopic model (safetensors) keeps only the topic centroids, not UMAP or HDBSCAN, so `model.transform` after loading is a nearest-centroid lookup. Against the fit-time labels it agrees on 83.4% of the reviews that were assigned a topic, and it leaves 7.7% in the outlier topic instead of 30.1%. **The Phase 8 API will use this same lookup**, so the per-review topic shown there is a nearest-topic assignment, not a cluster-membership test; this is why trends use it. The flagging is also run on the fit-time labels as a sensitivity check.
- **Rising-theme rule (set before looking at results):** for each month t and category, take the pooled share p0 over the previous 6 months (all 6 must have enough reviews: 100 for sentiment, 40 negatives for topics). Flag t when `z = (p - p0) / sqrt(p0 (1 - p0) / n) >= 3`, the rise `p - p0 >= 0.05`, and the category has at least 5 reviews in t. The binomial z accounts for small monthly counts; the extra thresholds keep tiny absolute changes out. Consecutive flagged months are merged into one episode. With about 1,020 topic tests, z >= 3 alone would flag about 1.4 months by chance if counts were pure binomial noise; real month-to-month variation makes it more.
- **Not a change-point detector.** The rule compares each month with the recent past, so it finds spikes and starts of rises, not the date of a lasting shift, and it does not correct for seasonality.
- **LR explanations** are the largest coefficients per class of the saved baseline (vectorizer prefix stripped). TF-IDF scaling means coefficients are comparable within a class but not across classes' scales.
- **Token attributions:** Integrated Gradients (Captum `LayerIntegratedGradients`, 30 steps) on DistilBERT's word-embedding layer, baseline = every non-special token replaced by `[PAD]`, target = the predicted class, word-piece scores summed into words. Captum is a new optional dependency (`explain` extra). I chose Captum because it works directly on the PyTorch model's embedding layer; I did not benchmark SHAP.
- **Gallery:** 10 test reviews of at most 60 words: one correct review per class (random, seed 42) and 7 of the Phase 4 error-sample reviews (so each error carries its hand-tagged failure type). It is a static HTML file, `reports/figures/explanations.html`.
- **A faithfulness check is reported next to the gallery:** on 100 random test reviews, delete the 3 top-attributed words and, separately, 3 random words, and compare the drop in the predicted class's probability. Five reviews with three words or fewer are skipped.

## D9. Phase 8 serving choices

- **Service layer is separate from FastAPI** (`api/service.py` loads models once and analyzes; `api/main.py` only does HTTP and validation). `create_app(service)` lets the tests inject a fake service, so the HTTP contract is tested without loading models, and one end-to-end test loads the real models when they exist.
- **Models load once in the lifespan handler.** Startup takes about 10 seconds on this machine (in Docker the health check allows 60 seconds before it counts failures).
- **Endpoints:** `POST /analyze` (label, probabilities, baseline's prediction, topic, aspects, explanation, `truncated`, latency), `POST /analyze/batch` (1 to 32 reviews; explanations off by default because they dominate the cost), `GET /health`. Schemas are in `api/schemas.py`.
- **Validation:** text is stripped, must be non-empty and at most 5,000 characters; a batch holds 1 to 32 texts, each checked the same way. Failures are 422 with FastAPI's per-field messages (the batch message names the offending index and the limit). Unexpected exceptions return a plain 500 with no internals.
- **Topic only for reviews predicted negative**, as the PRD asks. Topic is the nearest of the 9 saved centroids by cosine similarity, never "unassigned" (agreed with the project owner); the similarity is returned so a weak match is visible. The centroid lookup reproduces `BERTopic.transform` exactly on the 11,671 negative reviews (100% agreement, checked before writing the service), and it does not need `bertopic`, `umap` or `hdbscan` at serve time, which keeps the API image smaller. Because the lookup always picks a topic, a negative review about something the nine topics do not cover still gets one.
- **Attributions** use Integrated Gradients with 20 steps (30 in the Phase 7 gallery) to cut latency. A lock serializes explanations, since Captum hooks the embedding layer.
- **`truncated` flag:** the model reads at most 128 word pieces; the response says when a review was longer.
- **The Streamlit app calls the API** for the Analyze page (`API_URL`, default `http://localhost:8000`; compose sets `http://api:8000`). There is no in-process fallback. Themes & Trends and Model Comparison read the committed result files in `reports/`, so they work without the API.
- **Docker:** one Dockerfile, two targets (`api`, `app`). Model weights are not in the image; `./models` is mounted read-only at `/app/models` (agreed). The image uses CPU-only torch and downloads the public MiniLM encoder at build time. `docker-compose up` starts the API, waits for its health check, then starts the app. `reports/` is copied into both images, so a rebuild is needed to show new results.
- **New dependencies** (all in optional extras): `api` (fastapi, uvicorn, plus the model libraries), `app` (streamlit, requests), and `httpx` in `dev` for the test client.

## D10. Phase 9 wrap-up choices

- **CI:** GitHub Actions runs `ruff check .` and `pytest -q` on Python 3.11 with CPU torch and the `dev`, `api`, `app` and `explain` extras. Tests that need trained models skip themselves. Verified locally, not on GitHub: a fresh clone installed with those extras passed lint and gave 68 passed, 1 skipped (the real-model API test), in 4.5 minutes, most of it model and tokenizer downloads.
- **README numbers are tested.** `tests/test_readme_numbers.py` recomputes the headline numbers from `reports/*.json` and asserts that the README contains them, so a changed result file fails the tests until the README is updated.
- **Not done:** the full pipeline was not re-run from a clean clone (the data stream takes about 10 minutes and DistilBERT training needs a GPU); individual steps were run during their phases. App screenshots were not captured because the browser tool available to me could not open localhost, so the README has no screenshots. No LICENSE file was added; that is the repository owner's choice.
- **No timeline claims:** all commits are from a single day, so the README says nothing about how long the project took.
