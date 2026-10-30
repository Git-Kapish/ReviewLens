# Results

All numbers come from files in `reports/`. Re-run `make data` to regenerate.

## Phase 1: data (`reports/data_profile.json`)

Sample: 75,000 reviews drawn uniformly (seed 42) from all 4,624,615 `Video_Games` reviews, then cleaned.

### Cleaning funnel

| Step | Rows | Dropped |
|---|---|---|
| Raw sample | 75,000 | |
| Non-empty text | 74,947 | 53 |
| Exact duplicates removed (case-insensitive) | 69,491 | 5,456 |
| English only | 68,583 | 908 |
| Capped to target (random) | 60,000 | 8,583 |

### Splits

The latest 10% of reviews by timestamp (from 2022-04-24) are the time-based hold-out. The rest is split 80/10/10, stratified by label.

| Split | Rows | Date range | Negative | Neutral | Positive |
|---|---|---|---|---|---|
| train | 43,200 | 1999-11-05 to 2022-04-24 | 19.0% | 7.8% | 73.3% |
| val | 5,400 | 1999-12-27 to 2022-04-23 | 19.0% | 7.7% | 73.3% |
| test | 5,400 | 1999-11-28 to 2022-04-24 | 19.0% | 7.8% | 73.3% |
| timehold | 6,000 | 2022-04-24 to 2023-09-05 | 23.8% | 7.7% | 68.5% |
| all | 60,000 | 1999-11-05 to 2023-09-05 | 19.5% | 7.7% | 72.8% |

Class counts overall: 11,671 negative, 4,644 neutral, 43,685 positive.

### Other facts

- Review length: median 27 tokens, p95 219, p99 552 (whitespace split). 20.6% of reviews are under 10 tokens and 8.0% under 5.
- 85.4% of reviews are verified purchases.
- Negative share is higher in the time hold-out (23.8% vs 19.0%), so the later window is already different from train before any model is run. Phase 4 will report drift on it.

## Phase 2: TF-IDF + logistic regression baseline (`reports/baseline_metrics.json`)

Pipeline: word 1-2 gram TF-IDF (200K features, `min_df=2`, sublinear TF) + `LogisticRegression(class_weight="balanced")`, fit on train only. Selected on validation macro-F1 from a 6-point grid; the test set was scored once for the selected model.

### Grid (validation macro-F1)

| Char n-grams | C=0.25 | C=1 | C=4 |
|---|---|---|---|
| no | 0.6437 | **0.6587** | 0.6543 |
| yes (3-5, `char_wb`) | 0.6441 | 0.6528 | 0.6464 |

Selected: word features only, C=1. Char n-grams did not help and cost about 3x the fit time (170-210 s vs 40-60 s per fit).

### Test set (5,400 reviews)

| Metric | Value |
|---|---|
| Accuracy | 0.8130 |
| **Macro-F1 (reference number)** | **0.6584** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.7058 | 0.7920 | 0.7464 | 1,024 |
| neutral | 0.2687 | 0.4200 | 0.3277 | 419 |
| positive | 0.9463 | 0.8600 | 0.9011 | 3,957 |

Confusion matrix (rows = true, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | 811 | 129 | 84 |
| neutral | 134 | 176 | 109 |
| positive | 204 | 350 | 3,403 |

Validation macro-F1 for the selected model is 0.6587 (accuracy 0.8152), within 0.0003 of test.

### Reading the numbers

- Neutral (3-star) is the hard class: F1 0.33, and 350 of 3,957 positive reviews are predicted neutral. Accuracy (0.81) overstates quality because positive is 73% of the data; macro-F1 is the metric to track.
- Grid differences are small (0.644 to 0.659). A gap of about 0.01 between grid points is within what a different seed or split could move, so the choice of C is not strongly supported. Phase 4 will add bootstrap CIs.
- The word vocabulary hits the 200K `max_features` cap (199,981 features), so the cap itself is a limit on the baseline.

## Phase 3: DistilBERT fine-tune

Source: `reports/transformer_metrics.json` (Colab T4, fp16, 504.8 s for 3 epochs, 66,955,779 parameters). Config: `distilbert-base-uncased`, max_len 128, batch 32, lr 2e-5, class-weighted loss, seed 42, one run, no hyperparameter search.

| Epoch | Train loss | Val loss | Val macro-F1 |
|---|---|---|---|
| 1 | 0.711 | 0.619 | 0.6840 |
| 2 | 0.522 | 0.618 | 0.6919 |
| 3 (selected) | 0.401 | 0.716 | 0.6974 |

### Test set (5,400 reviews)

| Metric | DistilBERT | Baseline (Phase 2) |
|---|---|---|
| Accuracy | 0.8444 | 0.8130 |
| **Macro-F1** | **0.6949** | 0.6584 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.7825 | 0.8047 | 0.7935 | 1,024 |
| neutral | 0.2917 | 0.4797 | 0.3628 | 419 |
| positive | 0.9664 | 0.8934 | 0.9284 | 3,957 |

Confusion matrix (rows = true, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | 824 | 168 | 32 |
| neutral | 127 | 201 | 91 |
| positive | 102 | 320 | 3,535 |

Validation macro-F1 is 0.6974 against 0.6949 on test, so selecting the epoch on validation did not inflate the test number. Curves: `reports/figures/transformer_curves.png`.

### Reading the numbers

- DistilBERT is ahead of the baseline by 0.0365 macro-F1 (0.6949 vs 0.6584) on the same test split. That is a point estimate from one training run; whether the gap is outside noise is for the Phase 4 bootstrap CI to say.
- The gain is mostly on negative (F1 0.793 vs 0.746) and positive (0.928 vs 0.901). Neutral improves only from 0.328 to 0.363 and is still the weak class: 320 of 3,957 positive reviews are predicted neutral.
- Validation loss rises in epoch 3 (0.618 to 0.716) while validation macro-F1 still rises, a sign of growing over-confidence rather than worse ranking. I kept epoch 3 because the rule was best val macro-F1; there is no epoch 4 to know whether it would continue.
- Test and time hold-out probabilities are saved in `reports/predictions/` for the Phase 4 CIs and drift check.
