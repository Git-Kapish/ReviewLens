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

## Phase 4: confidence intervals, drift, error analysis, robustness

Sources: `reports/eval_metrics.json` (bootstrap, drift), `reports/robustness_baseline.json` and `reports/robustness_distilbert.json` (robustness), `docs/ERROR_ANALYSIS.md` and `reports/error_summary.json` (errors). Figures: `reports/figures/bootstrap_ci.png`, `reports/figures/robustness.png`.

### Bootstrap 95% CIs on macro-F1

1,000 paired resamples (the same review indices for both models), percentile intervals, seed 42.

| Split | n | Baseline | DistilBERT | Difference (DistilBERT minus baseline) |
|---|---|---|---|---|
| Test | 5,400 | 0.6584 (0.6436 to 0.6742) | 0.6949 (0.6797 to 0.7108) | **+0.0365 (0.0208 to 0.0520)** |
| Time hold-out | 6,000 | 0.6603 (0.6469 to 0.6751) | 0.7023 (0.6882 to 0.7167) | **+0.0419 (0.0275 to 0.0575)** |

- The difference CI excludes zero on both splits, and DistilBERT is ahead in all 1,000 resamples. So the Phase 3 gap is not explained by which reviews happened to land in the test set.
- What the CI does not cover: it resamples reviews, not training runs. Each model was trained once (one seed), so seed-to-seed variation is not in these intervals.

### Drift: time hold-out

The hold-out is the latest 10% of reviews (from 2022-04-24). Its negative share is 23.8% against 19.0% on the test split.

| Model | Test macro-F1 | Hold-out macro-F1 | Change | CIs overlap |
|---|---|---|---|---|
| Baseline | 0.6584 | 0.6603 | +0.0019 | yes |
| DistilBERT | 0.6949 | 0.7023 | +0.0074 | yes |

- Neither model scores lower on newer reviews; the changes are within noise. I see no evidence of performance drift on this window, even though the class mix shifted.
- The comparison is between different review sets (unpaired), and the hold-out has more negatives, which are easier than neutral for both models. A same-mix comparison was not run.

### Error analysis

60 random DistilBERT test errors, tagged by hand (details and caveats in `docs/ERROR_ANALYSIS.md`):

| Failure type | Count |
|---|---|
| mixed sentiment | 31 |
| other | 15 |
| neutral boundary | 10 |
| label noise | 4 |
| sarcasm | 0 |

38 of the 60 are predicted neutral, and 706 of all 840 test errors involve the neutral class.

### Robustness (test set, 5,400 reviews)

Change in macro-F1 after perturbing the review text, with the true label unchanged. The 95% CI on the change is a paired bootstrap over the same reviews. DistilBERT was scored on a Colab T4, the baseline locally; its clean predictions agree with the Phase 3 predictions on all rows.

| Perturbation | Baseline | DistilBERT |
|---|---|---|
| Typos (15% of words with 4+ letters, adjacent letters swapped) | 0.6584 to 0.6440 (-0.0144; -0.0233 to -0.0062) | 0.6949 to 0.6805 (-0.0144; -0.0249 to -0.0051) |
| UPPERCASE | no change | no change |
| lowercase | no change | no change |
| Emoji appended (neutral ones) | no change | 0.6949 to 0.6981 (+0.0032; -0.0019 to +0.0083) |
| Truncated to the first 8 tokens | 0.6584 to 0.5490 (-0.1094; -0.1285 to -0.0918) | 0.6949 to 0.5620 (-0.1329; -0.1509 to -0.1147) |
| "not" inserted before the first polar word (3,659 reviews) | 0.6649 to 0.5861 (-0.0788; -0.0919 to -0.0652) | 0.6939 to 0.4584 (-0.2355; -0.2547 to -0.2178) |

Reading it:

- **Case and emoji changes are zero by construction, not evidence of robustness.** The baseline lowercases and drops emoji in its preprocessing, and `distilbert-base-uncased` lowercases input, so upper/lower case produces the same model input. DistilBERT's emoji effect (+0.0032) is within noise.
- **Typos cost both models the same, about 0.014 macro-F1.**
- **Truncation is the largest label-preserving loss for both** (0.109 baseline, 0.133 DistilBERT). Cutting to 8 tokens removes most of the text in the 79% of reviews it changes, so part of this is lost information, not fragility. DistilBERT's drop is larger as a point estimate; I did not test whether the two drops differ significantly.
- **Negation is not a label-preserving test.** Putting "not" before one polar word (the result is often ungrammatical, for example "not Fun game to play") does not flip the review's label, so a drop is expected and the macro-F1 change should not be read as an error rate. What it does show is sensitivity: DistilBERT changes its prediction on 38.6% of affected reviews against 13.5% for the baseline, and flips between positive and negative on 10.2% against 6.6%. That DistilBERT reacts more is consistent with it reading the inserted word in context, but this test cannot say whether the reactions are correct.
