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
- **Truncation is the largest label-preserving loss for both** (0.109 baseline, 0.133 DistilBERT). Cutting to 8 tokens removes most of the text in the 4,365 reviews (81%) it changes, so part of this is lost information, not fragility. DistilBERT's drop is larger as a point estimate; I did not test whether the two drops differ significantly.
- **Negation is not a label-preserving test.** Putting "not" before one polar word (the result is often ungrammatical, for example "not Fun game to play") does not flip the review's label, so a drop is expected and the macro-F1 change should not be read as an error rate. What it does show is sensitivity: DistilBERT changes its prediction on 38.6% of affected reviews against 13.5% for the baseline, and flips between positive and negative on 10.2% against 6.6%. That DistilBERT reacts more is consistent with it reading the inserted word in context, but this test cannot say whether the reactions are correct.

## Phase 5: topics in negative reviews

Source: `reports/topics.json` (embeddings `all-MiniLM-L6-v2`, BERTopic on the 8,193 train negatives, `min_cluster_size` 40; see D6). Figure: `reports/figures/topic_sizes.png`. Labels are drafts in `reports/topic_labels.json`, not yet confirmed by the project owner.

| Grid `min_cluster_size` | Topics | Outliers (fit docs) | c_v |
|---|---|---|---|
| 20 | 9 | 36.7% | 0.432 |
| **40 (selected)** | 9 | 27.8% | **0.534** |
| 80 (7 topics, outside the 8-12 range) | 7 | 4.8% | 0.566 |

| # | Draft label | Reviews | Mean rating | c_v | Top words |
|---|---|---|---|---|---|
| 0 | Controller and accessory defects | 2,049 | 1.28 | 0.425 | controller, work, xbox, charge, wii, stick |
| 2 | Headset audio and mic failure | 1,427 | 1.38 | 0.835 | sound, mic, headphones, quality, microphone |
| 3 | Game will not install or run | 1,152 | 1.17 | 0.441 | game, code, download, disc, windows, steam |
| 1 | Boring or poor games | 1,149 | 1.45 | 0.506 | game, graphics, story, fun, played |
| 4 | Mouse and keyboard keys failing | 1,021 | 1.39 | 0.724 | mouse, keyboard, keys, click, buttons |
| 5 | Broke after short use | 745 | 1.17 | 0.629 | broke, stopped working, months, weeks |
| 6 | Switch and 3DS cases and screen protectors | 334 | 1.31 | 0.331 | screen, case, switch, 3ds, fit |
| 7 | Short verdicts: cheap, waste of money | 196 | 1.26 | 0.474 | cheap, worth, waste, crap, garbage |
| 8 | Packaging and condition on arrival | 81 | 1.26 | 0.445 | box, opened, package, received |

Mean c_v is 0.534. Of the 11,671 negative reviews, 8,154 have a topic and 3,517 (30.1%) are outliers.

### Reading the numbers

- **Hardware failures dominate.** Topics 0, 2, 4, 5 and 6 (controllers, headsets, mouse and keyboard, generic breakage, cases) are the bulk of the assigned reviews; software and game complaints are topics 1 and 3. The topic with the best coherence (headset, 0.835) is also the most specific one.
- **Outliers are a real share.** 27.8% of the fit documents and a higher share of unseen ones (val 38.0%, test 40.3%, time hold-out 30.8%) are assigned no topic. Held-out reviews are assigned through `transform`, and I did not investigate why they get more outliers than the fit documents, so the train topic mix may not carry over exactly to the other splits.
- **Some topics overlap.** "Broke after short use" (5) cuts across products and shares words with the headset, controller and keyboard topics; topic 7 is mostly one-line reviews with no product signal. I would not read their sizes as clean themes.
- **Coherence is a rough guide.** c_v was computed on the same documents the model was fit on, and the topic with the lowest score (6, cases and screen protectors, 0.331) is still easy to read from its reviews. The numbers rank settings but do not prove the topics are right.
- **Reproducible:** a second full run gave identical topic sizes, top words and c_v.

## Phase 6: aspect sentiment

Sources: `reports/aspects_metrics.json` (hand-labeled eval), `reports/aspects_dev.json` (threshold tuning), `reports/aspect_summary.json` (test-split aggregate), `reports/figures/aspect_sentiment.png`. Labels: `data/labeled/aspect_dev.csv` (40 sentences) and `aspect_eval.csv` (100 sentences, 82 gold aspect-sentiment pairs: 50 negative, 20 positive, 12 neutral). The labels are mine and not independently checked (see D7).

### Hand-labeled evaluation (100 test-split sentences, scored once)

| Configuration | Detection P / R / F1 | Sentiment, aspects found (n) | Sentiment, end to end (n = 82) |
|---|---|---|---|
| keyword only, DistilBERT | 0.827 / 0.524 / 0.642 | 0.674 (43) | 0.354 |
| **keyword + embedding, DistilBERT (main)** | **0.830 / 0.537 / 0.652** | **0.682 (44; 95% CI 0.534 to 0.800)** | **0.366 (0.270 to 0.474)** |
| keyword + embedding, baseline LR | 0.830 / 0.537 / 0.652 | 0.614 (44; 0.466 to 0.743) | 0.329 (0.237 to 0.437) |

- **Detection** compares predicted and gold aspects per sentence: 44 pairs found, 9 false, 38 missed; 58% of sentences have exactly the right aspect set. "End to end" counts a missed aspect as wrong. Intervals are Wilson 95% and wide, since n is 44 to 82.
- **The embedding map adds almost nothing here:** one extra correct pair (44 vs 43) and no change in false pairs. On the dev set the threshold gave F1 0.667 against 0.635 for keyword only, also about one pair.
- **DistilBERT vs baseline sentiment:** 0.682 vs 0.614 on the same detected pairs, but the intervals overlap heavily and I did not test the difference.

### Failure modes (from the 53 sentences with at least one issue)

- **Missed aspects (38 of 82 gold pairs) are the biggest problem.** The opinion is stated without a lexicon word: "Ads and microtransactions galore!" (price), "It worked for 2 days" (durability), "Easy to stow in car" (fit), "Can't even play if it's plugged in" (battery), "THIS GAME DOES NOT WORK AND I CANT SEE HOW TO GET IN TOUCH" (support). The embedding similarity recovered only one of them at the dev threshold.
- **False aspects (9) come from single words in the wrong sense:** `quality` (as in "sound quality") triggered durability in 3 sentences, `combat` in the studio name "Mortal combat" triggered gameplay, "does not support" triggered support_returns, and "power cable" triggered battery for a fault that I labeled durability.
- **Sentiment errors: 14 of the 44 detected pairs.** 9 of the 14 are a polar gold label predicted as neutral (6 negative, 3 positive), so the sentence-level model hedges toward neutral, as the review-level model did in Phase 4. Two are positive predicted as negative, from sentences with two opinions: "good graphics, too bad ... dumb story" got one negative label for both aspects.
- **Neutral gold labels are only 12 of the 82 pairs**, so the handling of neutral cannot be judged from this set.

### Aspect sentiment on the test split (5,400 reviews)

3,488 reviews (64.6%) mention at least one aspect. Per aspect, the share of mentioning reviews with negative, neutral and positive aspect sentiment (DistilBERT):

| Aspect | Reviews | Negative | Neutral | Positive |
|---|---|---|---|---|
| price_value | 1,012 | 23.6% | 16.3% | 60.1% |
| durability | 805 | 28.0% | 22.9% | 49.2% |
| comfort_fit | 684 | 12.3% | 21.5% | 66.2% |
| compatibility_setup | 669 | 29.0% | 18.8% | 52.2% |
| controls | 623 | 25.4% | 30.2% | 44.5% |
| gameplay | 620 | 11.3% | 27.6% | 61.1% |
| delivery_packaging | 545 | 24.2% | 15.2% | 60.6% |
| sound | 511 | 20.2% | 22.1% | 57.7% |
| story | 391 | 12.3% | 25.8% | 61.9% |
| battery | 367 | 26.4% | 22.9% | 50.7% |
| graphics | 353 | 11.3% | 21.2% | 67.4% |
| support_returns | 350 | **49.4%** | 15.4% | 35.1% |
| performance_bugs | 264 | 32.2% | 28.8% | 39.0% |

- **Customer support and returns is the only net-negative aspect** (49.4% negative against 35.1% positive). Every other aspect is net positive, which is expected in a dataset that is 73% positive.
- **In the 1,024 negative-rated test reviews,** the most mentioned aspects are durability (235 reviews, 23%), price_value (228, 22%), compatibility_setup (169, 17%) and support_returns (159, 16%).
- **How far to trust this table:** aspect detection has about 83% precision and 54% recall, and sentiment given a correct aspect is about 68% accurate on a small sample. The counts are undercounts that favor aspects with many keywords, and differences of a few points between aspects (for example 11.3% vs 12.3% negative) are within what these errors could produce. The ranking of support_returns as the worst is the one gap large enough to be unlikely to come from these errors, but it is also the aspect with the fewest labeled examples (2 gold pairs), so its sentiment accuracy is untested.

## Phase 7: trends and explainability

Sources: `reports/trends.json`, `reports/figures/trends.png` (trends); `reports/lr_top_ngrams.json`, `reports/figures/lr_top_ngrams.png`, `reports/explanations.json`, `reports/figures/explanations.html`, `reports/explain_metrics.json` (explanations). Rules and caveats are in D8.

### Sentiment over time (star-rating labels, all 60,000 reviews)

| Year | Reviews | Negative | Neutral | Positive |
|---|---|---|---|---|
| 2014 | 4,215 | 14.0% | 7.6% | 78.5% |
| 2016 | 4,773 | 17.4% | 7.7% | 74.9% |
| 2018 | 4,708 | 21.2% | 7.0% | 71.7% |
| 2020 | 6,719 | 21.5% | 7.0% | 71.5% |
| 2022 | 5,264 | 24.2% | 7.9% | 67.9% |
| 2023 (to September) | 2,469 | 22.9% | 7.1% | 70.0% |

- **Overall sentiment has declined:** the negative share rose from 14.0% in 2014 to 24.2% in 2022, with a small recovery in 2023. The rise is gradual, not a step, and the rule flags only 4 isolated months out of 372 tested (2016-09, 2018-09, 2020-04, 2021-08), none sustained.
- **Not necessarily a quality decline.** Over the same years the mix of what is reviewed changed (next table), and the dataset is a sample of one Amazon category; I cannot separate "products got worse" from "more hardware reviews, which draw more complaints", and a share of ratings is not a satisfaction measure.

### Topics among negative reviews, by year (nearest-centroid assignment)

| Topic | 2014 | 2018 | 2022 | 2023 |
|---|---|---|---|---|
| Boring or poor games | 27.5% | 12.6% | 5.2% | 4.6% |
| Game will not install or run | 18.2% | 11.1% | 7.9% | 9.5% |
| Controller and accessory defects | 12.9% | 20.3% | 24.8% | 22.8% |
| Headset audio and mic failure | 6.5% | 14.1% | 15.7% | 16.6% |
| Mouse and keyboard keys failing | 3.6% | 7.9% | 12.6% | 11.1% |
| Switch and 3DS cases and screen protectors | 2.0% | 5.6% | 8.0% | 7.2% |
| Broke after short use | 8.8% | 12.6% | 12.8% | 13.4% |
| Short verdicts: cheap, waste of money | 5.8% | 5.5% | 3.9% | 4.4% |
| Packaging and condition on arrival | 3.1% | 3.8% | 3.8% | 5.3% |
| Unassigned (nearest the outlier centroid) | 11.7% | 6.4% | 5.3% | 5.0% |

- **Complaints moved from games to hardware:** game-focused topics (boring games, will-not-run) fall from 45.7% of negative reviews in 2014 to 13.1% in 2022, while controller, headset, mouse and keyboard, and case topics rise from 25% to 61%. This matches the category's growing share of accessories, and it is the likely reason the overall negative share rose.

### Rising themes (rule: z >= 3, rise >= 5 points over a 6-month baseline, >= 5 reviews)

- **13 (month, topic) flags in 12 episodes, from 1,020 tests.** Pure binomial noise would give about 1.4 flags at z >= 3, so there is more month-to-month variation than chance, but **11 of the 12 episodes are a single month**, and none is a sustained rise. The only two-month episode is "Boring or poor games" in 2018-10 and 2018-11 (share 26% against a 10.6% baseline). The latest flags are "Game will not install or run" in 2022-11 and "Packaging and condition on arrival" in 2022-12; the data ends in 2023-09 with no flag in 2023.
- **The flags are not stable.** Running the same rule on the fit-time HDBSCAN labels gives 9 flagged months, and only 5 appear in both lists. I treat the flags as spikes worth a look, not as established trends.
- **Slow shifts are not flagged.** The yearly changes above (hardware topics up, game topics down) happen over years, and the 6-month baseline absorbs them.

### Logistic-regression n-grams (baseline, largest coefficients per class)

- **negative:** not, terrible, horrible, garbage, disappointed, awful, waste, worst
- **neutral:** but, okay, however, otherwise, ok, though, ds2, "but the"
- **positive:** great, love, awesome, perfect, amazing, best, excellent, perfectly

The negative and positive lists are what one would expect. The neutral class is mostly hedging words ("but", "however", "though"), which matches the Phase 4 finding that mixed reviews are what sit in the neutral class; "ds2" is a game-specific token that happens to occur in 3-star reviews, which suggests the neutral coefficients partly fit noise.

### DistilBERT token attributions (Integrated Gradients)

- **Gallery:** 10 test reviews (3 correct, 7 from the hand-tagged error sample) in `reports/figures/explanations.html`. In the correct reviews, the shaded words are the ones a reader would pick: "never ... installed" and "popped off after 20 minutes" for negative, "very happy with" for positive. In the errors, the model's reading is visible: "Smaller than expected" (5 stars, tagged label noise) is attributed to "smaller" for the predicted neutral class, and in "Lost interest quickly" (3 stars) "lost" pushes strongly toward negative (+3.02) while "quickly" pushes against it (-1.90).
- **Attributions are noisy on longer reviews:** several highlight function words and punctuation (for example "the", "." and "and" among the top words of some reviews), so the gallery is useful for short reviews and should not be read word by word on long ones.
- **Faithfulness check (95 random test reviews):** deleting the 3 most attributed words lowers the predicted class's probability by 0.225 on average, against 0.056 for 3 random words; the top-attributed deletion hurts more than random in 84.2% of reviews. So the attributions do track what the model uses. This is evidence about the model's behavior, not about whether its reasons are right.
- **Convergence:** Integrated Gradients attributions should sum to the logit difference from the baseline; the mean gap is 0.061 and the maximum 0.403 logit units over the 110 explained reviews, so a few attributions are only approximate with 30 steps.
