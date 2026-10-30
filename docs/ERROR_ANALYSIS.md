# Error analysis: DistilBERT on the test split

Sources: `reports/error_sample.csv` (the sample), `reports/error_tags.csv` (the tags), `reports/error_summary.json` (the counts below), `reports/transformer_metrics.json` (the confusion matrix).

## Method

- DistilBERT misclassifies 840 of the 5,400 test reviews. I drew 60 of them at random (seed 42; `sample_errors` in `src/reviewlens/eval/errors.py`).
- Each review was read in full and given one tag: sarcasm, mixed sentiment, neutral boundary, label noise, or other. The rules I used:
  - **mixed_sentiment**: the text states both pros and cons, or a concession ("great, but...").
  - **neutral_boundary**: a 3-star review whose text is mild, short or without a clear verdict.
  - **label_noise**: the text points the opposite way from a 1, 2, 4 or 5 star label.
  - **sarcasm**: the stated sentiment is the reverse of the intended one.
  - **other**: everything else, mostly descriptive or very short text with no usable opinion.
- **One tagger, no second opinion.** I did the tagging, not an independent annotator, and many reviews fit two tags (a 1-star "love the mouse, charger broken" is mixed and also arguably noisy). The counts are my reading, not ground truth; treat differences of a few reviews as noise. The one-line reason for each tag is in `error_tags.csv`.
- With n = 60 the shares carry wide uncertainty (for a share near 50% roughly plus or minus 13 points).

## Tag counts (n = 60)

| Failure type | Count | Share | Predicted neutral by DistilBERT | Baseline also wrong |
|---|---|---|---|---|
| mixed sentiment | 31 | 51.7% | 24 | 22 |
| other | 15 | 25.0% | 11 | 8 |
| neutral boundary | 10 | 16.7% | 0 | 6 |
| label noise | 4 | 6.7% | 3 | 4 |
| sarcasm | 0 | 0.0% | 0 | 0 |

The last two columns are computed from `error_sample.csv` and `error_tags.csv`.

## What the sample shows

- **Neutral is the model's hedge.** 38 of the 60 sampled errors are predictions of neutral, while only 12 of the 60 are true 3-star reviews. In the full test confusion matrix, 706 of the 840 errors (84%) involve the neutral class on one side; only 134 are direct negative/positive swaps.
- **Mixed reviews are the largest group.** 24 of the 31 mixed reviews were predicted neutral. These are cases such as "THE GOOD ... THE BAD ... Overall a great game" (row 451, 4 stars, labelled positive) where the label follows the reviewer's final verdict and the model averages the two halves. This is a labelling-scheme issue as much as a model one: the rating encodes the verdict, the text carries both sides.
- **The neutral boundary is real but small in this sample.** 10 reviews are true 3-star texts such as "Lost interest quickly" (row 3987) or "meets expectations and delivered on-time" (row 4776), where predicting positive or negative is a defensible reading. The model predicted a polar class for all 10.
- **Label noise is rare here (4 of 60)**, for example a 4-star review that lists only complaints (row 1373) or a 5-star "Smaller than expected" (row 2473). I only counted cases where the text clearly contradicts the label, so this is a lower bound.
- **No sarcasm in 60.** Zero of 60 means sarcasm is probably a small share of errors on this dataset (a 95% upper bound of about 5% by the rule of three), not that it never happens. Product reviews of games and peripherals are mostly literal.
- **"Other" is mostly missing signal.** Examples are an emoticon (row 1667), a review cut off after the date (row 3590), usage instructions (row 3724) and long plot summaries. One review (row 3696, "Giving this a 5 out of 5 stars! ... you won't be disappointed") is clearly positive and predicted negative; I found no cause.
- **The baseline shares most of these errors.** The baseline is also wrong on 40 of the 60 (the sample is DistilBERT's errors, so this says the hard cases are hard for both models, not how often the baseline errs overall).

## What could be done (not done in this phase)

- Add a mixed-sentiment signal, for example sentence-level scoring, which Phase 6 builds for aspects.
- Report neutral-class F1 with and without 3-star reviews treated as a soft boundary; the 3-class target is the main limit on macro-F1.
- Review length: several of the long mixed reviews exceed the 128-token window, so the verdict at the end may not reach the model. I did not test this, so it is a hypothesis only.
