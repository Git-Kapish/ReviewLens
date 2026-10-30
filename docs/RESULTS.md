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
