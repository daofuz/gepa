# SemBench Movie: 400 additional API-evaluated reviews

## Data collection

- Raw Movie rows: 2,000; unique review IDs: 1,865.
- Previously complete mini/nano pairs: 412 unique reviews.
- Newly sampled reviews: 400 unique, uniformly sampled from rows missing both outputs.
- API calls: 800/800 completed, zero errors (mini and nano for every review).
- Cache after collection: 812 complete unique reviews.
- New outcomes: 344 both-correct, 29 mini-only, 4 nano-only, 23 both-wrong.
- API usage recorded by the collector: 48,034 input tokens and 15,999 output tokens.

## Fixed comparison protocol

The original per-seed 82-review test and 40-review validation sets were rebuilt
from the old 412-review pool and kept unchanged.  All 400 new reviews are
train-only.  The expanded split is therefore:

- 690 train: 600 both-correct, 46 mini-only, 7 nano-only, 37 both-wrong.
- 40 validation: 32 both-correct, 4 mini-only, 1 nano-only, 3 both-wrong.
- 82 test: 72 both-correct, 5 mini-only, 1 nano-only, 4 both-wrong.
- Train plus validation contains 50 mini-only critical examples.

The router remains outcome-only: no positive/negative auxiliary target and no
mini/nano answer in the input.

## Results (three-seed mean)

| Labels | Batch | Threshold policy | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---|---:|---:|---:|---:|
| 200 | 8 | Validation max accuracy | 93.50% | **17.48%** | 93.33% | 100.00% |
| 730 | 8 | Validation max accuracy | 92.68% | 31.30% | 80.00% | 83.33% |
| 730 | 8 | Conservative critical + both-wrong | 93.50% | 9.76% | 93.33% | 100.00% |
| 730 | 32 | Validation max accuracy | 93.09% | 30.08% | 86.67% | 83.33% |
| 730 | 32 | Conservative critical + both-wrong | **93.90%** | 13.82% | **100.00%** | **100.00%** |

## Conclusion

The additional data improves the safest operating point, not the maximum
saving point.  With batch 32 (approximately matching the original number of
optimizer updates) and conservative validation calibration, every seed exactly
matches all-mini accuracy and routes every held-out mini-only and both-wrong
case to mini.  The price is 13.82% mini saving, 3.66 points below the original
200-label regret-CE baseline.

Aggressive max-accuracy tie-breaking still fails on one seed: it saves about
30% mini calls but loses 0.81 accuracy points on average and misses critical
cases.  More data alone therefore does not solve threshold instability.
