# Held-out-200 comparison: 100-label versus 200-label router

The 100-label mixture was selected on historical splits, not on these 200
rows. However, these rows had already been inspected in the earlier 200-label
evaluation, so this is a post-hoc held-out comparison rather than a new
untouched confirmation.

## Mean results across seeds 505, 606, and 707

| Method | Accuracy | Gap vs all-mini | Mini saving | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|---:|
| All mini | 91.50% | 0.00 pp | 0.00% | 100.00% | 100.00% |
| 100 labels: 50 / 46 / 3 / 1 | 90.17% | -1.33 pp | 38.00% | 77.08% | 66.67% |
| 200 labels: 133 / 46 / 7 / 14 | 90.17% | -1.33 pp | 34.17% | 77.08% | 61.90% |
| All nano | 85.00% | -6.50 pp | 100.00% | 0.00% | 0.00% |

## 100-label per-seed results

| Seed | Epoch | Threshold | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|---:|---:|
| 505 | 5 | 0.5688 | 89.50% | 45.50% | 62.50% | 42.86% |
| 606 | 7 | 0.8320 | 90.50% | 31.50% | 87.50% | 71.43% |
| 707 | 1 | 0.3952 | 90.50% | 37.00% | 81.25% | 85.71% |

The 100-label method matches the 200-label method's mean accuracy and critical
recall while saving 3.83 additional percentage points of mini calls. It is
also more stable across these three seeds: accuracy standard deviation is
0.47 point versus 0.94, and mini-saving standard deviation is 5.76 points
versus 24.00.

Neither router reaches the accuracy-first target on these rows: both average
1.33 points below all-mini and miss 22.92% of mini-only cases.
