# SemBench Movie untouched-200 evaluation

## Integrity

- Protocol written: 2026-07-23 15:40:32.
- Manifest locked: 2026-07-23 15:41:55.
- Final result written: 2026-07-23 15:48:00.
- Manifest SHA-256:
  `AF6D93C62505B355D954A32E1E1F5B0086B3BC30219B7B67441CF8B8D1508C1F`.
- The 200 manifest rows had neither mini nor nano output when selected.
- All 400 API calls completed with zero errors.
- The router method, training data recipe, seeds, epoch selection, and threshold
  policy were locked before the new outcomes were collected.
- The historical 82-row test was not added to training or validation.
- Retraining reproduced the prior best epochs and thresholds exactly:
  seed 505 = epoch 3 / 0.8914068341, seed 606 = epoch 2 / 0.7562788129,
  and seed 707 = epoch 5 / 0.5124733448.

## Untouched data

| Outcome | Count | Share |
|---|---:|---:|
| both-correct | 167 | 83.5% |
| mini-only (critical) | 16 | 8.0% |
| nano-only | 3 | 1.5% |
| both-wrong | 14 | 7.0% |
| Total | 200 | 100.0% |

## Baselines

| Policy | Answer accuracy | Mini calls | Mini saving |
|---|---:|---:|---:|
| All nano | 85.0% | 0/200 | 100.0% |
| All mini | 91.5% | 200/200 | 0.0% |
| Outcome oracle | 93.0% | 30/200 | 85.0% |

The outcome oracle follows the conservative target: mini for mini-only and
both-wrong, nano for both-correct and nano-only.

## Locked router results

| Seed | Best epoch | Validation threshold | Accuracy | Gap vs all-mini | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 505 | 3 | 0.8914 | 89.5% | -2.0 pp | 58.5% | 62.5% | 28.6% |
| 606 | 2 | 0.7563 | 89.5% | -2.0 pp | 42.5% | 68.8% | 64.3% |
| 707 | 5 | 0.5125 | 91.5% | 0.0 pp | 1.5% | 100.0% | 92.9% |
| Mean | — | — | 90.17% | -1.33 pp | 34.17% | 77.08% | 61.90% |

Across seeds, the population standard deviations were 0.94 percentage point
for accuracy, 24.00 points for mini saving, 16.40 points for critical recall,
and 26.30 points for the both-wrong-to-mini rate.

## Conclusion

The prior claim that this router can stay close to all-mini while saving a
meaningful fraction of mini calls did not robustly reproduce. The mean router
lost 1.33 accuracy points relative to all-mini, and its 34.17% average saving
came with only 77.08% critical recall. The only seed that matched all-mini
accuracy routed 98.5% of rows to mini, saving just 1.5%.

The main failure is threshold and seed instability. Validation can select an
aggressive threshold that looks safe on 40 rows but misses many of the 16 new
critical cases. Therefore the old 82-row result should be treated as
exploratory, not as a reliable estimate of deployment performance.

This untouched set is now consumed and must not be used to tune another method
that is then reported on the same rows as if they were untouched.
