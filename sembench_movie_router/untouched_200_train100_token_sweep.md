# Train-100 soft-token comparison on held-out-200

Fixed settings: 100 labels with the 50 both-correct / 46 mini-only / 3
nano-only / 1 both-wrong mixture, regret-weighted cross entropy, frozen
DistilBERT, 8 epochs, three seeds, and validation-only epoch/threshold
selection. Only the number of learned soft tokens changes.

These 200 rows had already been inspected, so this is a post-hoc held-out
comparison rather than a new untouched confirmation.

## Aggregate results

| Soft tokens | Trainable parameters | Accuracy | Gap vs all-mini | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4,610 | 90.17% | -1.33 pp | 38.00% | 77.08% | 66.67% |
| 8 | 7,682 | 90.83% | -0.67 pp | 21.83% | 89.58% | 71.43% |
| 16 | 13,826 | 87.83% | -3.67 pp | 54.67% | 43.75% | 52.38% |
| All mini | — | 91.50% | 0.00 pp | 0.00% | 100.00% | 100.00% |
| All nano | — | 85.00% | -6.50 pp | 100.00% | 0.00% | 0.00% |

## Eight-token per-seed results

| Seed | Epoch | Threshold | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|---:|---:|
| 505 | 5 | 0.9401 | 90.50% | 28.00% | 81.25% | 64.29% |
| 606 | 2 | 0.7479 | 90.50% | 20.00% | 87.50% | 71.43% |
| 707 | 2 | 0.9269 | 91.50% | 17.50% | 100.00% | 78.57% |

The 8-token configuration is the best accuracy-first operating point in this
comparison. Relative to 4 tokens, it gains 0.66 accuracy point and 12.50
points of critical recall, but gives up 16.17 points of mini saving.

The 16-token model is unstable. Seed 505 degenerates toward all-nano:
85.00% accuracy, 91.50% saving, and 0% critical recall. Increasing prompt
capacity therefore does not monotonically improve this small-data router.
