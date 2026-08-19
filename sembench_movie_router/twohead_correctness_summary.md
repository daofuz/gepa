# Two-head correctness router experiment

Setup: 100 labels with the 50 both-correct / 46 mini-only / 3 nano-only / 1
both-wrong mixture, 8 soft tokens, frozen DistilBERT, three seeds, and the
same held-out-200 rows. The two sigmoid outputs estimate P(nano correct) and
P(mini correct).

## Results

| Method | Routing policy | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---|---|---:|---:|---:|---:|
| Current one-head manual weights | Validation threshold | 90.83% | 21.83% | 89.58% | 71.43% |
| Two-head unweighted BCE | Direct correctness argmax | 91.50% | 0.00% | 100.00% | 100.00% |
| Two-head unweighted BCE | Validation calibrated | 88.17% | 62.00% | 47.92% | 35.71% |
| Two-head balanced BCE | Direct correctness argmax | 90.67% | 18.83% | 83.33% | 78.57% |
| Two-head balanced BCE | Validation calibrated | 89.33% | 40.17% | 68.75% | 52.38% |
| Two-head balanced BCE | Conservative validation | 90.17% | 23.33% | 81.25% | 64.29% |
| All mini | — | 91.50% | 0.00% | 100.00% | 100.00% |
| All nano | — | 85.00% | 100.00% | 0.00% | 0.00% |

## Interpretation

Unweighted BCE sees 96 mini-correct versus only 4 mini-incorrect training
labels, so the mini head learns that mini is almost always correct. Directly
choosing the larger correctness probability consequently routes every row to
mini. It matches all-mini but provides no savings.

Per-head inverse-frequency balancing makes the router non-degenerate. The
direct policy identifies about one of the three nano-only rows per seed on
average, but misses 2.67 of the 16 mini-only rows. Since beating all-mini
requires captured nano-only rows to outnumber missed mini-only rows, the net
accuracy remains 0.83 point below all-mini.

Requiring 100% mini-only recall and 100% both-wrong-to-mini on the 40-row
validation set does not generalize to the held-out rows. The bottleneck is not
only threshold calibration; four mini-incorrect training examples are
insufficient for the mini-correctness head to learn a stable boundary.

The current one-head, 8-token manual-weight router remains the better
accuracy/cost compromise. A meaningful next attempt would require more
nano-only and mini-incorrect training examples, rather than another threshold
rule on the same small validation set.
