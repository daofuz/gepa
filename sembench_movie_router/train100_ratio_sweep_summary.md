# 100-label critical-focused mixture sweep

The fresh untouched-200 test was excluded. This is an exploratory comparison
on the historical splits only. All non-mixture settings were fixed: 4 soft
tokens, frozen DistilBERT, regret-weighted cross entropy, 8 epochs, batch size
8, and validation max-accuracy threshold selection.

## Results

| Mixture (BC / MO / NO / BW) | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|
| 50 / 46 / 3 / 1 | 93.09% | 27.24% | 86.67% | 91.67% |
| 67 / 23 / 3 / 7 | 93.09% | 48.78% | 73.33% | 66.67% |
| 54 / 40 / 3 / 3 | 92.28% | 26.02% | 73.33% | 91.67% |
| 62 / 30 / 3 / 5 | 91.46% | 57.72% | 53.33% | 58.33% |
| 46 / 46 / 3 / 5 | 91.46% | 57.72% | 46.67% | 75.00% |

All-mini accuracy on these stratified historical tests is 93.90%.

## Best 100-label mixture

The most conservative useful mixture is:

- 50 both-correct
- 46 mini-only
- 3 nano-only
- 1 both-wrong

Per-seed results:

| Seed | Epoch | Threshold | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|---:|---:|
| 505 | 5 | 0.5688 | 92.68% | 29.27% | 80% | 100% |
| 606 | 7 | 0.8320 | 93.90% | 14.63% | 100% | 100% |
| 707 | 1 | 0.3952 | 92.68% | 37.80% | 80% | 75% |

## Interpretation

Increasing mini-only examples helps only when both-wrong examples are kept
very low. With the same 46 mini-only examples, changing both-wrong from 1 to 5
drops mean accuracy from 93.09% to 91.46% and critical recall from 86.67% to
46.67%. Both-wrong appears to be weakly learnable or noisy, even though its
conservative routing target remains mini.

The 100-label result does not dominate the prior 200-label method. On the same
historical evaluation:

| Training recipe | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|
| 200 labels: 133 / 46 / 7 / 14 | 93.50% | 21.95% | 93.33% | 100% |
| 100 labels: 50 / 46 / 3 / 1 | 93.09% | 27.24% | 86.67% | 91.67% |

Thus the 100-label method buys 5.29 additional percentage points of mini
saving but loses 0.41 accuracy point, 6.66 points of critical recall, and some
conservative both-wrong coverage. For an accuracy-first objective, the
200-label method remains safer.
