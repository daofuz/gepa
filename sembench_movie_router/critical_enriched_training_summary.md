# Smaller critical-enriched outcome-only training

## Fixed protocol

- Original per-seed 40-review validation and 82-review test remain unchanged.
- Router supervision is outcome-only; no semantic-label target or model answer input.
- Model and optimization: four soft tokens, frozen DistilBERT, regret-weighted CE,
  batch size 8, eight epochs, validation max-answer-accuracy threshold.

## Training compositions and results

| Training composition | Train size | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|---:|
| 126 BC / 17 MO / 3 NO / 14 BW (original) | 160 | 93.50% | 17.48% | 93.33% | 100.00% |
| 110 BC / 46 MO / 7 NO / 37 BW (all rare outcomes) | 200 | 93.50% | **31.71%** | 86.67% | 75.00% |
| 133 BC / 46 MO / 7 NO / 14 BW (critical-focused) | 200 | **93.50%** | **21.95%** | **93.33%** | **100.00%** |

BC = both-correct, MO = mini-only, NO = nano-only, BW = both-wrong.

The critical-focused composition is the best balanced exploratory point.  It
keeps every one of the 46 available train mini-only examples, but does not
amplify the largely unlearnable both-wrong bucket.  Relative to the original
160-row training set, it preserves accuracy and both risk metrics while adding
4.47 points of mini saving.

## Per-seed critical-focused result

| Seed | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---:|---:|---:|---:|
| 505 | 93.90% | 43.90% | 100.00% | 100.00% |
| 606 | 92.68% | 21.95% | 80.00% | 100.00% |
| 707 | 93.90% | 0.00% | 100.00% | 100.00% |

Mini usage remains highly variable across splits, so this result needs a fresh
untouched test set or additional tasks before being treated as a final estimate.
Repeated selection on the same 82-review test should be considered exploratory.
