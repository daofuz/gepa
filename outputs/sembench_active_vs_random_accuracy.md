# Matched active-versus-random selection comparison

## Protocol

- SemBench Movie, 100 training labels, seeds 505/606/707.
- Same frozen DistilBERT, 8 soft tokens, two correctness heads, inverse-cost
  weighted BCE, validation splits, checkpoint rule, and post-hoc held-out 200.
- Random selects 100 uniformly from each historical training pool.
- Active selects all 53 observable Nano--Mini disagreements and randomly fills
  the remaining 47 positions from agreements. Gold/outcome buckets are not
  used during selection.
- No answer-model API calls were made.

Active composition is 46 MO, 7 NO, 43--44 BC, and 3--4 BW. Random composition
is 5--11 MO, 0--2 NO, 80--89 BC, and 1--9 BW.

## Validation-max-accuracy policy

| Selection / prior | Accuracy (%) | Mini saving (%) | MO recall (%) |
|---|---:|---:|---:|
| Random, no prior | 88.33 +/- 1.70 | 44.00 +/- 16.59 | 52.08 +/- 25.69 |
| Active, no prior | 89.67 +/- 1.65 | 31.33 +/- 10.43 | 75.00 +/- 23.39 |
| Random, pointwise prior k=100 | 89.00 +/- 1.63 | 54.17 +/- 24.38 | 56.25 +/- 28.41 |
| Active, pointwise prior k=100 | **90.67 +/- 0.94** | 28.17 +/- 11.95 | **83.33 +/- 7.80** |

Active selection improves accuracy by 1.33 points without the prior and by
1.67 points with the pointwise prior. It improves MO recall by 22.92 and 27.08
points, respectively. The tradeoff is greater Mini use: saving drops by 12.67
points without the prior and by 26.00 points with it.

The active-plus-prior accuracy gain over matched random-plus-prior is positive
for every seed (+3, +1, and +1 points), unlike the prior-only result on random
data. It remains 0.83 points below all-Mini accuracy (91.50%).

## Accuracy-floor policy

| Selection / prior | Accuracy (%) | Mini saving (%) | MO recall (%) |
|---|---:|---:|---:|
| Random, no prior | 86.83 | 58.50 | 29.17 |
| Active, no prior | 89.50 | 35.33 | 68.75 |
| Random, pointwise prior k=100 | 88.00 | 57.17 | 43.75 |
| Active, pointwise prior k=100 | 88.83 | 48.67 | 56.25 |

The direct uplift rule reaches all-Mini accuracy for all configurations but
routes almost everything to Mini; it is not evidence of useful routing.

## Conclusion

Active disagreement selection produces a clearer and more consistent accuracy
benefit than adding a Mini-better prior to random data. Its gain comes from
recovering more Mini-only critical cases, at the cost of fewer saved Mini
calls. The result supports active selection as the main contribution and the
prior as a secondary ablation. Because the held-out 200 has already been used
post-hoc, these numbers require confirmation on a new untouched test set.
