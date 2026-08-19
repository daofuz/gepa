# Soft-prompt Mini-prior ablation

## Protocol

- SemBench Movie historical split; the locked 200-review set is excluded.
- 100 outcome-enriched training labels: 50 BC / 46 MO / 3 NO / 1 BW.
- Frozen DistilBERT, eight soft-prompt tokens, eight epochs, seeds 505/606/707.
- Hard target: Mini for MO; Nano for BC, NO, and BW.
- Loss: inverse-expected-cost weighted cross entropy plus
  `lambda / 2 * ((b_mini - b_nano) - logit(0.75))^2`.
- The recorded 4:1 Mini:Nano price ratio gives Mini-target examples about one
  quarter of the weight of Nano-target examples.
- No answer-model API calls were made.

## Test results

Values are three-seed population means. Accuracy, Mini saving, and critical
recall are percentages.

| Prior strength | Calibrated accuracy | Calibrated Mini saving | Calibrated MO recall | Fixed-0.5 accuracy | Fixed-0.5 Mini saving | Fixed-0.5 MO recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 91.87 | 55.28 | 53.33 | 89.02 | 100.00 | 0.00 |
| 0.01 | 90.65 | 41.87 | 46.67 | 89.02 | 100.00 | 0.00 |
| 0.1 | **92.28** | 47.56 | **60.00** | 89.02 | 100.00 | 0.00 |
| 1 | 91.46 | 41.87 | 53.33 | 89.02 | 93.09 | 6.67 |
| 10 | 91.46 | 41.46 | 53.33 | 89.43 | 92.68 | 13.33 |

All-Nano accuracy is 89.02%; all-Mini accuracy is 93.90%.

## Interpretation

The moderate prior (`lambda=0.1`) improves the calibrated three-seed mean by
0.41 accuracy points and 6.67 MO-recall points relative to WCE alone, but loses
7.72 points of Mini saving. The apparent gain is not stable: calibrated
accuracy has a 1.52-point population standard deviation and the three runs are
90.24%, 92.68%, and 93.90%. Stronger priors do not improve the mean.

At a fixed 0.5 threshold, weak priors still collapse to all-Nano. Even very
strong priors recover few MO cases. Validation calibration has a much larger
effect than the intercept prior and absorbs much of its global logit shift.

This is a useful negative/diagnostic result, not evidence for adding the prior
to the primary method. Keep WCE-only as the main objective and, if reported,
present the Mini prior as an ablation showing unstable safety gains and lower
cost saving.
