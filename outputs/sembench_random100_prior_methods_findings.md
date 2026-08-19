# Random-data Mini-better prior ablation

## Protocol

- SemBench Movie, natural random 100-label training data for each seed.
- Seeds 505, 606, and 707; every prior uses exactly the same examples within a seed.
- Frozen DistilBERT, 8 learned soft tokens, and separate Nano/Mini correctness heads.
- Data loss: inverse-expected-query-cost weighted BCE.
- Prior belief: Mini has higher base correctness than Nano.
- Prior strengths: effective-sample scales `k=20` and `k=100`.
- Threshold/checkpoint selection uses validation only.
- The existing 200-row held-out set is post-hoc and is not a fresh final test.
- No answer-model API calls were made.

The random training mixtures were:

| Seed | BC | MO | NO | BW | Nano acc. | Mini acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 505 | 86 | 11 | 2 | 1 | 88% | 97% |
| 606 | 89 | 5 | 0 | 6 | 89% | 94% |
| 707 | 80 | 11 | 0 | 9 | 80% | 91% |

## Prior formulations

- `bias_map`: Gaussian MAP prior on the two correctness-head intercepts,
  centered at Nano accuracy 0.80 and Mini accuracy 0.90.
- `mean_accuracy_kl`: KL penalty between those declared base rates and the
  batch-mean predicted correctness probabilities.
- `mean_dominance`: softplus ranking penalty requiring mean Mini correctness
  log-odds to exceed Nano's.
- `pointwise_dominance`: the same ranking preference on every input. This is
  the strongest assumption and can conflict with Nano-only cases.

## Held-out results

The main table uses the validation-max-accuracy policy. Values are mean +/-
population standard deviation over three seeds. Saving is relative to all-Mini.

| Prior | Accuracy (%) | Mini saving (%) | MO recall (%) |
|---|---:|---:|---:|
| None | 88.33 +/- 1.70 | 44.00 +/- 16.59 | 52.08 +/- 25.69 |
| Bias MAP, k=20 | 88.67 +/- 2.32 | 35.67 +/- 22.86 | 60.42 +/- 34.74 |
| Bias MAP, k=100 | 88.50 +/- 2.27 | 48.83 +/- 31.69 | 52.08 +/- 35.84 |
| Mean-accuracy KL, k=20 | 88.17 +/- 2.25 | 44.83 +/- 28.77 | 56.25 +/- 30.62 |
| Mean-accuracy KL, k=100 | 88.33 +/- 2.01 | 52.83 +/- 15.95 | 50.00 +/- 27.00 |
| Mean dominance, k=20 | 88.33 +/- 2.09 | 47.67 +/- 25.76 | 54.17 +/- 34.74 |
| Mean dominance, k=100 | 88.17 +/- 1.43 | 54.00 +/- 20.87 | 45.83 +/- 23.01 |
| Pointwise dominance, k=20 | 88.33 +/- 2.72 | 51.67 +/- 26.15 | 50.00 +/- 35.72 |
| Pointwise dominance, k=100 | **89.00 +/- 1.63** | **54.17 +/- 24.38** | **56.25 +/- 28.41** |

Baselines are 85.00% for all-Nano and 91.50% for all-Mini.

## Interpretation

`pointwise_dominance, k=100` has the best aggregate operating point. Relative
to no prior, its paired mean changes are +0.67 accuracy points, +10.17 Mini
saving points, and +4.17 MO-recall points. However, the improvement is not
consistent across seeds: accuracy changes are -3.0, +5.0, and 0.0 points, and
MO-recall changes are -56.25, +75.0, and -6.25 points. The apparent mean gain
therefore comes from one favorable seed and is not reliable evidence that the
prior works.

Bias MAP at `k=20` raises mean accuracy by 0.33 points and MO recall by 8.33
points, but loses 8.33 saving points and has even larger variance. The other
priors mostly move the threshold operating point rather than consistently
improving sample ranking.

The direct theoretical rule `P(Mini correct) >= P(Nano correct)` nearly
collapses to all-Mini for every dominance or bias prior: it reaches 91.50%
accuracy with approximately zero saving. Validation calibration is doing most
of the useful routing work.

The conclusion is negative but informative: on natural random-100 data, a
global belief that Mini is usually better does not replace critical-case data.
The three random samples contain only 5--11 MO cases and at most 2 NO cases,
and their validation-selected frontiers vary sharply. The pointwise prior is
worth retaining as an appendix ablation or rerunning with more seeds, but it
should not replace disagreement-guided data acquisition in the main method.
