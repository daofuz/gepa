# Strict two-head MAP prior ablation

## Protocol

- Status: post-hoc evaluation on the existing held-out 200 Movie rows; this is
  not a fresh final test.
- Router: frozen DistilBERT encoder, 8 learned soft tokens, and two correctness
  heads predicting `q_N(x) = P(Nano correct | x)` and
  `q_M(x) = P(Mini correct | x)`.
- Training data: 100 outcome-enriched rows per seed: 50 both-correct,
  46 Mini-only, 3 Nano-only, and 1 both-wrong.
- Three seeds: 505, 606, and 707.
- No answer-model API calls; all labels, outputs, and token counts were cached.

## Objective

For model head `h` in `{Nano, Mini}`, let
`w_h = 1 / E[C_h]`. The data term is

`L_WCE = sum_i sum_h w_h BCE(y_ih, q_ih) / (n sum_h w_h)`.

The cached expected costs have a Nano:Mini ratio of 1:4, giving normalized
head weights 0.8 and 0.2. The explicit Gaussian MAP term is

`L_prior = k/(2n) sum_h (b_h - logit(pi_h))^2`,

with declared prior correctness rates `pi_N = 0.80` and `pi_M = 0.90`.
Thus `L = L_WCE + L_prior`. The routing score is the predicted correctness
uplift `q_M(x) - q_N(x)`; validation-only thresholding controls the accuracy
and Mini-call tradeoff. This places the prior on model correctness rather than
on the final Mini routing rate.

## Held-out results

All values are mean +/- population standard deviation across three seeds.
`Saving` is the percentage of Mini calls avoided relative to all-Mini, and
`MO recall` is recall on the 16 Mini-only critical cases.

| Prior strength k | Accuracy (%) | Saving (%) | MO recall (%) |
|---:|---:|---:|---:|
| 0 | 88.67 +/- 0.62 | 49.83 +/- 7.35 | 58.33 +/- 5.89 |
| 5 | 88.50 +/- 1.08 | 47.83 +/- 3.97 | 56.25 +/- 8.84 |
| 20 | 88.83 +/- 1.03 | 46.33 +/- 4.25 | 60.42 +/- 7.80 |
| 100 | 88.50 +/- 0.71 | 50.00 +/- 8.95 | 56.25 +/- 5.10 |

The table uses the validation-max-accuracy policy. Baselines are 85.00%
accuracy for all-Nano and 91.50% for all-Mini.

Under the validation accuracy-floor policy, `k=0` obtains
87.83 +/- 1.18% accuracy, 71.67 +/- 11.09% saving, and
41.67 +/- 15.59% MO recall. `k=100` obtains 88.00 +/- 1.08% accuracy,
75.83 +/- 9.18% saving, and 41.67 +/- 14.73% MO recall. These differences are
small relative to seed variation.

Using the theoretically direct decision rule `q_M - q_N >= 0` sends every row
to Mini for all tested prior strengths. It exactly matches all-Mini accuracy
but saves no Mini calls.

## Interpretation

The stricter MAP formulation is conceptually cleaner than a prior on the route
intercept, but this experiment does not show a reliable empirical gain. The
best mean change over no prior is at `k=20`: +0.16 percentage points in
accuracy and +2.09 points in MO recall, while saving 3.50 fewer points of Mini
calls. This is within the across-seed variation.

The likely bottleneck is identification rather than regularization. In each
100-row enriched training set, Mini is correct on 96 rows; only the 3
Nano-only and 1 both-wrong rows provide evidence that Mini can fail. A global
base-rate prior cannot teach which inputs belong to those rare cases. The
result therefore supports acquiring more informative critical cases, rather
than presenting the MAP prior as a main contribution. It is suitable as a
negative or robustness ablation, with the post-hoc caveat above.
