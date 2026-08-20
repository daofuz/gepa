# Enriching the validation set does not help — and validation-picked thresholds hurt

Scripts: `scripts/run_computer_qa_knn_validation.py` (budget now settable via
`--n-train/--n-val`), `scripts/run_computer_qa_validation_acquisition.py`.
Data: `outputs/computer_qa_knn_validation_t50v20.json`,
`outputs/computer_qa_validation_acquisition.json`. Computer QA, held-out 205,
five seeds, no API calls.

Question under test: at a 50 train + 20 validation budget, does spending the
20 validation rows on *hard* questions buy a better threshold?

## 1. kNN acquisition finds nothing to enrich

k-NN disagreement propagation on raw-text DistilBERT embeddings picks
validation rows at the pool base rate, not above it:

| validation picked by | disagreements in 20 (5 seeds) | mean |
|---|---|---|
| random | 6, 4, 3, 5, 2 | 4.0 |
| kNN propagation | 5, 6, 4, 2, 3 | 4.0 |
| score boundary (pool median) | 3, 6, 3, 3, 5 | 4.0 |
| score spread (quantiles) | 5, 7, 3, 5, 3 | 4.6 |

Pool disagreement rate 18%, so a random 20 is expected to contain 3.6. Not an
implementation failure: acquisition scores spread normally (0→0.80, 54–68% of
candidates nonzero) and the greedy argmax does take the top-scoring candidate.
The embedding neighbourhood simply carries no information about whether Nano
and Mini disagree, so propagated labels are noise. Moving the search into
score space (boundary/spread), where the threshold actually lives, does not
rescue it either.

## 2. Consequently no acquisition strategy beats random

Test accuracy, paired design — same trained model per seed, the only variable
is which 20 rows become validation:

| criterion | random | kNN | boundary | spread |
|---|---|---|---|---|
| maxacc | 74.24±3.29 | 75.32±3.98 | 73.56±2.68 | 74.63±3.09 |
| cap50 | 72.68±1.80 | 72.49±1.56 | 72.68±1.23 | 72.88±1.73 |
| q50 | 75.80±1.76 | 76.68±1.13 | 76.10±1.07 | 75.80±2.44 |
| q70 | **78.63±1.61** | 77.95±1.13 | 77.76±1.43 | 77.27±1.40 |

Spreads of 0.6–1.1pt against standard deviations of 1.1–4.0. Random is
nominally best at q70. Same conclusion as the data-selection experiments:
the acquisition effect sits below seed noise.

## 3. The load-bearing finding: don't let 20 rows pick the threshold

Compare what the validation set is *used for*. Under `q*` it picks only the
epoch and the threshold comes from a pool-score quantile; under `maxacc`/`cap*`
it picks the threshold too:

| validation's job | accuracy (random acquisition) |
|---|---|
| epoch only, threshold = pool quantile (q70) | 78.63±1.61 |
| epoch + threshold, unconstrained (maxacc) | 74.24±3.29 |
| epoch + threshold, under a rate cap (cap50) | 72.68±1.80 |

Handing threshold selection to 20 labeled rows costs **4–6 points** and
doubles the variance versus the quantile rule. This is the same lesson the
closing single-head experiment reached from the other direction — what
transfers is the distributional position, not a threshold value — and it is
why enriching those 20 rows was never going to be the lever.

## 4. Aside: the 80.98% headline is not budget-legal

`softprompt_router_balanced50_risk_best_expected_utility` reaches 80.98% at a
53.7% Mini rate (above all-Mini's 80.49%), but its 50 training rows come from
bucket-targeted sampling (`mini_only=25, nano_only=10, both_correct=10,
both_wrong=5`), which presupposes mini/nano correctness for the whole pool —
i.e. it spends the full label budget to build a 50-row sample. Under a legal
random-50 protocol the same pipeline lands at 77–79%. Worth stating
explicitly in the paper before a reviewer asks.

## Verdict

Keep the validation set random and small, or drop it entirely; spend nothing
on acquiring hard questions for it, and set the threshold from the unlabeled
pool's score quantile rather than searching it on labeled rows.
