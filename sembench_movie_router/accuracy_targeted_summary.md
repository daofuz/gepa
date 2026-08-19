# SemBench Movie accuracy-targeted routing summary

## Protocol

- 412 unique reviews after deduplication.
- 200 unique labeled reviews per seed: 160 train + 40 validation.
- 82 unique held-out reviews per seed: 72 both-correct, 5 mini-only,
  1 nano-only, and 4 both-wrong.
- Train, validation, and test review IDs are disjoint.
- Thresholds are selected using validation data only.
- No new OpenAI API calls; cached GPT-5 mini/nano outputs are reused.

## Baselines

| Policy | Accuracy | Mini rate |
|---|---:|---:|
| All nano | 89.02% | 0% |
| All mini | 93.90% | 100% |
| Outcome oracle | 95.12% | 10.98% |

## Main comparison (three-seed mean)

The main threshold policy maximizes validation accuracy, then chooses the
threshold with fewer mini calls on ties.

| Method | Accuracy | Std. | Mini rate | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|---:|
| 4-token sentiment CE + nano disagreement | **94.31%** | 0.57 pp | **38.62%** | 93.33% | 58.33% |
| 4-token sentiment focal + nano disagreement | 93.90% | 1.72 pp | 45.53% | 86.67% | 58.33% |
| Direct regret-weighted CE | 93.50% | 0.57 pp | 82.52% | 93.33% | **100%** |
| Direct standard CE | 92.68% | 1.00 pp | 62.20% | 73.33% | 66.67% |

The recommended cascade is 0.41 percentage points above all-mini on average,
while avoiding 61.38% of mini calls. Its three test accuracies are 93.90%,
93.90%, and 95.12%.

## Soft-token sweep for the recommended cascade

| Soft tokens | Accuracy | Std. | Mini rate |
|---|---:|---:|---:|
| 4 | **94.31%** | 0.57 pp | 38.62% |
| 8 | 92.68% | 1.99 pp | 33.74% |
| 16 | 93.09% | 0.57 pp | 26.42% |
| 32 | 92.68% | 1.00 pp | 41.87% |

Increasing soft-prompt length did not help at this data size. Four tokens are
both more accurate and more stable.

## Interpretation

Direct routing has difficulty learning model-specific failure from review text.
In the deduplicated data, 25 of 26 mini-only cases are positive reviews that
nano classified as negative. The nano-first cascade exposes this useful signal:
the local soft-prompt sentiment model estimates the semantic label, and the
router escalates when its estimate disagrees sufficiently with nano.

Both-wrong is still a mini training target, but it cannot be guaranteed at
inference because the outcome is unknown until ground truth is observed. The
direct regret-weighted router sends all held-out both-wrong cases to mini, but
requires 82.52% mini usage. The recommended accuracy/cost cascade sends 58.33%
of both-wrong cases to mini; this does not change answer accuracy because both
models are wrong on those rows.
