# Movie router transfer to Computer Science QA

## Setup

- Source task: SemBench Movie positive-review routing.
- Target task: Computer Science multiple-choice QA.
- Router: 4 learned soft tokens, frozen DistilBERT, regret-weighted cross-entropy.
- Target split: 165 train, 40 validation, 205 fixed test.
- Router input: category, source, question, and options; no nano answer.
- Results are averaged over seeds 505, 606, and 707.

## Results on the 205-question test set

| Method | Accuracy | Mini route rate | Critical recall | Both-wrong routed to mini |
|---|---:|---:|---:|---:|
| All mini | 80.49% | 100.00% | 100.00% | 100.00% |
| All nano | 71.22% | 0.00% | 0.00% | 0.00% |
| Strict zero-shot transfer | 80.33% +/- 0.23 pp | 96.91% | 98.72% | 96.97% |
| Movie weights + QA threshold calibration | 78.37% +/- 0.92 pp | 71.87% | 76.92% | 75.76% |
| Retrain the same router on QA labels | 75.61% +/- 1.44 pp | 53.98% | 48.72% | 54.55% |

## Interpretation

The strict zero-shot result is close to all-mini accuracy, but it routes 96.91% of questions to mini. It therefore does not demonstrate useful cross-task routing. Calibrating only the threshold on 40 QA examples reduces mini usage to 71.87%, but loses 2.11 percentage points relative to all-mini. Retraining the same direct router saves more mini calls, but its 4.88-point accuracy gap shows that question text alone and 205 target-task labels are insufficient for reliable critical-case detection on this QA task.

