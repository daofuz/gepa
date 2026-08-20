# How many labels does a validation-picked threshold need — and should they be hard ones?

Script: `scripts/run_computer_qa_valsize_threshold.py`. Data:
`outputs/computer_qa_valsize_threshold.json`. Computer QA, held-out 205,
five seeds, no API calls. Follow-up to
`outputs/validation_acquisition_findings.md`, testing the objection that the
pool-quantile rule "depends on the data distribution" and that only a
labeled validation set can truly determine the threshold. Diagnostic, not
budget-legal: validation rows here are not charged to the 50-row budget.

Design: one model per seed (50 random train rows), one scoring pass over the
~155 remaining pool rows, then operating-point selection re-run on nested
validation prefixes (10/20/50/100/155) under two orderings — uniform random
and sequential kNN disagreement propagation ("pick hard questions for
validation") — plus an oracle that selects on the test set itself.

## 1. Validation-searched thresholds need ~100 random rows to catch up

Test accuracy (mean±std) and realized Mini rate, maxacc criterion:

| validation rows | random order | rate | kNN order | rate |
|---|---|---|---|---|
| 10 | 73.07±2.36 | 0.17 | 73.07±3.71 | 0.20 |
| 20 | 74.34±3.05 | 0.31 | 75.32±3.98 | 0.43 |
| 50 | 75.12±2.45 | 0.44 | 75.02±4.11 | 0.42 |
| 100 | **78.54±1.75** | 0.78 | 74.93±2.70 | 0.40 |
| 155 | 78.44±1.78 | 0.76 | — | |
| oracle | 81.07±0.57 | 0.82 | | |

Reference: q70 quantile rule sits at 77.4–78.0 (±1.0–1.4) at rate ≈0.68 for
*every* validation size — it only uses the rows to pick an epoch, and even 10
rows suffice for that.

So the objection is half right: threshold search does converge — at about
100 labeled rows it matches the quantile rule's accuracy and keeps climbing
toward the oracle. But the crossover price is steep: 5× the 20-row
validation spend, and the matched accuracy arrives at rate 0.78 vs the
quantile's 0.68 (+0.5pt accuracy for +10pt Mini rate, below the frontier
slope — all-Mini buys 80.5 at rate 1.0). At any budget-plausible size
(10–50 rows) the search loses 3–4pt and carries 2–3× the variance.

On distribution-dependence: the quantile is read off the *deployment pool's*
score distribution, so it tracks the data by construction; its flat 77.4–78.0
line across all validation sizes is that robustness made visible.

## 2. Hard-question validation actively breaks threshold search

The kNN column is the experiment the acquisition line kept circling: enrich
the validation set with hard questions, then let it set the threshold. At
10–50 rows it does nothing (kNN cannot enrich here — see the acquisition
findings); at 100 rows it *costs* 3.6pt (74.93 vs 78.54), with per-seed
operating points swinging from rate 0.08 to 0.94.

The mechanism is instructive: a threshold is a quantile of the score
distribution, so the validation set's job is to *represent* the deployment
distribution. Skewing it toward hard cases is precisely the distribution
shift the objection worried about — self-inflicted. Representative (random)
sampling is not the naive baseline here; it is the requirement.

## 3. Rate-capped search stays broken at every size

cap50 reaches only 72.6–74.6 even with all 155 rows, realizing rates
0.13–0.37 against its 0.50 cap: with coarse validation the feasible
threshold set is quantized, and the search systematically undershoots the
budget it is allowed. Quantile targeting (q50: rate 0.42–0.51 realized)
does not have this failure mode.

## Verdict

"Only a validation set can determine the threshold" holds only in the
oracle limit. In the regime this project operates in (≤50 labeled rows
total), the quantile rule dominates: equal-or-better accuracy, pinned cost,
one-fifth the labels, and immunity to the representativeness failure that
hard-question enrichment introduces. If a future deployment can afford
~100+ labeled rows drawn *uniformly* from its own distribution, threshold
search becomes viable — but those rows must be random, not acquired.
