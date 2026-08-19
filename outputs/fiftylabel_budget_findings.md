# A 50-label protocol with a dialable cost knob

Scripts: `scripts/run_sembench_50label_budget.py`,
`scripts/run_computer_qa_50label_budget.py`. Data:
`outputs/sembench_50label_budget.json`, `outputs/computer_qa_50label_budget.json`.
Five seeds, held-out Movie 200 / CQA 205, no API calls.

Question: can the whole gold budget (train + validation) fit in 50 labels,
and can the serving cost be controlled instead of landing wherever
validation-max-accuracy puts it?

Protocol: train on N uniform random gold rows (the scaling study showed
training gains nothing from curated disagreements); validation =
disagreement-enriched (the validation study showed that is where
disagreements pay). Splits: t25_v25 (train 25 + val 18 disagreements / 7
agreements) and t30_v20 (train 30 + val 15/5). Seven operating-point
criteria on the same per-epoch scores (exactly paired): `maxacc` (status
quo), `capB` (val-max-accuracy s.t. VALIDATION Mini rate ≤ B), `qB`
(threshold = score quantile on 100–200 UNLABELED pool rows, gold never read;
validation only picks the epoch), B ∈ {30, 50, 70}%.

## 50 labels match the 80–112-label protocols

- **Movie** t25_v25, maxacc: **90.8±0.5** @ 0.85 Mini rate — 0.2pt below the
  80-label enriched-validation result (91.0±0.45), far above the 90-label
  standard protocol (88.6±1.8 @ unstable 0.56±0.22).
- **CQA** maxacc: t30_v20 **79.2±1.1** / t25_v25 78.6±1.6 — matching the
  112-label previous best (79.2±1.3) at 45% of the gold.

The validation-heavy split wins or ties: on Movie t25_v25 beats t30_v20
everywhere; on CQA they split by criterion. Labels moved from training to a
disagreement-enriched validation keep paying; labels 51–112 of the old
protocols were nearly worthless.

## The quantile criterion turns cost into a knob

Movie menu (t25_v25, no prior; total cost = Mini-rate·1 + (1−rate)·0.25):

| criterion | accuracy | Mini rate | cost |
|---|---|---|---|
| maxacc | 90.8±0.5 | 0.85±0.07 | 0.89 |
| q70 | 89.6±0.4 | **0.69±0.04** | 0.77 |
| q50 | 88.2±1.1 | **0.45±0.04** | 0.59 |
| q30 | 86.3±0.7 | **0.28±0.05** | 0.46 |

CQA: q70 77.6–78.0±1.1–1.8 @ 0.68–0.71±0.04–0.05, q50 ≈76 @ ~0.5±0.05–0.09,
q30 ≈74–75 @ ~0.3±0.03–0.08.

The realized Mini rate lands within a few points of the requested budget with
seed-std 0.01–0.09 on both datasets — the operating point's cost is no longer
a random variable but a dial, at ~1pt accuracy below maxacc per 15pt of
budget on Movie.

## Why `capB` fails and `qB` doesn't

`capB` enforces the budget on the VALIDATION Mini rate, but the enriched
validation is deliberately composition-skewed (60–72% disagreements), so its
rate under a threshold is a biased estimator of the deployment rate — on CQA
cap50 realizes 0.28±0.14 (undershooting the budget by half, losing 3–5pt of
available accuracy). `qB` estimates the rate where it is actually defined,
on an unlabeled pool sample, which is free. Rule: **composition-skewed
validation sets may choose epochs and compare models, but never estimate
rates; rates come from pool quantiles.**

## Zero-validation variant (added 2026-08-17)

Post-hoc analysis on the stored per-epoch held-out scores of
`outputs/{sembench,computer_qa}_disagree_validation.json` shows that the
validation set can be dropped entirely for the quantile tiers: **fixing the
epoch to the last one (8) and taking the threshold from pool-score
quantiles matches the validation-chosen epoch at every budget on both
datasets** (Movie: last-epoch 86.6/88.1–89.2/90.0–90.4 vs val-chosen
86.7/88.0–88.4/89.6–90.1 at budgets 0.3/0.5/0.7; CQA: 73.8–74.3/76.1–76.3/
78.0–78.3 vs 73.9–74.6/76.1–76.5/77.8–78.1 — all differences ≤0.8pt, both
signs). This is consistent with the epoch-channel decomposition (epoch
choice worth −0.1 to +0.6pt).

Minimal protocol, then: **25 random gold training labels, train a fixed 8
epochs, threshold = pool-score quantile at the chosen budget. No
validation, no enrichment, no probing.** A fixed absolute threshold VALUE
does not transfer (score distributions shift across seeds/epochs; the same
operating point maps to thresholds from −0.04 to +0.32) — the fixed object
is the RULE (quantile position), not the number. The enriched validation
remains necessary only for the validation-searched max-accuracy tier; the
quantile menu at budget 0.9 comes within ~0.5pt of it on Movie.

## Recommended recipe (both datasets, 50 gold labels total)

1. Label 25 uniform random rows → train (with mean_prob prior optional).
2. Run both models over an unlabeled candidate pool (or reuse shadow-run
   history); label 18 disagreements + 7 agreements → validation.
3. Train; per epoch score validation and a ~200-row unlabeled pool sample.
4. Pick the budget B; threshold = pool-score quantile at B; epoch by
   validation accuracy under that threshold. maxacc is the B≈85–90% end of
   the same menu.
