# Spend annotated disagreements on validation, not training

Scripts: `scripts/run_sembench_disagree_validation.py`,
`scripts/run_computer_qa_disagree_validation.py`. Data:
`outputs/sembench_disagree_validation.json`,
`outputs/computer_qa_disagree_validation.json`. Five seeds, no API calls.

## Diagnosis

Threshold search only reacts to validation rows where the routing decision
changes the selected answer's correctness — the rows where **exactly one
model is correct**. On agreement rows the routed answer is identical either
way; on both-wrong-differently rows (multi-class only) it is wrong either
way. The standard stratified Movie validation holds **5 such rows out of
40**; Computer QA holds ~9 of 62. The operating-point instability that
dominated every recent sweep (Mini-rate std 0.2–0.3) is threshold+checkpoint
selection fit on a handful of effective examples.

## Design

Same training path as the scale/augment sweeps (a fresh randomness
realization — per-epoch evaluation shifts DataLoader RNG draws, so runs are
not bit-identical, but standard-design aggregates match those sweeps within
seed noise). Every epoch is scored on two validation designs; (epoch,
threshold) is picked per design from the same per-epoch models, so the
comparison is exactly paired:

- **standard** — stratified validation (Movie 40 gold rows / CQA 62).
- **disagree_val** — Movie: 20 disagreements + 10 agreements (30 gold rows);
  CQA: up to 25 disagreements + 10 agreements (35 gold rows).

## Movie: large, uniform gains at 25% less validation gold

| training config | standard | disagree_val |
|---|---|---|
| random40, none | 88.6±1.8 @ rate 0.56±0.22 | **91.0±0.45** @ 0.86±0.10 |
| random40, mean_prob | 88.2±1.8 @ 0.47±0.27 | 90.7±0.81 @ 0.81±0.10 |
| d25_a25, none | 89.6±1.2 @ 0.61±0.22 | 90.9±0.58 @ 0.87±0.10 |
| d25_a25, mean_prob | 89.4±1.6 @ 0.59±0.26 | **91.4±0.20** @ 0.92±0.04 |

Accuracy +1.3 to +2.5pt; accuracy std shrinks 2–8×; Mini-rate std shrinks
2–6×; Mini-only recall rises from 0.49–0.69 to 0.88–0.96. Informative rows:
5/40 → 20/30. On a binary task every disagreement is an
exactly-one-model-correct row, so the enrichment is free of dilution and the
rows are selectable without gold (disagreement is observable).

**Cost caveat:** the chosen operating points move to 0.81–0.92 Mini rate.
This is the selection criterion, not the validation set: with a trustworthy
signal, validation-max-accuracy faithfully finds its optimum, which on Movie
sits near all-Mini. Cost control needs a budget-constrained criterion on the
same enriched validation (per-epoch held-out score vectors are stored in the
result JSONs, so operating-point policies can be re-evaluated post hoc
without retraining).

## Computer QA: direction confirmed, magnitude structurally diluted

| training config | standard | disagree_val |
|---|---|---|
| random50, none | 75.7±1.7 @ 0.48±0.16 | 77.2±1.9 @ 0.70±0.18 |
| random50, mean_prob | 79.2±1.3 @ 0.75±0.16 | 79.3±1.4 @ 0.81±0.11 |
| d25_a25, none | 76.4±2.8 | 76.1±2.5 (enrichment infeasible) |
| d25_a25, mean_prob | 77.3±2.2 | identical (enrichment infeasible) |

Two structural reasons the effect is smaller than Movie's:

1. **Dilution**: 24–28% of multi-class disagreements are
   both-wrong-differently, which carry no threshold signal. Truly informative
   rows go 9 → 18–19, a 2× enrichment against Movie's 4×.
2. **Allocation conflict**: when training already consumed the pool's ~25
   disagreements (d25_a25), the enriched validation clamps to the standard
   supply (12–14 informative) and the effect vanishes. On a small pool,
   training and validation compete for the same disagreement rows. Combined
   with the scaling finding that training gains nothing from them, the
   allocation answer is: **give the disagreements to validation.**

The already-best CQA config (random50 + mean_prob prior) is flat on accuracy
but still tightens Mini-rate std (0.16 → 0.11) — the prior was already doing
part of the anchoring job that the enriched validation provides.

## Implications

- The acquisition chapter's recipe becomes: train on cheap random labels
  (plus a prior), and spend disagreement annotation on the **validation set
  that selects the operating point**. Same rows, different role, and the
  single largest stability improvement measured on this line: Movie
  91.0±0.45 (random40 training) / 91.4±0.20 (d25 training) versus the
  88–90.5 band with std up to 2.0 that every training-side intervention
  produced.
- On binary tasks the enrichment is selectable gold-free (disagree ≡ exactly
  one correct); on k-way tasks it dilutes by the both-wrong-differently
  share, which bounds the expected benefit before running anything.
- Next step: budget-constrained (epoch, threshold) selection on the enriched
  validation — reusable from the stored per-epoch scores — to keep the
  stability gain while choosing cheaper operating points.
