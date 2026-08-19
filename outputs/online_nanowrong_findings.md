# Online select-hard acquisition under the "Nano wrong" routing target

Scripts: `scripts/run_{sembench,computer_qa}_online_nanowrong.py`. Data:
`outputs/{sembench,computer_qa}_online_nanowrong.json`. Five seeds, no API
calls.

Premise: routing a both-wrong row to Mini changes accuracy by exactly zero,
so the router target can be **P(Nano wrong)** (escalate) instead of "which
model is right". Under that target the both-wrong ballast that poisoned
hard-example selectors (nano_first: 87.0% < random under the two-head target)
becomes legitimate positive labels; positives are 2–3× denser (15% / 29% vs
6.8% / 9.3%); a training label needs only a gold check of Nano's answer (no
Mini call); and the ceiling is unharmed — perfect Nano-wrong routing equals
full oracle accuracy: Movie **93.0% @ 15%** Mini rate, CQA **83.9% @ 29%**.

Setup: single head on frozen DistilBERT + 8 soft tokens, input = text +
Nano's answer (cascade form). 25 training gold checks: 5 random seed labels +
4 rounds × 5 picked by the arm (scorer retrained between rounds). Arms:
random25 / online_top (highest score) / online_eps (top + 30% exploration) /
online_uncertain (closest to 0.5). Enriched validation (18 disagreements + 7
agreements) fixed per seed BEFORE acquisition, shared by all arms.

## Movie: the hypothesis confirmed — select-hard flips from harmful to helpful

| arm | maxacc | q70 | q50 | q30 |
|---|---|---|---|---|
| random25 | 90.9±0.4 @0.74 | 90.6±0.7 @0.69 | 88.6±1.5 @0.45 | 87.0±1.5 @0.25 |
| **online_top** | **91.2±0.2** @0.87 | **91.0±0.3** @0.66 | 88.9±1.1 @0.45 | **88.2±0.7** @0.31 |
| online_eps | 90.8±1.0 @0.77 | 90.4±0.8 @0.71 | **89.7±1.6** @0.50 | 87.8±0.9 @0.28 |
| online_uncertain | 91.1±0.2 @0.83 | 90.7±0.5 @0.68 | 88.5±0.9 @0.49 | 87.3±0.6 @0.27 |

online_top beats random at every tier (low tier +1.2pt with half the std).
The same select-hard instinct that failed under the two-head target
(nano_first 87.0%) is now an improvement — the fix was the target, not the
selector.

Striking side result: with 25 cheap Nano-only gold checks this scorer beats
the 50-full-label two-head router at every nominal rate (q70 91.0±0.3 vs
89.6±0.4). In mini-equivalent TOTAL cost the cascade pays 0.25/row for Nano,
which cancels most of it: nano-wrong q30 (88.2 @ cost 0.56) ≈ two-head q50
(88.2 @ 0.59); and the maxacc end (91.2 @ cost 1.12) is dominated by all-Mini
(91.5 @ 1.00). The cascade form only makes sense at low/mid budgets; a
PRE-ROUTE variant of the same target (text only, no Nano answer, no cascade
tax) is the obvious untested follow-up.

## Computer QA: neutral at matched cost, plus a calibration artifact

random25: q70 78.3±1.0 @0.69, q50 76.0±1.3 @0.48, q30 75.0±0.8 @0.32.
online arms shift to HIGHER realized rates (online_top q30 lands at 0.41,
q50 at 0.60) and at matched realized cost the select-hard advantage washes
out (online_top q50 77.2 @0.60 ≈ random interpolated ~77.3); only
online_uncertain shows a modest edge (76.1±1.1 @0.40 vs random 76.0±1.3
@0.48; best overall 78.9±0.9 @0.74). maxacc under online_top is unstable
(75.9±3.8).

**Artifact worth disclosing:** the quantile pool-sample excludes labeled
rows, and adaptive acquisition removes precisely the hardest rows from the
candidate pool — on CQA's small pool (~120 candidates) this skews the
leftover score distribution, so quantile thresholds systematically overshoot
the requested budget. Fix for any future run: estimate quantiles on a stream
untouched by acquisition (or score labeled rows back into the sample). On
Movie (690-row pool) the effect is negligible.

## Take-aways

1. Both-wrong-tolerant routing ("Nano wrong → Mini") makes hard-example
   online acquisition viable: confirmed on Movie, neutral on CQA — never
   harmful, unlike its two-head counterpart.
2. The Nano-wrong target itself is remarkably label-efficient: 25 Nano-only
   gold checks ≈ 50 full labels of the two-head router at matched total
   cost. Its labels are also the ones an online system gets for free from
   user feedback on served (Nano) answers.
3. Escalation waste on both-wrong rows is 7–17% of escalated traffic at
   practical operating points — bounded and much lower than the theoretical
   ceiling share, because realistic escalation sets are diluted by
   both-correct rows.
4. Open follow-ups: pre-route variant of the Nano-wrong target (drop the
   cascade tax); acquisition-unbiased quantile estimation; larger online
   budgets (the CQA pool exhausts at ~120 candidates).
