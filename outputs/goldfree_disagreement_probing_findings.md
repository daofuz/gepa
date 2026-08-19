# Gold-free probe acquisition: how few Mini calls buy the critical set?

Setting requested: raw pool, **no gold labels at all**, both models runnable,
minimize Mini calls while harvesting mini-only (MO) examples.
Script: `scripts/sim_goldfree_disagreement_probing.py`; data
`outputs/goldfree_disagreement_probing.json`. Five seeds. No API calls.

## Why disagreement is the right target

On a binary task both models can only be wrong *together* by giving the same
wrong label, so **disagreement is exactly MO ∪ NO** and is observable from the
two predictions alone -- no gold. Movie historical pool (812): 64 disagreements,
of which 55 are MO, so `P(MO | disagree) = 86%` against a base rate of 6.8%.

This is the opposite of the MO-vs-BW problem inside the Nano-wrong set, where
every free feature sits at chance (question length 0.511, CoT length 0.514,
Nano confidence 0.460, options 0.402; only an execution re-roll reaches 0.652).
MO-vs-BW asks a capability-gap question about *gold*; disagree-vs-agree is a
property of the two *outputs*, needs no gold, and each probe labels itself.

## The free filter dominates everything learned

`P(disagree | Nano predicted POSITIVE) = 0.2%` (1 of 459).
`P(disagree | Nano predicted NEGATIVE) = 17.8%` (63 of 353).

All but one critical example lives in the minority class Nano predicts. MO
found (mean over 5 seeds) at a given Mini-probe budget:

| Mini probes | random | +TF-IDF active | minority-class only | minority + active |
|---:|---:|---:|---:|---:|
| 100 | 6.6 | 9.0 | **14.2** | 10.8 |
| 200 | 13.2 | 24.2 | **29.6** | 25.2 |
| 300 | 18.4 | 37.6 | **45.6** | 38.0 |
| 350 | 21.2 | 42.2 | **53.4 (97% recall, ±0.5)** | 42.6 |
| 500 | 32.8 | 52.2 | 54.4 (99%) | 52.8 |

Probing the minority class exhaustively is 353 calls = 43% of the pool and
returns 99% of all MO. A learned TF-IDF disagreement ranker helps a lot over
uniform random probing (2.0x at 300 probes) but is *worse than the free class
rule*, and adding it inside the restricted set makes things slightly worse at
every budget -- inside a set that is already 18% dense it only adds ranking
noise and wastes the cold-start rounds.

## Cost for the winning Movie recipe

`disagree_oracle` (25 disagreements + 25 agreements, 90.67%±0.85 with a
mean_prob prior) needs 25 observed disagreements. Minority-class probing
reaches 24.2 at **150 Mini calls**; uniform random needs ~350; the recipe as
published assumed 812. In Mini-equivalents (Nano = 1/4 Mini): 812 Nano + 150
Mini = 353, against 812 Nano + 812 Mini = 1015. **2.9x cheaper, and the
selection needs zero gold.** Gold is then required only for the 50 selected
rows, to supply the two correctness heads.

## Transfer and limits

- Computer QA is 8--10-way multiple choice, so two wrong answers can differ and
  disagreement is no longer BW-free: `P(MO | disagree) = 51%` there, not 86%.
  The gold-free substitute that does work is Nano self-disagreement (CoT answer
  vs 1-token direct answer): `P(disagree | self-disagree) = 28.6%` vs 9.6%, and
  as a probe ranker (self-disagree first, high proxy confidence next) it finds
  5/8/11/13 MO at 20/40/60/80 probes against a random expectation of
  1.9/3.7/5.6/7.4 -- about 2.6x at the smallest budget.
- The class rule is measured, not assumed. Before anything clever, compute
  `P(disagree | cheap_pred = c)` per class: it is free and on Movie it collapses
  the search space by 2.3x.
- Untested: whether the 50 rows selected this way reproduce 90.67%±0.85 once
  trained. Selection quality and end accuracy have already come apart once
  (`nano_first`: y=36% MO yet 87.0%).
