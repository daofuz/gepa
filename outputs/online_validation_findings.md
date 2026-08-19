# Probe-built validation: the recipe works end to end with no oracle anywhere

Scripts: `scripts/run_{sembench,computer_qa}_online_validation.py`. Data:
`outputs/{sembench,computer_qa}_online_validation.json`. Five seeds, no API
calls (cache simulation).

Question: the enriched validation (18 disagreements + 7 agreements) was so
far built with oracle knowledge of the pool. Can it instead be built by
paid probing — each probe is one Mini call whose agree/disagree outcome
self-labels — and does a validation selected that way (with whatever bias
the selection carries) still deliver the enrichment gains? Two arms:
uniform random probing, and an online-learned prober (frozen DistilBERT + 8
soft tokens on [text + Nano answer] predicting P(disagree), retrained every
20 probes, probing its top-ranked candidates). Downstream identical to the
50-label protocol: same random-25 training rows, two-head router, no prior,
quantile + max-accuracy criteria.

## Effectiveness: PASS on both datasets — no oracle needed anywhere

Held-out operating points of probe-built validations vs the oracle-built
reference (t25_v25, no prior):

| dataset | criterion | probe-built (online arm) | oracle reference |
|---|---|---|---|
| Movie | maxacc | 91.4±0.2 | 90.8±0.5 |
| Movie | q70 | 90.1±0.5 | 89.6±0.4 |
| Movie | q50 | 88.6±0.5 | 88.2±1.1 |
| Movie | q30 | 87.0±0.5 | 86.3±0.7 |
| CQA | maxacc | 78.6±1.4 | 78.6±1.6 |
| CQA | q70 | 78.0±1.6 | 77.7±1.8 |
| CQA | q50 | 75.9±1.9 | 75.9±1.8 |
| CQA | q30 | 74.0±1.8 | 74.0±1.9 |

Every tier matches or slightly exceeds the reference (random-probe arm is
equivalent). The feared selection bias of learned probing does not distort
threshold placement. This closes the pipeline: **random 25 training gold +
probe-discovered validation (25 gold) + pool-quantile cost knob — 50 gold
labels total, zero oracle knowledge, on both datasets.**

## Efficiency: the learned prober barely beats random probing

Mini probes to collect 18 disagreements:

| dataset | random probing | learned prober | analytic random | artifact class rule |
|---|---|---|---|---|
| Movie | 222±24 (2/5 hit the 250 cap) | 198±43 (2/5 hit the cap) | 228 | ~101 (not a method) |
| CQA | 126±15 | 118±10 | 82 (batch granularity inflates) | — |

The learned prober saves ~11% on Movie (unstably — it either finds the
signal early and finishes in 150–170, or never does and hits the cap) and
~6% on CQA. Cause: cold start on a rare positive — the 10 seed probes yield
~1 disagreement on Movie, and by the time the scorer has enough positives to
learn from, most of the budget is spent. The 2× efficiency of the Movie
class rule is not recoverable by generic learning at this budget.

## Embedding-based probers (Movie only, added 2026-08-16)

Two training-free arms that dodge the neural scorer's rare-positive cold
start (frozen-DistilBERT mean-pooled embeddings of [text + Nano answer];
they update at negligible cost, so they probe in batches of 5 instead of 20):

| arm | probes to 18 disagreements | mechanism |
|---|---|---|
| random | 222±24 (2/5 at cap) | — |
| neural scorer | 198±43 (2/5 at cap) | retrained P(disagree) head |
| **knn propagation** | **153±28 (0/5 at cap)** | distance-weighted vote of the 5 nearest probed rows |
| cluster Thompson | 206±11 | Beta posteriors over 8 k-means clusters |

**kNN propagation is the winner among general methods**: 31% cheaper than
random, 23% cheaper than the neural scorer, never hits the probe cap, and
needs no training at all — the first found disagreement immediately
promotes its semantic neighbors. It still does not reach the
dataset-specific class rule (~101 probes): local similarity recovers part
of, not all of, that structure. Cluster Thompson fails because generic
k-means clusters on untrained embeddings do not align with where
disagreements live (8 arms with similar rates leave Thompson nothing to
exploit) — stratified probing is only as good as its strata. All arms'
validations again reproduce the reference operating points downstream.

## Recommended recipe (updated)

Drop the retrained neural prober. **Plain random probing suffices**; if
probe budget matters, **kNN propagation on frozen embeddings** cuts it ~30%
with zero training and one hyperparameter (k=5). Final budget per deployment, with nothing dataset-specific and
no oracle:

1. Nano over the candidate pool (default traffic, not incremental).
2. Random Mini probes until 18 disagreements observed (~220 Movie at a 7.9%
   disagreement rate, ~125 CQA at 22%; expected ≈ 18 / disagreement-rate).
   Agreement rows for the validation come free from probe misses.
3. 50 gold annotations: 25 random training rows + the 25 validation rows.
4. Pool-quantile threshold at the chosen budget; validation picks the epoch.

Shadow-run deployments skip step 2's cost entirely (disagreements come from
logs). The learned prober remains an option only when the disagreement rate
is very low and probe budgets are large enough to amortize its cold start.
